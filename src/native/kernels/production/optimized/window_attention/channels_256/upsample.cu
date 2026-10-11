// DLSS5 Phoenix optimized kernel: window_attention/channels_256/upsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
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
__device__ __forceinline__ void run_window_attention_channels_256_upsample(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t packed_half_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t matrix_accumulator_8 = 0;
    uint32_t matrix_accumulator_9 = 0;
    uint32_t matrix_accumulator_10 = 0;
    uint32_t matrix_accumulator_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t packed_half_16 = 0;
    uint32_t packed_half_17 = 0;
    uint32_t packed_half_18 = 0;
    uint32_t packed_half_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_fragment_24 = 0;
    uint32_t matrix_fragment_25 = 0;
    uint32_t packed_half_26 = 0;
    uint32_t matrix_fragment_27 = 0;
    uint32_t matrix_accumulator_28 = 0;
    uint32_t matrix_accumulator_29 = 0;
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
    uint32_t index_word_132 = 0;
    uint32_t address_word_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_fragment_138 = 0;
    uint32_t matrix_fragment_139 = 0;
    uint32_t packed_half_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t index_word_142 = 0;
    uint32_t address_word_143 = 0;
    uint32_t index_word_144 = 0;
    uint32_t index_word_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t index_word_147 = 0;
    uint32_t address_word_148 = 0;
    uint32_t index_word_149 = 0;
    uint32_t address_word_150 = 0;
    uint32_t index_word_151 = 0;
    uint32_t address_word_152 = 0;
    uint32_t address_word_153 = 0;
    uint32_t address_word_154 = 0;
    uint32_t address_word_155 = 0;
    uint32_t address_word_156 = 0;
    uint32_t address_word_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t address_word_164 = 0;
    uint32_t address_word_165 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_parameter_word_13 = 0;
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

    address_word_2 = threadIdx.x & 31u;

    uniform_parameter_word_21 = blockIdx.y;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_fragment_138 = threadIdx.y;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_accumulator_10 = 0;
    matrix_accumulator_11 = 0;

    uniform_index_word_10 = parameters[parameter_optional_control_10];
    uniform_index_word_11 = parameters[parameter_optional_control_11];

    matrix_fragment_36 = threadIdx.x & 31u;

    uniform_index_word_9 = blockIdx.x;

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_address_word_4 = uniform_address_word_4 + 0x1u;

    address_word_156 = parameters[parameter_trained_weights_low_4];
    address_word_157 = parameters[parameter_trained_weights_high_5];

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_address_word_4, 0u) << 0x1u >> 32) + uniform_address_word_4;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_address_word_4)) * (1ll << 0x1fu) >> 32) + 0x3u;

    uniform_index_word_8 = (uniform_parameter_word_21 << 0x3u) + uniform_index_word_11;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_10));

    uniform_address_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x1u >> 32) + uniform_index_word_8;

    index_word_151 = address_word_2;

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_address_word_7, 0u) << 0x2u >> 32) + uniform_index_word_6;

    uniform_index_word_6 = uniform_index_word_15 + 0x1u;

    uniform_index_word_9 = (uniform_index_word_9 << 0x3u) + uniform_index_word_10;

    uniform_address_word_7 = 0u - uniform_index_word_5 + uniform_index_word_15;

    packed_half_0 = uint32_t(int32_t(index_word_151) >> 0x2u);

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_9, 0u) << 0x1u >> 32) + uniform_index_word_9;

    uniform_condition_0 = uint32_t(uniform_address_word_7) != 0u;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1fu) >> 32) + 0x3u;

    packed_half_3 = uint32_t(combine_words(packed_half_0, 0u) << 0x2u >> 32) + packed_half_0;

    uniform_parameter_word_13 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    packed_half_26 = address_word_2;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    packed_half_3 = packed_half_3 & 0xfffffffcu;

    uniform_index_word_18 = uniform_index_word_14 & 0xfffffffcu;

    matrix_fragment_4 =
        uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) + uniform_index_word_12;

    uniform_index_word_15 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_6;

    matrix_fragment_27 = packed_half_0 + uniform_parameter_word_13 + (0u - packed_half_3);

    if (uniform_condition_0) {
        uniform_index_word_10 = uniform_address_word_7 + 0xffffffffu;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_index_word_19 = uniform_index_word_15 & 0xfffffffcu;

    condition_0 = int32_t(matrix_fragment_4) >= int32_t(uniform_index_word_18);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);
    }

    split_address_words(address_word_156, address_word_157,
                        uint64_t(uint32_t(matrix_fragment_138)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_156, address_word_157));

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_9) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_8;

    condition_1 = int32_t(matrix_fragment_27) >= int32_t(uniform_index_word_19);

    condition_0 = int32_t(matrix_fragment_4) > -1 && !condition_0;

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(combine_words(uniform_index_word_11, 0u) << 0x3u >> 32) +
                                uniform_index_word_10;
    }

    split_address_words(packed_half_16, packed_half_17,
                        combine_words(address_word_156, address_word_157) + 0x58200u);

    condition_1 = int32_t(matrix_fragment_27) > -1 && !condition_1;

    uniform_address_word_7 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x2u >> 32) + uniform_index_word_9;

    uniform_index_word_15 = uniform_index_word_15 << (0x2u & 31u);

    uniform_index_word_14 = uniform_index_word_14 << (0x1u & 31u);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x3u);
    }

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    condition_0 = condition_0 & condition_1;

    uniform_address_word_4 = 0u;

    packed_half_3 = uniform_index_word_12;

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_address_word_7 = uint32_t(int32_t(uniform_address_word_7) >> 0x2u);

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
            packed_half_26 =
                uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) + packed_half_3;

            matrix_fragment_27 = matrix_fragment_27 * 0x4u;

            index_word_151 = index_word_151 & 0xfffffffcu;
        }

        split_address_words(packed_half_18, packed_half_19,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, packed_half_17));

        if (condition_0) {
            matrix_fragment_14 = parameters[parameter_input_low_0];
            matrix_fragment_15 = parameters[parameter_input_high_1];

            packed_half_26 = packed_half_26 * uniform_index_word_15 + matrix_fragment_27;

            matrix_fragment_13 = packed_half_26 + address_word_2 + (0u - index_word_151);
        }

        address_word_2 = matrix_fragment_36;

        index_word_151 = address_word_2;

        packed_half_26 = address_word_2;

        packed_half_0 = uint32_t(int32_t(index_word_151) >> 0x2u);

        matrix_accumulator_20 = uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) +
                                uniform_index_word_12;

        matrix_fragment_12 =
            uint32_t(combine_words(packed_half_0, 0u) << 0x2u >> 32) + packed_half_0;

        if (condition_0) {
            split_address_words(matrix_fragment_14, matrix_fragment_15,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_14, matrix_fragment_15));
        }

        condition_3 = int32_t(matrix_accumulator_20) >= int32_t(uniform_index_word_18);

        matrix_fragment_27 = matrix_fragment_12 & 0xfffffffcu;

        matrix_fragment_12 = uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) +
                             uniform_parameter_word_20;

        matrix_fragment_27 = packed_half_0 + uniform_parameter_word_13 + (0u - matrix_fragment_27);

        condition_3 = int32_t(matrix_accumulator_20) > -1 && !condition_3;

        condition_1 = int32_t(matrix_fragment_12) >= int32_t(uniform_index_word_18);

        condition_2 = int32_t(matrix_fragment_27) >= int32_t(uniform_index_word_19);

        condition_1 = int32_t(matrix_fragment_12) > -1 && !condition_1;

        matrix_fragment_12 = 0u;

        condition_2 = int32_t(matrix_fragment_27) > -1 && !condition_2;

        condition_1 = condition_1 & condition_2;

        if (condition_0) {
            matrix_fragment_12 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
        }

        condition_0 = condition_3 & condition_2;

        if (condition_1) {
            matrix_accumulator_29 = packed_half_3 + uniform_index_word_18;

            matrix_accumulator_20 = parameters[parameter_input_low_0];
            matrix_accumulator_21 = parameters[parameter_input_high_1];

            matrix_fragment_13 =
                uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) + packed_half_3;

            matrix_accumulator_28 = matrix_fragment_27 * 0x4u;

            matrix_accumulator_32 =
                uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) +
                matrix_accumulator_29;

            matrix_fragment_31 = matrix_fragment_27 * 0x4u;
        }

        if (condition_0) {
            matrix_accumulator_29 = packed_half_3 + uniform_index_word_18;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + 0x2u;

            matrix_fragment_24 = parameters[parameter_input_low_0];
            matrix_fragment_25 = parameters[parameter_input_high_1];

            matrix_fragment_14 = matrix_accumulator_32 + 0x2u;
        }

        if (condition_0) {
            matrix_fragment_30 = matrix_fragment_27 * 0x4u;

            matrix_accumulator_29 =
                uint32_t(int64_t(int32_t(packed_half_26)) * (1ll << 0x1cu) >> 32) +
                matrix_accumulator_29;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 * uniform_index_word_15 + matrix_accumulator_28;
        }

        if (condition_0) {
            matrix_accumulator_28 = index_word_151 & 0xfffffffcu;
        }

        if (condition_1) {
            matrix_fragment_15 = matrix_fragment_14 * uniform_index_word_15 + matrix_fragment_31;

            matrix_fragment_14 = index_word_151 & 0xfffffffcu;
        }

        if (condition_0) {
            matrix_accumulator_22 = parameters[parameter_input_low_0];
            matrix_accumulator_23 = parameters[parameter_input_high_1];

            matrix_accumulator_29 =
                matrix_accumulator_29 * uniform_index_word_15 + matrix_fragment_30;
        }

        if (condition_1) {
            matrix_fragment_30 = index_word_151 & 0xfffffffcu;
        }

        load_four_words_cached<CheckBounds>(combine_words(packed_half_18, packed_half_19),
                                            matrix_accumulator_32, matrix_accumulator_33,
                                            matrix_fragment_34, matrix_fragment_35, memory_ranges,
                                            error_counts);

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + address_word_2 + (0u - matrix_fragment_14);

            matrix_fragment_15 = matrix_fragment_15 + address_word_2 + (0u - matrix_fragment_30);
        }

        if (condition_0) {
            matrix_accumulator_29 =
                matrix_accumulator_29 + address_word_2 + (0u - matrix_accumulator_28);
        }

        if (condition_1) {
            split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_20, matrix_accumulator_21));
        }

        matrix_fragment_13 = 0u;

        if (condition_1) {
            split_address_words(matrix_fragment_24, matrix_fragment_25,
                                int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_24, matrix_fragment_25));
        }

        matrix_fragment_14 = 0;
        matrix_fragment_15 = 0;

        if (condition_1) {
            matrix_fragment_13 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_20, matrix_accumulator_21), memory_ranges,
                error_counts);

            matrix_fragment_15 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25), memory_ranges, error_counts);
        }

        if (condition_0) {
            split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                                int64_t(int32_t(matrix_accumulator_29)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_22, matrix_accumulator_23));
        }

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_18, packed_half_19) - 0x200ull, matrix_accumulator_28,
            matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);

        if (condition_0) {
            matrix_fragment_14 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_22, matrix_accumulator_23), memory_ranges,
                error_counts);
        }

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x1e0u);

        uniform_index_word_10 = uniform_index_word_5 + 0x20u;

        split_address_words(packed_half_16, packed_half_17,
                            combine_words(packed_half_16, packed_half_17) + 0x2000u);

        packed_half_3 = packed_half_3 + uniform_index_word_14;

        uniform_index_word_5 = uniform_index_word_10;

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_28, matrix_accumulator_29,
                                 matrix_accumulator_10, matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                                 matrix_fragment_5);
    } while (!uniform_condition_0);
    uniform_index_word_14 = parameters[parameter_tensor_height_8];
    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_fragment_12 = uint32_t(uint32_t(address_word_2) >> 0x1u);

    packed_half_3 = address_word_2 * 0x2u;

    matrix_fragment_15 = index_word_151 & 0x7ffffffcu;

    uniform_condition_5 = false;

    matrix_fragment_13 = matrix_fragment_12 & 0x4u;

    address_word_152 = parameters[parameter_trained_weights_low_4];
    address_word_153 = parameters[parameter_trained_weights_high_5];

    packed_half_3 = packed_half_3 & 0x8u;

    matrix_fragment_14 = matrix_fragment_13 | 0x13u & address_word_2;

    matrix_fragment_12 = packed_half_3 | 0x13u & address_word_2;

    packed_half_16 = 0u - matrix_fragment_15 + address_word_2;

    matrix_fragment_14 = matrix_fragment_14 | packed_half_3;

    packed_half_3 = matrix_fragment_138 * 0x20u;

    matrix_fragment_13 = matrix_fragment_12 | matrix_fragment_13;

    uniform_index_word_10 = uniform_index_word_14 & 0xfffffffcu;

    matrix_fragment_12 = address_word_2;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);

    packed_half_18 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x8u;

    matrix_fragment_44 = packed_half_16 * 0x2u + packed_half_3;

    uniform_condition_3 = uint32_t(uniform_index_word_10) != uint32_t(0x4u);

    packed_half_26 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x38u;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_15) >> 0x1fu);

    packed_half_3 = uint32_t(uint32_t(packed_half_3) >> 0x1u);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_14;

    packed_half_19 = uint32_t(int32_t(matrix_fragment_12) >> 0x6u);

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_15;

    matrix_fragment_25 = uint32_t(int32_t(packed_half_18) >> 0x1fu);

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    matrix_accumulator_21 = uint32_t(int32_t(packed_half_26) >> 0x1fu);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    matrix_fragment_15 = packed_half_3 + 0x4u;

    packed_half_17 = packed_half_3 + 0x8u;

    matrix_accumulator_141 = packed_half_3 + 0xcu;

    packed_half_3 = uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1u >> 32) + packed_half_19;

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);
    }

    packed_half_16 = index_word_151 & 0xfffffffcu;

    uniform_condition_1 = int32_t(uniform_address_word_7) >= int32_t(uniform_index_word_11);

    matrix_fragment_50 =
        uint32_t(combine_words(matrix_fragment_25, 0u) << 0x4u >> 32) + packed_half_18;

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_0;
    }

    matrix_fragment_12 =
        uint32_t(combine_words(matrix_accumulator_21, 0u) << 0x4u >> 32) + packed_half_26;

    uniform_index_word_18 = uniform_index_word_15 & 0xfffffffcu;

    matrix_accumulator_22 = packed_half_3 & 0x3ffffffeu;

    uniform_condition_0 = true;

    packed_half_161 = matrix_fragment_15 + address_word_2 + (0u - packed_half_16);

    uniform_condition_1 =
        int32_t(uniform_index_word_9) > int32_t(0u - 0x4u) && !uniform_condition_1;

    packed_half_3 = matrix_fragment_50 & 0xfff0u;

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    matrix_fragment_15 = matrix_fragment_12 & 0xfff0u;

    uniform_condition_2 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    matrix_fragment_25 =
        uint32_t(combine_words(matrix_fragment_25, 0u) << 0x5u >> 32) + packed_half_18;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_10), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    packed_half_18 = packed_half_18 + (0u - packed_half_3);

    matrix_fragment_24 =
        uint32_t(combine_words(matrix_accumulator_21, 0u) << 0x5u >> 32) + packed_half_26;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_2 | uniform_condition_1);

    packed_half_26 = packed_half_26 + (0u - matrix_fragment_15);

    packed_half_3 = 0x8u ^ matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_9), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    matrix_fragment_15 = 0x10u ^ matrix_fragment_13;

    uniform_condition_4 = uint32_t(uniform_index_word_18) != uint32_t(0x4u);

    matrix_fragment_14 = 0x18u ^ matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_11), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_20 = result_word;
    }

    address_word_159 = packed_half_17 + address_word_2 + (0u - packed_half_16);

    uniform_index_word_12 = uniform_condition_5 ? uniform_address_word_7 : 0u;

    matrix_accumulator_141 = matrix_accumulator_141 + address_word_2 + (0u - packed_half_16);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_10), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        packed_half_17 = result_word;
    }

    matrix_accumulator_22 = packed_half_19 + (0u - matrix_accumulator_22);

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_10), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        packed_half_16 = result_word;
    }

    uniform_index_word_12 = !uniform_condition_4 ? uniform_index_word_12 : uniform_address_word_7;

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    condition_0 = uniform_condition_0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_10), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        packed_half_19 = result_word;
    }

    address_word_150 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x10u;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    condition_2 = (address_word_2 & 0x10u) != 0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_11), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_23 = result_word;
    }

    index_word_147 = uint32_t(int32_t(address_word_150) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_9), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_32 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_9), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_9), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_8), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_8), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_8), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_27 = result_word;
    }

    matrix_fragment_51 = !condition_2 ? matrix_accumulator_20 : matrix_accumulator_23;

    matrix_fragment_53 = !condition_2 ? matrix_accumulator_23 : matrix_accumulator_20;

    matrix_accumulator_20 =
        uint32_t(combine_words(index_word_147, 0u) << 0x4u >> 32) + address_word_150;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_8), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_30 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_11), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_11), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_62 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    uniform_index_word_12 = uniform_address_word_7 + 0x1u;

    if (uniform_condition_3) {
        uniform_condition_5 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_36 = !condition_2 ? matrix_fragment_46 : packed_half_17;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_9 = result_word;
    }

    matrix_fragment_46 = !condition_2 ? packed_half_17 : matrix_fragment_46;

    packed_half_17 = uint32_t(int32_t(matrix_accumulator_20) >> 0x4u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_8 = result_word;
    }

    uniform_condition_6 = false;

    index_word_144 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x20u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(packed_half_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_11 = result_word;
    }

    address_word_148 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_56 = !condition_2 ? matrix_accumulator_29 : matrix_accumulator_10;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    matrix_fragment_68 = !condition_2 ? matrix_accumulator_10 : matrix_accumulator_29;

    index_word_142 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x28u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_33 = result_word;
    }

    index_word_132 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x30u;

    packed_half_3 = uniform_index_word_5;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_60 = !condition_2 ? matrix_fragment_62 : matrix_fragment_55;

    matrix_fragment_39 = !condition_2 ? packed_half_16 : packed_half_19;

    if (condition_0) {
        packed_half_3 = packed_half_3 * 0x8u + matrix_fragment_138;
    }

    matrix_fragment_74 = !condition_2 ? matrix_fragment_72 : matrix_accumulator_9;

    matrix_fragment_72 = !condition_2 ? matrix_accumulator_9 : matrix_fragment_72;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    matrix_fragment_69 = !condition_2 ? matrix_accumulator_8 : matrix_fragment_35;

    matrix_fragment_15 = !condition_2 ? matrix_fragment_35 : matrix_accumulator_8;

    split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (condition_0) {
        packed_half_3 = packed_half_3 * 0x80u;
    }

    uniform_condition_2 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_11);

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_5;
    }

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    if (condition_0) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_accumulator_20, 0u) << 0x1u >> 32) + packed_half_17;

    uniform_condition_0 = true;

    address_word_143 = uint32_t(int32_t(index_word_144) >> 0x1fu);

    uniform_condition_2 =
        int32_t(uniform_index_word_9) > int32_t(0u - 0x8u) && !uniform_condition_2;

    matrix_fragment_65 = !condition_2 ? matrix_accumulator_11 : matrix_fragment_34;

    matrix_fragment_67 = !condition_2 ? matrix_fragment_34 : matrix_accumulator_11;

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    split_address_words(matrix_accumulator_10, matrix_accumulator_11,
                        uint64_t(uint32_t(matrix_accumulator_141)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    matrix_fragment_34 = matrix_fragment_4 & 0x3ffffffeu;

    uniform_condition_5 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_2);

    index_word_145 = uint32_t(int32_t(address_word_148) >> 0x1fu);

    if (condition_0) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    matrix_fragment_57 =
        uint32_t(combine_words(address_word_143, 0u) << 0x4u >> 32) + index_word_144;

    matrix_accumulator_23 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_10, matrix_accumulator_11) + 0x78200ull, memory_ranges,
        error_counts);

    if (!condition_0) {
        matrix_fragment_4 = uniform_index_word_5;

        matrix_fragment_5 = uniform_index_word_5;

        matrix_fragment_7 = uniform_index_word_5;

        matrix_fragment_6 = uniform_index_word_5;
    }

    matrix_fragment_61 =
        uint32_t(combine_words(index_word_145, 0u) << 0x4u >> 32) + address_word_148;

    uniform_parameter_word_13 = uniform_condition_6 ? uniform_index_word_12 : 0u;

    address_word_133 = uint32_t(int32_t(index_word_142) >> 0x1fu);

    if (uniform_condition_5) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_fragment_62 = !condition_2 ? matrix_fragment_55 : matrix_fragment_62;

    matrix_accumulator_10 = uint32_t(int32_t(matrix_fragment_57) >> 0x4u);

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_8, matrix_accumulator_9), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11;
    }

    uniform_parameter_word_13 =
        !uniform_condition_4 ? uniform_parameter_word_13 : uniform_index_word_12;

    matrix_accumulator_11 =
        uint32_t(combine_words(matrix_fragment_57, 0u) << 0x1u >> 32) + matrix_accumulator_10;

    packed_half_3 = uint32_t(int32_t(index_word_132) >> 0x1fu);

    matrix_fragment_55 =
        uint32_t(combine_words(address_word_133, 0u) << 0x4u >> 32) + index_word_142;

    matrix_accumulator_8 = uint32_t(int32_t(matrix_fragment_61) >> 0x4u);

    if (uniform_condition_5) {
        uniform_index_word_5 = uniform_parameter_word_13 + uniform_index_word_5;
    }

    matrix_fragment_41 = !condition_2 ? packed_half_19 : packed_half_16;

    matrix_accumulator_11 = matrix_accumulator_11 & 0x3ffffffeu;

    matrix_accumulator_9 =
        uint32_t(combine_words(matrix_fragment_61, 0u) << 0x1u >> 32) + matrix_accumulator_8;

    matrix_fragment_14 = !condition_2 ? matrix_fragment_27 : matrix_fragment_30;

    matrix_fragment_52 = !condition_2 ? matrix_fragment_30 : matrix_fragment_27;

    packed_half_19 = uint32_t(combine_words(packed_half_3, 0u) << 0x4u >> 32) + index_word_132;

    matrix_fragment_30 = uint32_t(int32_t(matrix_fragment_55) >> 0x4u);

    matrix_fragment_43 = !condition_2 ? matrix_fragment_45 : matrix_accumulator_32;

    matrix_fragment_54 = !condition_2 ? matrix_fragment_58 : matrix_accumulator_33;

    condition_1 = uniform_condition_5;

    matrix_fragment_45 = !condition_2 ? matrix_accumulator_32 : matrix_fragment_45;

    matrix_fragment_58 = !condition_2 ? matrix_accumulator_33 : matrix_fragment_58;

    matrix_accumulator_9 = matrix_accumulator_9 & 0x3ffffffeu;

    matrix_accumulator_32 = uint32_t(int32_t(packed_half_19) >> 0x4u);

    matrix_accumulator_33 = matrix_accumulator_10 + (0u - matrix_accumulator_11);

    matrix_fragment_42 = !condition_2 ? matrix_fragment_48 : matrix_accumulator_21;

    packed_half_16 =
        uint32_t(combine_words(matrix_fragment_55, 0u) << 0x1u >> 32) + matrix_fragment_30;

    matrix_accumulator_11 = uniform_index_word_5;

    matrix_fragment_48 = !condition_2 ? matrix_accumulator_21 : matrix_fragment_48;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 + 0x1u;
    }

    matrix_fragment_59 = !condition_2 ? matrix_fragment_37 : matrix_fragment_40;

    matrix_accumulator_21 = !condition_2 ? matrix_fragment_40 : matrix_fragment_37;

    matrix_fragment_34 = packed_half_17 + (0u - matrix_fragment_34);

    split_address_words(packed_half_160, packed_half_161,
                        uint64_t(uint32_t(packed_half_161)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    packed_half_17 =
        uint32_t(combine_words(packed_half_19, 0u) << 0x1u >> 32) + matrix_accumulator_32;

    matrix_fragment_37 = matrix_accumulator_8 + (0u - matrix_accumulator_9);

    matrix_accumulator_9 = packed_half_16 & 0x3ffffffeu;

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_49 = !condition_2 ? matrix_fragment_47 : matrix_accumulator_28;

    matrix_fragment_47 = !condition_2 ? matrix_accumulator_28 : matrix_fragment_47;

    packed_half_17 = packed_half_17 & 0x3ffffffeu;

    matrix_accumulator_28 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x78200ull, memory_ranges, error_counts);

    address_word_163 = uint32_t(int32_t(matrix_fragment_44) >> 0x1u);

    if (!uniform_condition_5) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    uniform_condition_6 = false;

    matrix_fragment_30 = matrix_fragment_30 + (0u - matrix_accumulator_9);

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    matrix_fragment_66 = !condition_2 ? matrix_fragment_64 : matrix_fragment_31;

    if (condition_1) {
        matrix_accumulator_9 = matrix_accumulator_11 * 0x8u + matrix_fragment_138;
    }

    matrix_fragment_64 = !condition_2 ? matrix_fragment_31 : matrix_fragment_64;

    matrix_fragment_31 = matrix_accumulator_32 + (0u - packed_half_17);

    uniform_condition_0 = true;

    if (!uniform_condition_5) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    split_address_words(address_word_162, address_word_163,
                        int64_t(int32_t(address_word_163)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    split_address_words(packed_half_16, packed_half_17,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    split_address_words(packed_half_158, address_word_159,
                        uint64_t(uint32_t(address_word_159)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_27 = load_aligned_word<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x78200ull, memory_ranges,
        error_counts);

    index_word_149 = uint32_t(int32_t(index_word_151) >> 0x1fu);

    if (condition_1) {
        matrix_accumulator_9 = matrix_accumulator_9 * 0x80u;
    }

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_1);

    matrix_accumulator_29 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_158, address_word_159) + 0x78200ull, memory_ranges, error_counts);

    matrix_fragment_71 = uint32_t(int32_t(matrix_fragment_12) >> 0x4u);

    if (condition_1) {
        split_address_words(packed_half_16, packed_half_17,
                            int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_16, packed_half_17));
    }

    if (!condition_1) {
        matrix_accumulator_8 = uniform_index_word_5;

        matrix_accumulator_9 = uniform_index_word_5;

        matrix_accumulator_10 = uniform_index_word_5;

        matrix_accumulator_11 = uniform_index_word_5;
    }

    uniform_parameter_word_13 = uniform_condition_6 ? uniform_address_word_7 : 0u;

    if (condition_1) {
        split_address_words(packed_half_16, packed_half_17,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, packed_half_17));
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;
    }

    uniform_parameter_word_13 =
        !uniform_condition_4 ? uniform_parameter_word_13 : uniform_address_word_7;

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(combine_words(packed_half_16, packed_half_17),
                                                   matrix_accumulator_8, matrix_accumulator_9,
                                                   matrix_accumulator_10, matrix_accumulator_11,
                                                   memory_ranges, error_counts);
    }

    condition_1 = uniform_condition_0;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_parameter_word_13 + uniform_index_word_5;
    }

    condition_0 = (address_word_2 & 0x4u) != 0;

    matrix_fragment_13 = !condition_2 ? matrix_fragment_63 : matrix_fragment_38;

    matrix_fragment_12 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1u >> 32) + matrix_fragment_71;

    matrix_fragment_35 = uint32_t(int32_t(matrix_fragment_50) >> 0x4u);

    packed_half_17 = uniform_index_word_5;

    matrix_fragment_40 = !condition_0 ? matrix_fragment_36 : matrix_fragment_39;

    matrix_fragment_36 = !condition_0 ? matrix_fragment_39 : matrix_fragment_36;

    matrix_fragment_44 = !condition_0 ? matrix_fragment_48 : matrix_fragment_53;

    matrix_fragment_39 = !condition_0 ? matrix_fragment_43 : matrix_fragment_56;

    matrix_fragment_48 = !condition_0 ? matrix_fragment_53 : matrix_fragment_48;

    matrix_fragment_43 = !condition_0 ? matrix_fragment_56 : matrix_fragment_43;

    matrix_fragment_78 = !condition_0 ? matrix_fragment_66 : matrix_fragment_65;

    if (condition_1) {
        packed_half_17 = packed_half_17 * 0x8u + matrix_fragment_138;
    }

    matrix_fragment_53 = !condition_0 ? matrix_fragment_49 : matrix_fragment_14;

    matrix_fragment_66 = !condition_0 ? matrix_fragment_65 : matrix_fragment_66;

    matrix_fragment_56 = !condition_0 ? matrix_fragment_58 : matrix_fragment_15;

    matrix_fragment_49 = !condition_0 ? matrix_fragment_14 : matrix_fragment_49;

    matrix_fragment_58 = !condition_0 ? matrix_fragment_15 : matrix_fragment_58;

    matrix_fragment_65 = uint32_t(combine_words(index_word_149, 0u) << 0x4u >> 32) + packed_half_0;

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (condition_1) {
        packed_half_17 = packed_half_17 * 0x80u;
    }

    if (uniform_condition_3) {
        uniform_parameter_word_13 = uniform_index_word_6 + 0x1u;
    }

    matrix_fragment_63 = !condition_2 ? matrix_fragment_38 : matrix_fragment_63;

    matrix_fragment_65 = matrix_fragment_65 & 0xfff0u;

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_12 = matrix_fragment_12 & 0x3ffffffeu;

    if (condition_1) {
        split_address_words(packed_half_16, packed_half_17,
                            int64_t(int32_t(packed_half_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_parameter_word_13) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_80 = !condition_0 ? matrix_fragment_72 : matrix_fragment_63;

    matrix_fragment_82 = !condition_0 ? matrix_fragment_74 : matrix_fragment_13;

    matrix_fragment_72 = !condition_0 ? matrix_fragment_63 : matrix_fragment_72;

    matrix_fragment_65 = packed_half_0 + (0u - matrix_fragment_65);

    uniform_condition_6 = false;

    matrix_accumulator_32 = matrix_fragment_71 + (0u - matrix_fragment_12);

    matrix_fragment_74 = !condition_0 ? matrix_fragment_13 : matrix_fragment_74;

    matrix_fragment_63 = matrix_accumulator_20 & 0xfff0u;

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    if (!condition_1) {
        matrix_fragment_12 = uniform_index_word_5;

        matrix_fragment_13 = uniform_index_word_5;

        matrix_fragment_14 = uniform_index_word_5;

        matrix_fragment_15 = uniform_index_word_5;
    }

    if (condition_1) {
        split_address_words(packed_half_16, packed_half_17,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, packed_half_17));
    }

    uniform_condition_0 = true;

    matrix_accumulator_20 = permute_signed(matrix_fragment_65, 0u, 0x8880u);

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    matrix_fragment_71 = address_word_150 + (0u - matrix_fragment_63);

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(packed_half_16, packed_half_17), matrix_fragment_12, matrix_fragment_13,
            matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);
    }

    matrix_fragment_50 =
        uint32_t(combine_words(matrix_fragment_50, 0u) << 0x1u >> 32) + matrix_fragment_35;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_2);

    matrix_fragment_38 = !condition_0 ? matrix_fragment_42 : matrix_fragment_51;

    packed_half_16 = permute_signed(matrix_accumulator_20, 0u, 0x7710u);

    packed_half_17 = permute_signed(packed_half_18, 0u, 0x8880u);

    uniform_index_word_5 = 0u;

    matrix_accumulator_20 = permute_signed(matrix_fragment_71, 0u, 0x8880u);

    if (!uniform_condition_4) {
        uniform_index_word_12 = uniform_condition_6 ? uniform_index_word_12 : 0u;
    }

    packed_half_16 = uint32_t(uint32_t(packed_half_16) >> 0xdu);

    if (uniform_condition_0) {
        uniform_index_word_18 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_index_word_19 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    packed_half_17 = permute_signed(packed_half_17, 0u, 0x7710u);

    matrix_accumulator_20 = permute_signed(matrix_accumulator_20, 0u, 0x7710u);

    matrix_fragment_42 = !condition_0 ? matrix_fragment_51 : matrix_fragment_42;

    packed_half_16 = packed_half_16 & 0x3u;

    matrix_fragment_51 = !condition_0 ? matrix_fragment_47 : matrix_fragment_52;

    matrix_fragment_47 = !condition_0 ? matrix_fragment_52 : matrix_fragment_47;

    packed_half_17 = uint32_t(uint32_t(packed_half_17) >> 0xdu);

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;
    }

    matrix_accumulator_20 = uint32_t(uint32_t(matrix_accumulator_20) >> 0xdu);

    matrix_fragment_52 = !condition_0 ? matrix_fragment_54 : matrix_fragment_69;

    matrix_fragment_54 = !condition_0 ? matrix_fragment_69 : matrix_fragment_54;

    condition_1 = uniform_condition_0;

    matrix_fragment_69 = matrix_fragment_65 + packed_half_16;

    packed_half_17 = packed_half_17 & 0x3u;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_accumulator_20 = matrix_accumulator_20 & 0x3u;

    matrix_fragment_50 = matrix_fragment_50 & 0x3ffffffeu;

    matrix_fragment_76 = !condition_0 ? matrix_fragment_64 : matrix_fragment_67;

    matrix_fragment_70 = !condition_0 ? matrix_fragment_60 : matrix_fragment_59;

    matrix_fragment_64 = !condition_0 ? matrix_fragment_67 : matrix_fragment_64;

    matrix_fragment_60 = !condition_0 ? matrix_fragment_59 : matrix_fragment_60;

    packed_half_16 = matrix_fragment_69 & 0xfcu;

    matrix_fragment_67 = packed_half_18 + packed_half_17;

    matrix_fragment_59 = matrix_fragment_71 + matrix_accumulator_20;

    matrix_fragment_63 = uniform_index_word_5;

    matrix_fragment_35 = matrix_fragment_35 + (0u - matrix_fragment_50);

    matrix_fragment_50 = !condition_0 ? matrix_fragment_46 : matrix_fragment_41;

    matrix_fragment_84 = 0u - packed_half_16;

    matrix_fragment_46 = !condition_0 ? matrix_fragment_41 : matrix_fragment_46;

    packed_half_17 = matrix_fragment_67 & 0xfcu;

    packed_half_16 = matrix_fragment_59 & 0xfcu;

    if (condition_1) {
        matrix_fragment_63 = matrix_fragment_63 * 0x8u + matrix_fragment_138;
    }

    matrix_fragment_41 = !condition_0 ? matrix_fragment_45 : matrix_fragment_68;

    matrix_fragment_45 = !condition_0 ? matrix_fragment_68 : matrix_fragment_45;

    matrix_fragment_68 = !condition_0 ? matrix_fragment_62 : matrix_accumulator_21;

    matrix_fragment_62 = !condition_0 ? matrix_accumulator_21 : matrix_fragment_62;

    matrix_fragment_73 = 0u - packed_half_17;

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        combine_words(uniform_index_word_18, uniform_index_word_19));

    packed_half_16 = 0u - packed_half_16;

    if (condition_1) {
        packed_half_17 = matrix_fragment_63 * 0x80u;
    }

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_84 = permute_signed(matrix_fragment_84, 0u, 0x7710u);

    if (condition_1) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(packed_half_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));
    }

    packed_half_16 = permute_signed(packed_half_16, 0u, 0x7710u);

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_73 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    packed_half_17 = matrix_fragment_57 & 0xfff0u;

    matrix_fragment_63 = matrix_fragment_65 + matrix_fragment_84;

    matrix_fragment_57 = matrix_fragment_71 + packed_half_16;

    matrix_fragment_65 = packed_half_18 + matrix_fragment_73;

    matrix_fragment_71 = packed_half_19 & 0xfff0u;

    matrix_fragment_75 = index_word_144 + (0u - packed_half_17);

    if (!condition_1) {
        packed_half_16 = uniform_index_word_5;

        packed_half_17 = uniform_index_word_5;

        packed_half_18 = uniform_index_word_5;

        packed_half_19 = uniform_index_word_5;
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));

        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_20, matrix_accumulator_21), packed_half_16,
            packed_half_17, packed_half_18, packed_half_19, memory_ranges, error_counts);
    }

    matrix_fragment_61 = matrix_fragment_61 & 0xfff0u;

    matrix_fragment_55 = matrix_fragment_55 & 0xfff0u;

    matrix_fragment_61 = address_word_148 + (0u - matrix_fragment_61);

    matrix_fragment_81 = index_word_142 + (0u - matrix_fragment_55);

    matrix_fragment_73 = permute_signed(matrix_fragment_61, 0u, 0x8880u);

    matrix_fragment_85 = index_word_132 + (0u - matrix_fragment_71);

    matrix_fragment_71 = permute_signed(matrix_fragment_75, 0u, 0x8880u);

    matrix_fragment_55 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    matrix_fragment_73 = permute_signed(matrix_fragment_81, 0u, 0x8880u);

    matrix_fragment_55 = uint32_t(uint32_t(matrix_fragment_55) >> 0xdu);

    matrix_fragment_71 = permute_signed(matrix_fragment_71, 0u, 0x7710u);

    matrix_fragment_77 = permute_signed(matrix_fragment_85, 0u, 0x8880u);

    matrix_fragment_73 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    matrix_fragment_84 = matrix_fragment_55 & 0x3u;

    matrix_fragment_71 = uint32_t(uint32_t(matrix_fragment_71) >> 0xdu);

    matrix_fragment_55 = permute_signed(matrix_fragment_77, 0u, 0x7710u);

    matrix_fragment_79 = permute_signed(packed_half_26, 0u, 0x8880u);

    matrix_fragment_73 = uint32_t(uint32_t(matrix_fragment_73) >> 0xdu);

    matrix_accumulator_20 = matrix_fragment_71 & 0x3u;

    matrix_fragment_55 = uint32_t(uint32_t(matrix_fragment_55) >> 0xdu);

    matrix_fragment_77 = matrix_fragment_61 + matrix_fragment_84;

    matrix_accumulator_21 = permute_signed(matrix_fragment_79, 0u, 0x7710u);

    matrix_fragment_84 = matrix_fragment_73 & 0x3u;

    matrix_fragment_86 = matrix_fragment_55 & 0x3u;

    matrix_fragment_79 = matrix_fragment_75 + matrix_accumulator_20;

    matrix_fragment_55 = uint32_t(uint32_t(matrix_accumulator_21) >> 0xdu);

    matrix_accumulator_20 = matrix_fragment_77 & 0xfcu;

    matrix_fragment_83 = matrix_fragment_81 + matrix_fragment_84;

    matrix_fragment_55 = matrix_fragment_55 & 0x3u;

    matrix_accumulator_21 = matrix_fragment_79 & 0xfcu;

    matrix_fragment_84 = 0u - matrix_accumulator_20;

    matrix_accumulator_20 = matrix_fragment_83 & 0xfcu;

    matrix_fragment_73 = packed_half_26 + matrix_fragment_55;

    matrix_fragment_71 = matrix_fragment_85 + matrix_fragment_86;

    matrix_fragment_86 = 0u - matrix_accumulator_21;

    matrix_fragment_84 = permute_signed(matrix_fragment_84, 0u, 0x7710u);

    matrix_fragment_55 = 0u - matrix_accumulator_20;

    matrix_accumulator_20 = matrix_fragment_73 & 0xfcu;

    matrix_fragment_86 = permute_signed(matrix_fragment_86, 0u, 0x7710u);

    matrix_fragment_61 = matrix_fragment_61 + matrix_fragment_84;

    matrix_fragment_84 = permute_signed(matrix_fragment_55, 0u, 0x7710u);

    matrix_fragment_87 = 0u - matrix_accumulator_20;

    matrix_fragment_55 = matrix_fragment_75 + matrix_fragment_86;

    matrix_fragment_75 = permute_signed(matrix_fragment_69, 0u, 0x8880u);

    matrix_accumulator_20 = matrix_fragment_81 + matrix_fragment_84;

    matrix_fragment_84 = permute_signed(matrix_fragment_67, 0u, 0x8880u);

    matrix_fragment_81 = permute_signed(matrix_fragment_87, 0u, 0x7710u);

    matrix_fragment_67 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

    matrix_fragment_69 = address_word_2;

    matrix_fragment_75 = permute_signed(matrix_fragment_59, 0u, 0x8880u);

    matrix_fragment_59 = matrix_fragment_84;

    packed_half_26 = packed_half_26 + matrix_fragment_81;

    matrix_fragment_81 = permute_signed(matrix_fragment_67, 0u, 0x9910u);

    matrix_fragment_69 = uint32_t(int32_t(matrix_fragment_69) >> 0x7u);

    matrix_accumulator_21 = matrix_fragment_71 & 0xfcu;

    matrix_fragment_67 = uint32_t(int32_t(matrix_fragment_59) >> 0x2u);

    matrix_fragment_84 = (matrix_fragment_69 << 0x2u) + uniform_index_word_8;

    matrix_fragment_59 = matrix_fragment_81;

    matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_25) >> 0x5u);

    matrix_fragment_67 = permute_signed(matrix_fragment_67, 0u, 0x9910u);

    matrix_accumulator_21 = 0u - matrix_accumulator_21;

    matrix_fragment_59 = matrix_fragment_59 + matrix_fragment_84;

    matrix_fragment_69 =
        uint32_t(combine_words(index_word_147, 0u) << 0x5u >> 32) + address_word_150;

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

    matrix_fragment_84 = (matrix_fragment_25 << 0x2u) + uniform_index_word_8;

    matrix_fragment_25 = matrix_fragment_67;

    matrix_fragment_77 = permute_signed(matrix_fragment_77, 0u, 0x8880u);

    matrix_fragment_86 = permute_signed(matrix_accumulator_21, 0u, 0x7710u);

    matrix_fragment_69 = uint32_t(int32_t(matrix_fragment_69) >> 0x5u);

    matrix_fragment_75 = permute_signed(matrix_fragment_75, 0u, 0x9910u);

    condition_3 = int32_t(matrix_fragment_59) >= int32_t(0u);

    condition_4 = int32_t(matrix_fragment_59) >= int32_t(uniform_index_word_14);

    matrix_fragment_84 = matrix_fragment_25 + matrix_fragment_84;

    matrix_fragment_59 = matrix_fragment_77;

    matrix_accumulator_21 = matrix_fragment_85 + matrix_fragment_86;

    matrix_fragment_25 =
        uint32_t(combine_words(index_word_145, 0u) << 0x5u >> 32) + address_word_148;

    matrix_fragment_86 = (matrix_fragment_69 << 0x2u) + uniform_index_word_8;

    matrix_fragment_69 = matrix_fragment_75;

    matrix_fragment_59 = uint32_t(int32_t(matrix_fragment_59) >> 0x2u);

    matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_25) >> 0x5u);

    matrix_fragment_67 = permute_signed(matrix_fragment_79, 0u, 0x8880u);

    matrix_fragment_77 = permute_signed(matrix_fragment_83, 0u, 0x8880u);

    matrix_fragment_69 = matrix_fragment_69 + matrix_fragment_86;

    matrix_fragment_79 = permute_signed(matrix_fragment_59, 0u, 0x9910u);

    matrix_fragment_86 = (matrix_fragment_25 << 0x2u) + uniform_index_word_8;

    matrix_fragment_59 =
        uint32_t(combine_words(address_word_133, 0u) << 0x5u >> 32) + index_word_142;

    matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_67) >> 0x2u);

    matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_77) >> 0x2u);

    matrix_fragment_59 = uint32_t(int32_t(matrix_fragment_59) >> 0x5u);

    matrix_fragment_25 = permute_signed(matrix_fragment_25, 0u, 0x9910u);

    matrix_fragment_77 = permute_signed(matrix_fragment_77, 0u, 0x9910u);

    matrix_fragment_88 = (matrix_fragment_59 << 0x2u) + uniform_index_word_8;

    matrix_fragment_59 = matrix_fragment_25;

    matrix_fragment_75 =
        uint32_t(combine_words(address_word_143, 0u) << 0x5u >> 32) + index_word_144;

    matrix_fragment_25 = matrix_fragment_77;

    matrix_fragment_77 = permute_signed(matrix_fragment_73, 0u, 0x8880u);

    matrix_fragment_67 = matrix_fragment_79;

    matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_24) >> 0x5u);

    matrix_fragment_63 = matrix_fragment_63 & 0xffffu;

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_75) >> 0x5u);

    matrix_fragment_25 = matrix_fragment_25 + matrix_fragment_88;

    matrix_fragment_88 = (matrix_fragment_73 << 0x2u) + uniform_index_word_8;

    matrix_fragment_65 = matrix_fragment_65 & 0xffffu;

    matrix_fragment_67 = matrix_fragment_67 + matrix_fragment_86;

    matrix_fragment_73 = permute_signed(matrix_fragment_63, 0u, 0x8880u);

    matrix_fragment_86 = (matrix_fragment_75 << 0x2u) + uniform_index_word_8;

    matrix_fragment_65 = permute_signed(matrix_fragment_65, 0u, 0x8880u);

    matrix_fragment_63 = (matrix_accumulator_22 << 0x2u) + uniform_index_word_9;

    matrix_accumulator_22 = matrix_fragment_73;

    matrix_fragment_59 = matrix_fragment_59 + matrix_fragment_86;

    matrix_fragment_86 = (matrix_fragment_35 << 0x2u) + uniform_index_word_9;

    matrix_fragment_35 = matrix_fragment_65;

    matrix_fragment_65 = matrix_accumulator_22 + matrix_fragment_63;

    matrix_fragment_86 = matrix_fragment_35 + matrix_fragment_86;

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_4);

    condition_5 = int32_t(matrix_fragment_65) >= int32_t(uniform_index_word_15);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_63 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    condition_6 = int32_t(matrix_fragment_65) < int32_t(0u) || condition_5;

    matrix_fragment_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_4 = !condition_6 ? matrix_fragment_4 : 0u;

    matrix_fragment_35 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_6 = !condition_6 ? matrix_fragment_6 : 0u;

    matrix_fragment_53 = !condition_4 ? matrix_fragment_4 : 0u;

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_57 = matrix_fragment_57 & 0xffffu;

    matrix_fragment_35 = !condition_6 ? matrix_fragment_35 : 0u;

    matrix_fragment_63 = !condition_6 ? matrix_fragment_63 : 0u;

    matrix_fragment_65 = !condition_4 ? matrix_fragment_6 : 0u;

    condition_6 = int32_t(matrix_fragment_86) >= int32_t(uniform_index_word_15);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_50 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_61 = matrix_fragment_61 & 0xffffu;

    matrix_fragment_4 = permute_signed(matrix_fragment_57, 0u, 0x8880u);

    matrix_fragment_51 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    condition_6 = int32_t(matrix_fragment_86) < int32_t(0u) || condition_6;

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_61 = permute_signed(matrix_fragment_61, 0u, 0x8880u);

    matrix_fragment_40 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_57 = (matrix_fragment_34 << 0x2u) + uniform_index_word_9;

    matrix_fragment_34 = matrix_fragment_4;

    matrix_fragment_78 = (matrix_fragment_37 << 0x2u) + uniform_index_word_9;

    condition_0 = int32_t(matrix_fragment_84) >= int32_t(uniform_index_word_14);

    matrix_fragment_37 = matrix_fragment_61;

    matrix_fragment_50 = !condition_6 ? matrix_fragment_50 : 0u;

    matrix_fragment_51 = !condition_6 ? matrix_fragment_51 : 0u;

    matrix_fragment_76 = !condition_6 ? matrix_fragment_76 : 0u;

    matrix_fragment_40 = !condition_6 ? matrix_fragment_40 : 0u;

    matrix_fragment_57 = matrix_fragment_34 + matrix_fragment_57;

    matrix_fragment_35 = !condition_4 ? matrix_fragment_35 : 0u;

    matrix_fragment_78 = matrix_fragment_37 + matrix_fragment_78;

    matrix_fragment_50 = !condition_0 ? matrix_fragment_50 : 0u;

    matrix_fragment_51 = !condition_0 ? matrix_fragment_51 : 0u;

    matrix_fragment_76 = !condition_0 ? matrix_fragment_76 : 0u;

    matrix_fragment_40 = !condition_0 ? matrix_fragment_40 : 0u;

    condition_0 = int32_t(matrix_fragment_57) >= int32_t(uniform_index_word_15);

    matrix_fragment_34 = unpack_fp8_bits(matrix_accumulator_8);

    matrix_fragment_37 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

    matrix_fragment_6 = condition_3 ? matrix_fragment_35 : 0u;

    matrix_accumulator_8 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_fragment_36 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_35 = unpack_fp8_bits(matrix_accumulator_10);

    matrix_fragment_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    condition_0 = int32_t(matrix_fragment_57) < int32_t(0u) || condition_0;

    condition_1 = int32_t(matrix_fragment_84) >= int32_t(0u);

    matrix_fragment_63 = !condition_4 ? matrix_fragment_63 : 0u;

    condition_4 = int32_t(matrix_fragment_69) >= int32_t(uniform_index_word_14);

    matrix_fragment_49 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_8),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_36 = !condition_0 ? matrix_fragment_36 : 0u;

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_37 = !condition_0 ? matrix_fragment_37 : 0u;

    matrix_fragment_34 = condition_1 ? matrix_fragment_50 : 0u;

    matrix_accumulator_10 = condition_1 ? matrix_fragment_40 : 0u;

    matrix_fragment_49 = !condition_0 ? matrix_fragment_49 : 0u;

    matrix_fragment_66 = !condition_0 ? matrix_fragment_66 : 0u;

    matrix_fragment_40 = !condition_4 ? matrix_fragment_36 : 0u;

    matrix_fragment_50 = !condition_4 ? matrix_fragment_37 : 0u;

    condition_0 = int32_t(matrix_fragment_78) >= int32_t(uniform_index_word_15);

    matrix_fragment_36 = unpack_fp8_bits(matrix_accumulator_9);

    matrix_fragment_37 = unpack_fp8_bits(matrix_accumulator_11);

    matrix_accumulator_9 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

    matrix_accumulator_11 = unpack_fp8_bits(matrix_accumulator_11 >> 16);

    matrix_fragment_55 = matrix_fragment_55 & 0xffffu;

    matrix_fragment_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    condition_0 = int32_t(matrix_fragment_78) < int32_t(0u) || condition_0;

    matrix_fragment_35 = condition_1 ? matrix_fragment_51 : 0u;

    matrix_fragment_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_51 = permute_signed(matrix_fragment_55, 0u, 0x8880u);

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_64 = (matrix_accumulator_33 << 0x2u) + uniform_index_word_9;

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_11),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    condition_5 = int32_t(matrix_fragment_69) >= int32_t(0u);

    matrix_fragment_49 = !condition_4 ? matrix_fragment_49 : 0u;

    matrix_fragment_66 = !condition_4 ? matrix_fragment_66 : 0u;

    condition_6 = int32_t(matrix_fragment_67) >= int32_t(uniform_index_word_14);

    matrix_fragment_37 = !condition_0 ? matrix_fragment_37 : 0u;

    matrix_fragment_46 = !condition_0 ? matrix_fragment_46 : 0u;

    matrix_fragment_47 = !condition_0 ? matrix_fragment_47 : 0u;

    matrix_fragment_72 = !condition_0 ? matrix_fragment_72 : 0u;

    matrix_fragment_75 = permute_signed(matrix_fragment_71, 0u, 0x8880u);

    matrix_fragment_51 = matrix_fragment_51 + matrix_fragment_64;

    matrix_fragment_71 = uint32_t(combine_words(packed_half_3, 0u) << 0x5u >> 32) + index_word_132;

    matrix_accumulator_33 = condition_5 ? matrix_fragment_40 : 0u;

    matrix_fragment_36 = condition_5 ? matrix_fragment_49 : 0u;

    matrix_accumulator_9 = condition_5 ? matrix_fragment_66 : 0u;

    matrix_accumulator_11 = condition_5 ? matrix_fragment_50 : 0u;

    condition_5 = int32_t(matrix_fragment_25) >= int32_t(0u);

    condition_0 = int32_t(matrix_fragment_25) >= int32_t(uniform_index_word_14);

    matrix_fragment_40 = !condition_6 ? matrix_fragment_37 : 0u;

    matrix_fragment_46 = !condition_6 ? matrix_fragment_46 : 0u;

    matrix_fragment_47 = !condition_6 ? matrix_fragment_47 : 0u;

    matrix_fragment_72 = !condition_6 ? matrix_fragment_72 : 0u;

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_71 = uint32_t(int32_t(matrix_fragment_71) >> 0x5u);

    matrix_fragment_24 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

    condition_6 = int32_t(matrix_fragment_51) >= int32_t(uniform_index_word_15);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_accumulator_20 = matrix_accumulator_20 & 0xffffu;

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_71 = (matrix_fragment_71 << 0x2u) + uniform_index_word_8;

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_24 = permute_signed(matrix_fragment_24, 0u, 0x9910u);

    matrix_fragment_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    condition_6 = int32_t(matrix_fragment_51) < int32_t(0u) || condition_6;

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_38 = matrix_accumulator_21 & 0xffffu;

    matrix_accumulator_21 = (matrix_fragment_30 << 0x2u) + uniform_index_word_9;

    matrix_accumulator_20 = permute_signed(matrix_accumulator_20, 0u, 0x8880u);

    matrix_fragment_7 = condition_3 ? matrix_fragment_53 : 0u;

    matrix_fragment_4 = condition_3 ? matrix_fragment_63 : 0u;

    matrix_fragment_5 = condition_3 ? matrix_fragment_65 : 0u;

    condition_3 = int32_t(matrix_fragment_67) >= int32_t(0u);

    matrix_fragment_24 = matrix_fragment_24 + matrix_fragment_71;

    condition_4 = int32_t(matrix_fragment_59) >= int32_t(uniform_index_word_14);

    matrix_fragment_25 = !condition_6 ? matrix_fragment_25 : 0u;

    matrix_fragment_12 = !condition_6 ? matrix_fragment_12 : 0u;

    matrix_fragment_37 = !condition_6 ? matrix_fragment_37 : 0u;

    matrix_fragment_14 = !condition_6 ? matrix_fragment_14 : 0u;

    matrix_accumulator_20 = matrix_accumulator_20 + matrix_accumulator_21;

    matrix_fragment_46 = condition_3 ? matrix_fragment_46 : 0u;

    matrix_fragment_47 = condition_3 ? matrix_fragment_47 : 0u;

    matrix_fragment_40 = condition_3 ? matrix_fragment_40 : 0u;

    matrix_fragment_72 = condition_3 ? matrix_fragment_72 : 0u;

    condition_3 = int32_t(matrix_fragment_24) >= int32_t(0u);

    condition_6 = int32_t(matrix_fragment_24) >= int32_t(uniform_index_word_14);

    matrix_fragment_25 = !condition_4 ? matrix_fragment_25 : 0u;

    matrix_accumulator_21 = !condition_4 ? matrix_fragment_12 : 0u;

    matrix_fragment_37 = !condition_4 ? matrix_fragment_37 : 0u;

    matrix_fragment_24 = !condition_4 ? matrix_fragment_14 : 0u;

    condition_4 = int32_t(matrix_accumulator_20) >= int32_t(uniform_index_word_15);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    packed_half_26 = packed_half_26 & 0xffffu;

    matrix_fragment_38 = permute_signed(matrix_fragment_38, 0u, 0x8880u);

    condition_4 = int32_t(matrix_accumulator_20) < int32_t(0u) || condition_4;

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_30 = permute_signed(packed_half_26, 0u, 0x8880u);

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_31 = (matrix_fragment_31 << 0x2u) + uniform_index_word_9;

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_39 = (matrix_accumulator_32 << 0x2u) + uniform_index_word_9;

    packed_half_26 = matrix_fragment_38;

    matrix_fragment_14 = !condition_4 ? matrix_fragment_14 : 0u;

    matrix_accumulator_8 = condition_1 ? matrix_fragment_76 : 0u;

    condition_1 = int32_t(matrix_fragment_59) >= int32_t(0u);

    matrix_fragment_12 = !condition_4 ? matrix_fragment_12 : 0u;

    matrix_fragment_13 = !condition_4 ? matrix_fragment_13 : 0u;

    matrix_fragment_15 = !condition_4 ? matrix_fragment_15 : 0u;

    matrix_fragment_31 = packed_half_26 + matrix_fragment_31;

    matrix_fragment_30 = matrix_fragment_30 + matrix_fragment_39;

    matrix_fragment_14 = !condition_0 ? matrix_fragment_14 : 0u;

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_77) >> 0x2u);

    matrix_accumulator_20 = condition_1 ? matrix_fragment_25 : 0u;

    matrix_fragment_12 = !condition_0 ? matrix_fragment_12 : 0u;

    matrix_fragment_25 = condition_1 ? matrix_accumulator_21 : 0u;

    matrix_fragment_13 = !condition_0 ? matrix_fragment_13 : 0u;

    matrix_accumulator_21 = !condition_0 ? matrix_fragment_15 : 0u;

    condition_4 = int32_t(matrix_fragment_31) >= int32_t(uniform_index_word_15);

    condition_0 = int32_t(matrix_fragment_30) >= int32_t(uniform_index_word_15);

    packed_half_26 = condition_5 ? matrix_fragment_14 : 0u;

    uniform_index_word_8 = 0u;

    matrix_fragment_75 = permute_signed(matrix_fragment_75, 0u, 0x9910u);

    matrix_fragment_14 = unpack_fp8_bits(packed_half_16 >> 16);

    matrix_fragment_15 = condition_5 ? matrix_fragment_12 : 0u;

    matrix_fragment_12 = unpack_fp8_bits(packed_half_18);

    matrix_fragment_38 = condition_1 ? matrix_fragment_24 : 0u;

    matrix_fragment_39 = condition_5 ? matrix_accumulator_21 : 0u;

    condition_4 = int32_t(matrix_fragment_31) < int32_t(0u) || condition_4;

    condition_0 = int32_t(matrix_fragment_30) < int32_t(0u) || condition_0;

    matrix_accumulator_21 = unpack_fp8_bits(packed_half_16);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_24 = unpack_fp8_bits(packed_half_17);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_73 = matrix_fragment_75;

    packed_half_17 = unpack_fp8_bits(packed_half_17 >> 16);

    matrix_fragment_30 = unpack_fp8_bits(packed_half_18 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(packed_half_19 >> 16);

    matrix_fragment_54 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_32 = condition_5 ? matrix_fragment_13 : 0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_42 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_13 = unpack_fp8_bits(packed_half_19);

    matrix_accumulator_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_22 = matrix_fragment_73 + matrix_fragment_88;

    matrix_fragment_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_5 = !condition_4 ? matrix_fragment_14 : 0u;

    matrix_fragment_27 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_42 = !condition_4 ? matrix_fragment_42 : 0u;

    matrix_accumulator_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_58));

    condition_5 = int32_t(matrix_accumulator_22) >= int32_t(uniform_index_word_14);

    matrix_fragment_5 = !condition_6 ? matrix_fragment_5 : 0u;

    matrix_fragment_30 = !condition_4 ? matrix_fragment_30 : 0u;

    matrix_accumulator_28 = !condition_0 ? matrix_accumulator_28 : 0u;

    matrix_fragment_31 = !condition_0 ? matrix_fragment_31 : 0u;

    matrix_fragment_27 = !condition_0 ? matrix_fragment_27 : 0u;

    matrix_fragment_37 = condition_1 ? matrix_fragment_37 : 0u;

    matrix_fragment_42 = !condition_6 ? matrix_fragment_42 : 0u;

    matrix_fragment_54 = !condition_4 ? matrix_fragment_54 : 0u;

    matrix_accumulator_29 = !condition_0 ? matrix_accumulator_29 : 0u;

    condition_1 = int32_t(matrix_accumulator_22) >= int32_t(0u);

    matrix_fragment_5 = condition_3 ? matrix_fragment_5 : 0u;

    matrix_fragment_30 = !condition_6 ? matrix_fragment_30 : 0u;

    matrix_accumulator_28 = !condition_5 ? matrix_accumulator_28 : 0u;

    matrix_fragment_31 = !condition_5 ? matrix_fragment_31 : 0u;

    uniform_index_word_5 = 0x400u;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = (uniform_index_word_8 << 0x18u) + uniform_index_word_5;

    matrix_fragment_27 = !condition_5 ? matrix_fragment_27 : 0u;

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = condition_3 ? matrix_fragment_42 : 0u;

    matrix_fragment_54 = !condition_6 ? matrix_fragment_54 : 0u;

    matrix_accumulator_29 = !condition_5 ? matrix_accumulator_29 : 0u;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = condition_3 ? matrix_fragment_30 : 0u;

    matrix_accumulator_28 = condition_1 ? matrix_accumulator_28 : 0u;

    matrix_fragment_31 = condition_1 ? matrix_fragment_31 : 0u;

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_25 = condition_1 ? matrix_fragment_27 : 0u;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_7 = (matrix_fragment_138 << 0x9u) + uniform_index_word_8;

    packed_half_26 = condition_3 ? matrix_fragment_54 : 0u;

    matrix_fragment_27 = condition_1 ? matrix_accumulator_29 : 0u;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                   select_half_operand<2, false, false>(0u));

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                                       select_half_operand<2, false, false>(0u));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                                   select_half_operand<2, false, false>(0u));

    packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(0u));

    packed_half_140 = address_word_2 * 0x10u + matrix_fragment_7;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                   select_half_operand<2, false, false>(packed_half_16));

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                   select_half_operand<2, false, false>(packed_half_17));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                                   select_half_operand<2, false, false>(packed_half_18));

    packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                   select_half_operand<2, false, false>(packed_half_19));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                      select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                   select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    uniform_index_word_12 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_13 = parameters[parameter_trained_weights_high_5];

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x1000u, packed_half_16,
                                         packed_half_17, packed_half_18, packed_half_19,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, packed_half_140 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x3000u, matrix_fragment_24,
                                         matrix_fragment_25, packed_half_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_accumulator_10 = 0;
    matrix_accumulator_11 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    packed_half_16 = 0;
    packed_half_17 = 0;

    packed_half_18 = 0;
    packed_half_19 = 0;

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    split_address_words(address_word_154, address_word_155,
                        combine_words(uniform_index_word_12, uniform_parameter_word_13));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    packed_half_26 = 0;
    matrix_fragment_27 = 0;

    uniform_index_word_5 = 0u;

    __syncthreads();
    do {
        split_address_words(address_word_164, address_word_165,
                            uint64_t(uint32_t(matrix_fragment_138)) * uint64_t(uint32_t(0x8000u)) +
                                combine_words(address_word_154, address_word_155));

        split_address_words(address_word_164, address_word_165,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_164, address_word_165));

        load_four_words_cached<CheckBounds>(combine_words(address_word_164, address_word_165),
                                            matrix_fragment_36, matrix_fragment_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x1000ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x1200ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x2000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x2200ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x3000ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x3200ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        matrix_fragment_139 = (address_word_2 << 0x4u) + uniform_index_word_8;

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x4000ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x4200ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1000u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2000u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_139 + 0x2200u,
                                            matrix_accumulator_100, matrix_accumulator_101,
                                            matrix_accumulator_102, matrix_accumulator_103,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x200u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1200u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x5000ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_40, matrix_fragment_41, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_42, matrix_fragment_43, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x5200ull, matrix_fragment_84,
            matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_40, matrix_fragment_41, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_42, matrix_fragment_43, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3200u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x6000ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_40, matrix_fragment_41, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_42, matrix_fragment_43, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x6200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2400u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_44, matrix_fragment_45, matrix_accumulator_130, matrix_accumulator_131);

        multiply_accumulate_half(
            matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_46, matrix_fragment_47, matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(
            matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_48, matrix_fragment_49, matrix_accumulator_126, matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x400u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_50, matrix_fragment_51, matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x7000ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1400u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3400u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_164, address_word_165) + 0x7200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x600u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1600u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x800u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2800u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2600u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1800u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3600u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0xa00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3800u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0xc00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_112,
                                 matrix_accumulator_113);

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            uint64_t(uint32_t(matrix_fragment_138)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(address_word_154, address_word_155));

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_130,
                                 matrix_accumulator_131);

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_76, matrix_fragment_77));

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_134,
                                 matrix_accumulator_135);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1a00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2a00u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x40000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x40200ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_134,
                                 matrix_accumulator_135);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2c00u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3c00u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1c00u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3a00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0xe00u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x1e00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x2e00u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_100,
                                 matrix_accumulator_101);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_128,
                                 matrix_accumulator_129);

        matrix_fragment_60 = 0x2b28u;

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_112,
                                 matrix_accumulator_113);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_139 + 0x3e00u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        matrix_accumulator_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x44004400u);

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_accumulator_117 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x44004400u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

        matrix_accumulator_115 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_accumulator_110 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_accumulator_111 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_100,
                                 matrix_accumulator_101);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_128), 0x44004400u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_129), 0x44004400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_66 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_102,
                                 matrix_accumulator_103);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x44004400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x44004400u);

        matrix_accumulator_125 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_66),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_112,
                                 matrix_accumulator_113);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_accumulator_124 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_accumulator_129 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0x44004400u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_45 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_100), 0x44004400u);

        matrix_accumulator_128 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x44004400u);

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101), 0x44004400u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_45),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_105), 0x44004400u);

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_64 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_63 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                          select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                          select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_112), 0x44004400u);

        matrix_fragment_64 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_113), 0x44004400u);

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_47 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_108), 0x44004400u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                          select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

        matrix_fragment_65 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_65),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(matrix_fragment_60), 0x37283728u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_51 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                          select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_49 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                                 matrix_fragment_5);

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(packed_half_16, packed_half_17, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, packed_half_16,
                                 packed_half_17);

        multiply_accumulate_half(packed_half_18, packed_half_19, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, packed_half_18,
                                 packed_half_19);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(packed_half_26, matrix_fragment_27, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_58, matrix_fragment_59, packed_half_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_34,
                                 matrix_fragment_35);

        split_address_words(address_word_154, address_word_155,
                            combine_words(address_word_154, address_word_155) + 0x400u);

        uniform_index_word_5 = uniform_index_word_9;
    } while (!uniform_condition_0);
    split_address_words(matrix_fragment_44, matrix_fragment_45,
                        uint64_t(uint32_t(matrix_accumulator_141)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    address_word_162 = load_aligned_word<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x78000ull, memory_ranges,
        error_counts);

    packed_half_160 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x78000ull, memory_ranges, error_counts);

    packed_half_158 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_158, address_word_159) + 0x78000ull, memory_ranges, error_counts);

    matrix_fragment_44 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x78000ull, memory_ranges,
        error_counts);

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = 0u;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_140, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                                      select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = 0x400u;

    matrix_accumulator_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x1000u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                       select_half_operand<2, false, false>(0u));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_19),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_10));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_140 + 0x2000u, matrix_accumulator_8, matrix_accumulator_9,
        matrix_accumulator_10, matrix_accumulator_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_17),
                                       select_half_operand<2, false, false>(packed_half_18));

    uniform_index_word_12 = (uniform_index_word_8 << 0x18u) + uniform_index_word_5;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x3000u, packed_half_16,
                                        packed_half_17, packed_half_18, packed_half_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = 0u;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                          select_half_operand<2, false, false>(0u));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(matrix_accumulator_22));

    uniform_index_word_8 = uniform_index_word_12;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                          select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(matrix_fragment_25));

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x1000u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       select_half_operand<2, false, false>(packed_half_26));

    packed_half_26 = unpack_fp8_bits(matrix_fragment_38);

    store_four_shared_words<CheckBounds>(
        shared_tile, packed_half_140 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_38 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_fragment_39);

    store_four_shared_words<CheckBounds>(
        shared_tile, packed_half_140 + 0x3000u, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_fragment_39 >> 16);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_40);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_40 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_41);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_41 >> 16);

    matrix_fragment_40 = unpack_fp8_bits(matrix_accumulator_8);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_42);

    matrix_fragment_50 = unpack_fp8_bits(matrix_fragment_42 >> 16);

    matrix_fragment_46 = unpack_fp8_bits(matrix_fragment_43);

    matrix_fragment_51 = unpack_fp8_bits(matrix_fragment_43 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_fragment_48 = unpack_fp8_bits(matrix_accumulator_11);

    matrix_fragment_49 = unpack_fp8_bits(packed_half_16);

    matrix_fragment_42 = unpack_fp8_bits(matrix_accumulator_9);

    matrix_fragment_43 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

    matrix_fragment_47 = unpack_fp8_bits(matrix_accumulator_10);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_36);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_37);

    matrix_fragment_54 = unpack_fp8_bits(packed_half_16 >> 16);

    matrix_fragment_55 = unpack_fp8_bits(packed_half_17);

    matrix_fragment_56 = unpack_fp8_bits(packed_half_17 >> 16);

    matrix_fragment_57 = unpack_fp8_bits(packed_half_18);

    matrix_fragment_59 = unpack_fp8_bits(packed_half_19);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_36 >> 16);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_37 >> 16);

    matrix_fragment_52 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

    matrix_fragment_53 = unpack_fp8_bits(matrix_accumulator_11 >> 16);

    matrix_fragment_58 = unpack_fp8_bits(packed_half_18 >> 16);

    matrix_fragment_60 = unpack_fp8_bits(packed_half_19 >> 16);

    matrix_accumulator_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(address_word_162));

    packed_half_18 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                         select_half_operand<2, false, false>(address_word_162));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(address_word_162));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(packed_half_158));

    split_address_words(matrix_fragment_48, matrix_fragment_49,
                        combine_words(address_word_156, address_word_157) + 0x48200u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_19 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                         select_half_operand<2, false, false>(address_word_162));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                         select_half_operand<2, false, false>(address_word_162));

    packed_half_17 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                         select_half_operand<2, false, false>(address_word_162));

    matrix_accumulator_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(address_word_162));

    matrix_accumulator_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(address_word_162));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                                             select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_44, matrix_fragment_45) - 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_44, matrix_fragment_45),
                                            matrix_fragment_12, matrix_fragment_13,
                                            matrix_fragment_14, matrix_fragment_15, memory_ranges,
                                            error_counts);

        matrix_fragment_60 = (address_word_2 << 0x4u) + uniform_index_word_8;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            combine_words(matrix_fragment_48, matrix_fragment_49) + 0x2000u);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_60, matrix_accumulator_20,
                                            matrix_accumulator_21, matrix_accumulator_22,
                                            matrix_accumulator_23, shared_bytes, memory_ranges,
                                            error_counts);

        uniform_index_word_8 = uniform_index_word_8 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_60 + 0x1000u, matrix_fragment_24, matrix_fragment_25,
            packed_half_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_9;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_60 + 0x2000u, matrix_accumulator_28, matrix_accumulator_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_60 + 0x3000u, matrix_accumulator_32, matrix_accumulator_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(packed_half_16, packed_half_17, matrix_accumulator_20,
                                 matrix_accumulator_21, matrix_accumulator_22,
                                 matrix_accumulator_23, matrix_fragment_4, matrix_fragment_5,
                                 packed_half_16, packed_half_17);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_20,
                                 matrix_accumulator_21, matrix_accumulator_22,
                                 matrix_accumulator_23, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_36, matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_accumulator_20,
                                 matrix_accumulator_21, matrix_accumulator_22,
                                 matrix_accumulator_23, matrix_fragment_12, matrix_fragment_13,
                                 matrix_fragment_54, matrix_fragment_55);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_accumulator_20,
                                 matrix_accumulator_21, matrix_accumulator_22,
                                 matrix_accumulator_23, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_56, matrix_fragment_57);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_24,
                                 matrix_fragment_25, packed_half_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_24,
                                 matrix_fragment_25, packed_half_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                                 matrix_fragment_25, packed_half_26, matrix_fragment_27,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_62,
                                 matrix_fragment_63);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                                 matrix_fragment_25, packed_half_26, matrix_fragment_27,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_28,
                                 matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_28,
                                 matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_accumulator_28,
                                 matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_64,
                                 matrix_fragment_65);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_accumulator_28,
                                 matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(packed_half_18, packed_half_19, matrix_accumulator_32,
                                 matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_4, matrix_fragment_5, packed_half_18,
                                 packed_half_19);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_32,
                                 matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_accumulator_32,
                                 matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_accumulator_32,
                                 matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_58,
                                 matrix_fragment_59);
    } while (!uniform_condition_0);
    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(address_word_154, address_word_155,
                        uint64_t(uint32_t(matrix_fragment_138)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    split_address_words(address_word_154, address_word_155,
                        combine_words(address_word_154, address_word_155) + 0x78e00u);

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_17),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_accumulator_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_accumulator_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x1000u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                      select_half_operand<2, false, false>(matrix_accumulator_11));

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                      select_half_operand<2, false, false>(matrix_accumulator_10));

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                          select_half_operand<2, false, false>(packed_half_17));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                          select_half_operand<2, false, false>(packed_half_16));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                   select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(
        shared_tile, packed_half_140 + 0x2000u, matrix_accumulator_8, matrix_accumulator_9,
        matrix_accumulator_10, matrix_accumulator_11, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                       select_half_operand<2, false, false>(packed_half_17));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_19),
                                       select_half_operand<2, false, false>(packed_half_16));

    packed_half_18 = 0;
    packed_half_19 = 0;

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_accumulator_21));

    packed_half_16 = 0;
    packed_half_17 = 0;

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_140 + 0x3000u, matrix_fragment_44,
                                         matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_10 = 0;
    matrix_accumulator_11 = 0;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_fragment_56 = 0;
    matrix_fragment_57 = 0;

    matrix_fragment_58 = 0;
    matrix_fragment_59 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_fragment_138 = 0;
    matrix_fragment_139 = 0;

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    packed_half_140 = 0;
    matrix_accumulator_141 = 0;

    uniform_index_word_5 = 0u;

    uniform_index_word_8 = uniform_index_word_12;

    __syncthreads();
    do {
        split_address_words(packed_half_158, address_word_159,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_154, address_word_155));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_158, address_word_159) - 0xa00ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_158, address_word_159) - 0x800ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_158, address_word_159) - 0x600ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_158, address_word_159) - 0x400ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_158, address_word_159) - 0x200ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(packed_half_158, address_word_159),
                                            matrix_fragment_80, matrix_fragment_81,
                                            matrix_fragment_82, matrix_fragment_83, memory_ranges,
                                            error_counts);

        packed_half_146 = (address_word_2 << 0x4u) + uniform_index_word_8;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(address_word_154, address_word_155,
                            combine_words(address_word_154, address_word_155) + 0x6000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_146, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_8 = uniform_index_word_8 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_146 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_9;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_146 + 0x2000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_146 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(packed_half_16, packed_half_17, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, packed_half_16,
                                 packed_half_17);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(packed_half_18, packed_half_19, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, packed_half_18,
                                 packed_half_19);

        multiply_accumulate_half(packed_half_140, matrix_accumulator_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, packed_half_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_fragment_138, matrix_fragment_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_138,
                                 matrix_fragment_139);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(packed_half_26, matrix_fragment_27, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, packed_half_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_34,
                                 matrix_fragment_35);
    } while (!uniform_condition_0);
    address_word_159 = threadIdx.y;

    split_address_words(matrix_fragment_60, matrix_fragment_61,
                        uint64_t(uint32_t(address_word_159)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_93 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_60, matrix_fragment_61) + 0xb8400ull, memory_ranges,
        error_counts);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    address_word_163 = address_word_159 * 0x800u;

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    split_address_words(address_word_162, address_word_163,
                        uint64_t(uint32_t(address_word_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    split_address_words(address_word_162, address_word_163,
                        int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_162, address_word_163));

    __syncthreads();

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    packed_half_146 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    address_word_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8600ull, matrix_fragment_96,
        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_139),
                                      select_half_operand<2, false, false>(matrix_fragment_139),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8400ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                      select_half_operand<2, false, false>(packed_half_140),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_138),
                                      select_half_operand<2, false, false>(matrix_fragment_138),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_146), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    matrix_fragment_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                                             select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_154), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    address_word_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8800ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    address_word_154 = add_half_pairs(select_half_operand<2, false, false>(address_word_154),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_146), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_154), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_fragment_73 = permute_signed(matrix_fragment_66, matrix_fragment_66, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_155), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x4100410u);

    matrix_fragment_66 = permute_signed(matrix_fragment_67, matrix_fragment_67, 0x5432u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_66 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_85 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    address_word_154 = add_half_pairs(select_half_operand<2, false, false>(address_word_154),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_71 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5432u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x4100410u);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 = half_to_float_bits((matrix_fragment_79 & 65535u) * 65537u);

    matrix_fragment_66 = rsqrt_bits(matrix_fragment_66);

    matrix_fragment_79 = half_to_float_bits((matrix_fragment_79 >> 16) * 65537u);

    address_word_155 = add_half_pairs(select_half_operand<2, false, false>(address_word_155),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_67 = rsqrt_bits(matrix_fragment_85);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    matrix_fragment_70 = rsqrt_bits(matrix_fragment_70);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_79 = rsqrt_bits(matrix_fragment_79);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_65 = half_to_float_bits((matrix_fragment_84 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_85 = permute_signed(matrix_fragment_64, matrix_fragment_64, 0x5432u);

    matrix_fragment_84 = half_to_float_bits((matrix_fragment_84 >> 16) * 65537u);

    matrix_fragment_67 = pack_half(matrix_fragment_67, matrix_fragment_66);

    matrix_fragment_66 = permute_signed(matrix_fragment_71, matrix_fragment_71, 0x5432u);

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_65 = rsqrt_bits(matrix_fragment_65);

    packed_half_158 = pack_half(matrix_fragment_79, matrix_fragment_70);

    matrix_fragment_84 = rsqrt_bits(matrix_fragment_84);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x4100410u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    address_word_155 = add_half_pairs(select_half_operand<2, false, false>(address_word_155),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8a00ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_85 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_139),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_158 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_160 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_135 = rsqrt_bits(matrix_fragment_85);

    matrix_accumulator_134 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x4100410u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_85 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_fragment_139 = pack_half(matrix_fragment_84, matrix_fragment_65);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_84 = permute_signed(matrix_fragment_69, matrix_fragment_69, 0x5432u);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    matrix_fragment_68 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    matrix_accumulator_136 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x4100410u);

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_68 = rsqrt_bits(matrix_fragment_68);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_138),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_69 = rsqrt_bits(matrix_accumulator_136);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x4100410u);

    matrix_fragment_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    packed_half_161 = half_to_float_bits((matrix_fragment_70 & 65535u) * 65537u);

    packed_half_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8e00ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_accumulator_118 = half_to_float_bits((matrix_fragment_70 >> 16) * 65537u);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_120 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    address_word_165 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa8c00ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9000ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_accumulator_122 = pack_half(matrix_fragment_69, matrix_fragment_68);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9200ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    packed_half_161 = rsqrt_bits(packed_half_161);

    matrix_accumulator_118 = rsqrt_bits(matrix_accumulator_118);

    matrix_fragment_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_accumulator_116 = pack_half(matrix_accumulator_134, matrix_accumulator_135);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_120 = rsqrt_bits(matrix_accumulator_120);

    matrix_accumulator_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                           select_half_operand<2, false, false>(0u));

    address_word_165 = rsqrt_bits(address_word_165);

    packed_half_161 = pack_half(matrix_accumulator_118, packed_half_161);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_121 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                      select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_116 = pack_half(0u, matrix_fragment_93);

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_94 = permute_signed(matrix_fragment_77, matrix_fragment_77, 0x5432u);

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_accumulator_118 = pack_half(address_word_165, matrix_accumulator_120);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_accumulator_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                      select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x4100410u);

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_77 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_accumulator_124 = half_to_float_bits((matrix_fragment_95 & 65535u) * 65537u);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 = half_to_float_bits((matrix_fragment_95 >> 16) * 65537u);

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_124 = rsqrt_bits(matrix_accumulator_124);

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_accumulator_125 = rsqrt_bits(matrix_accumulator_125);

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_158),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_127 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x4100410u);

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_119 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                      select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_110 = half_to_float_bits((matrix_accumulator_127 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_60,
                             matrix_fragment_61);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9400ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_fragment_61 = half_to_float_bits((matrix_accumulator_127 >> 16) * 65537u);

    matrix_accumulator_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                           select_half_operand<2, false, false>(0u));

    packed_half_140 = pack_half(matrix_accumulator_125, matrix_accumulator_124);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_62 = permute_signed(address_word_155, address_word_155, 0x5432u);

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_138),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    address_word_155 = add_half_pairs(select_half_operand<2, false, false>(address_word_155),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_139),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(0u));

    address_word_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(address_word_155), 0x4100410u);

    matrix_fragment_60 = rsqrt_bits(matrix_accumulator_110);

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_accumulator_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_accumulator_134 = half_to_float_bits((address_word_155 >> 16) * 65537u);

    matrix_fragment_63 = permute_signed(address_word_154, address_word_154, 0x5432u);

    matrix_fragment_61 = rsqrt_bits(matrix_fragment_61);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(address_word_154),
                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = half_to_float_bits((address_word_155 & 65535u) * 65537u);

    matrix_fragment_97 = permute_signed(packed_half_146, packed_half_146, 0x5432u);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       select_half_operand<2, false, false>(0u));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = rsqrt_bits(matrix_accumulator_117);

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x4100410u);

    packed_half_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(packed_half_140));

    packed_half_146 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_146), 0x4100410u);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_138 = half_to_float_bits((matrix_accumulator_130 & 65535u) * 65537u);

    matrix_fragment_139 = pack_half(matrix_fragment_61, matrix_fragment_60);

    matrix_accumulator_141 = half_to_float_bits((matrix_accumulator_130 >> 16) * 65537u);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9600ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                       select_half_operand<2, false, false>(0u));

    packed_half_140 = half_to_float_bits((packed_half_146 & 65535u) * 65537u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    packed_half_146 = half_to_float_bits((packed_half_146 >> 16) * 65537u);

    matrix_fragment_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                            select_half_operand<2, false, false>(matrix_fragment_139));

    packed_half_26 = pack_half(matrix_accumulator_134, matrix_accumulator_117);

    matrix_fragment_138 = rsqrt_bits(matrix_fragment_138);

    matrix_accumulator_141 = rsqrt_bits(matrix_accumulator_141);

    packed_half_140 = rsqrt_bits(packed_half_140);

    matrix_accumulator_117 = rsqrt_bits(packed_half_146);

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_26));

    packed_half_26 = pack_half(matrix_accumulator_141, matrix_fragment_138);

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_accumulator_117 = pack_half(matrix_accumulator_117, packed_half_140);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_139 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_139),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_136 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                      select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                      select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_26));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_58), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_59 = result_word;
    }

    matrix_fragment_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_138 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_138 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                        select_half_operand<2, false, false>(matrix_fragment_138));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9800ull, matrix_fragment_96,
        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_fragment_59 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9c00ull, matrix_accumulator_20,
        matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
        error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_59), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_82,
                             matrix_fragment_83);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9a00ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                        select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_57 = permute_signed(matrix_fragment_58, matrix_fragment_58, 0x5432u);

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                   select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xa9e00ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_72,
                             matrix_fragment_73);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xaa000ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    matrix_fragment_95 = half_to_float_bits((matrix_fragment_94 & 65535u) * 65537u);

    matrix_fragment_94 = half_to_float_bits((matrix_fragment_94 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_95 = rsqrt_bits(matrix_fragment_95);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_94 = rsqrt_bits(matrix_fragment_94);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_24 = permute_signed(matrix_fragment_93, matrix_fragment_93, 0x5432u);

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x4100410u);

    matrix_fragment_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                             select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_27 = half_to_float_bits((matrix_fragment_93 & 65535u) * 65537u);

    matrix_fragment_92 = half_to_float_bits((matrix_fragment_93 >> 16) * 65537u);

    matrix_fragment_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                                             select_half_operand<2, false, false>(packed_half_18));

    matrix_fragment_27 = rsqrt_bits(matrix_fragment_27);

    matrix_fragment_93 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                                             select_half_operand<2, false, false>(packed_half_17));

    packed_half_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_19),
                                         select_half_operand<2, false, false>(packed_half_19));

    packed_half_140 = rsqrt_bits(matrix_fragment_92);

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_95 = pack_half(matrix_fragment_94, matrix_fragment_95);

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(packed_half_26));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_93), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_36 = result_word;
    }

    matrix_fragment_27 = pack_half(packed_half_140, matrix_fragment_27);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(packed_half_26));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                   select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0xaa200ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_93), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_43 = permute_signed(matrix_fragment_40, matrix_fragment_40, 0x5432u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_41 = permute_signed(matrix_fragment_42, matrix_fragment_42, 0x5432u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x4100410u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x4100410u);

    matrix_fragment_43 = half_to_float_bits((matrix_fragment_40 & 65535u) * 65537u);

    matrix_fragment_40 = half_to_float_bits((matrix_fragment_40 >> 16) * 65537u);

    matrix_fragment_41 = half_to_float_bits((matrix_fragment_42 & 65535u) * 65537u);

    matrix_fragment_42 = half_to_float_bits((matrix_fragment_42 >> 16) * 65537u);

    matrix_fragment_43 = rsqrt_bits(matrix_fragment_43);

    matrix_fragment_40 = rsqrt_bits(matrix_fragment_40);

    matrix_fragment_41 = rsqrt_bits(matrix_fragment_41);

    matrix_fragment_42 = rsqrt_bits(matrix_fragment_42);

    matrix_fragment_43 = pack_half(matrix_fragment_40, matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_18 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                                         select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_92 = pack_half(matrix_fragment_42, matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    packed_half_19 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_19),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                         select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    packed_half_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_19),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    packed_half_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    packed_half_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                            select_half_operand<0, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_24,
                             matrix_fragment_25, packed_half_26, matrix_fragment_27,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_82,
                             matrix_fragment_83);

    packed_half_26 = 0x29c0u;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_19),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_122 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3c203c20u);

    matrix_accumulator_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3c203c20u);

    matrix_accumulator_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_110 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3c203c20u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_114 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3c203c20u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_17),
                                       select_half_operand<2, false, false>(packed_half_18));

    matrix_accumulator_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_122 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3e473e47u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_124 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3e473e47u);

    matrix_accumulator_110 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3e473e47u);

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3e473e47u);

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3c203c20u);

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    matrix_accumulator_123 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3c203c20u);

    matrix_accumulator_125 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3c203c20u);

    matrix_accumulator_111 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3c203c20u);

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_accumulator_130 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3e473e47u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_42 = (matrix_accumulator_122 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = (matrix_accumulator_124 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 = (matrix_accumulator_110 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_108 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_123 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3e473e47u);

    matrix_accumulator_125 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3e473e47u);

    matrix_accumulator_111 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3e473e47u);

    matrix_accumulator_115 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3e473e47u);

    matrix_accumulator_131 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3c203c20u);

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    packed_half_17 = (matrix_accumulator_130 << 0x5u) + 0x7ff88000u;

    packed_half_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                    select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_40 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_131 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3e473e47u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_fragment_43 = (matrix_accumulator_123 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = (matrix_accumulator_125 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_accumulator_111 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    packed_half_18 = (matrix_accumulator_131 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_25 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                       select_half_operand<2, false, false>(matrix_fragment_94));

    packed_half_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                    select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_24 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                       select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_27 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_89 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    packed_half_16 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    packed_half_19 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                        select_half_operand<2, false, false>(packed_half_19));

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_66 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_73 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_72 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                       select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_67 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                       select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_74 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_68 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_85 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_84 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_71 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_70 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_75 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                       select_half_operand<2, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_accumulator_117 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    {
        const uint32_t bits = address_word_2;
        condition_0 = bool(bits & 1u);
        condition_1 = bool(bits & 2u);
        condition_3 = bool(bits & 8u);
    }

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_accumulator_109 = condition_0 ? matrix_fragment_91 : matrix_accumulator_116;

    matrix_accumulator_110 = condition_0 ? matrix_accumulator_117 : matrix_fragment_86;

    matrix_accumulator_116 = condition_0 ? matrix_accumulator_116 : matrix_fragment_91;

    matrix_accumulator_117 = condition_0 ? matrix_fragment_86 : matrix_accumulator_117;

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    matrix_fragment_90 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_77 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_78 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_62 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_63 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_61 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_60 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_118 = !condition_1 ? matrix_accumulator_109 : matrix_accumulator_110;

    matrix_accumulator_119 = !condition_1 ? matrix_accumulator_110 : matrix_accumulator_109;

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                       select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_21 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3c203c20u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_23 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23), 0x3c203c20u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3c203c20u);

    matrix_accumulator_20 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3c203c20u);

    matrix_accumulator_22 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22), 0x3c203c20u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_21 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3e473e47u);

    packed_half_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(packed_half_26), 0x3d343d34u);

    matrix_accumulator_23 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23), 0x3e473e47u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3e473e47u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_accumulator_20 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3e473e47u);

    matrix_accumulator_22 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22), 0x3e473e47u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_fragment_53 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = (matrix_fragment_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = (matrix_fragment_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    matrix_fragment_55 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = (matrix_accumulator_21 << 0x5u) + 0x7ff88000u;

    matrix_fragment_82 = (matrix_accumulator_23 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    packed_half_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_26), 0x3c203c20u);

    matrix_fragment_81 = (matrix_accumulator_20 << 0x5u) + 0x7ff88000u;

    matrix_fragment_80 = (matrix_accumulator_22 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                        select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                       select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_26), 0x3e473e47u);

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_112 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_111 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_26 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_115 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_114 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_109 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_110 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_22 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_113 = (matrix_accumulator_113 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_22 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                       select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_23 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_14 = address_word_2 * 0x4u;

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_36 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_fragment_15 = matrix_fragment_14 & 0x1cu;

    matrix_accumulator_22 = condition_0 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_39 = condition_0 ? matrix_accumulator_20 : matrix_accumulator_21;

    matrix_fragment_37 = condition_0 ? matrix_accumulator_21 : matrix_accumulator_20;

    matrix_fragment_15 =
        uint32_t(int64_t(int32_t(address_word_2)) * (1ll << 0x1du) >> 32) + matrix_fragment_15;

    matrix_fragment_14 = !condition_1 ? matrix_accumulator_116 : matrix_accumulator_117;

    matrix_fragment_56 = !condition_1 ? matrix_fragment_36 : matrix_fragment_39;

    matrix_accumulator_116 = !condition_1 ? matrix_accumulator_117 : matrix_accumulator_116;

    matrix_fragment_38 = !condition_1 ? matrix_accumulator_22 : matrix_fragment_37;

    matrix_fragment_57 = !condition_1 ? matrix_fragment_37 : matrix_accumulator_22;

    matrix_fragment_36 = !condition_1 ? matrix_fragment_39 : matrix_fragment_36;

    matrix_fragment_13 = 0x1u ^ matrix_fragment_15;

    matrix_fragment_12 = 0x2u ^ matrix_fragment_15;

    matrix_accumulator_23 = 0x3u ^ matrix_fragment_15;

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
        matrix_accumulator_21 = result_word;
    }

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
            : "r"(matrix_accumulator_116), "r"(matrix_accumulator_23), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_116 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_38), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_57), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_36), "r"(matrix_accumulator_23), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_36 = result_word;
    }

    matrix_fragment_14 = !condition_3 ? matrix_accumulator_118 : matrix_accumulator_21;

    matrix_accumulator_21 = !condition_3 ? matrix_accumulator_21 : matrix_accumulator_118;

    matrix_fragment_37 = !condition_3 ? matrix_accumulator_119 : matrix_accumulator_116;

    matrix_accumulator_116 = !condition_3 ? matrix_accumulator_116 : matrix_accumulator_119;

    matrix_accumulator_22 = !condition_3 ? matrix_fragment_38 : matrix_fragment_39;

    matrix_fragment_38 = !condition_3 ? matrix_fragment_39 : matrix_fragment_38;

    matrix_accumulator_20 = !condition_2 ? matrix_fragment_14 : matrix_fragment_37;

    matrix_fragment_15 = !condition_3 ? matrix_fragment_57 : matrix_fragment_36;

    matrix_fragment_57 = !condition_3 ? matrix_fragment_36 : matrix_fragment_57;

    matrix_fragment_36 = packed_half_0 + 0x8u;

    matrix_fragment_14 = !condition_2 ? matrix_fragment_37 : matrix_fragment_14;

    matrix_fragment_13 = !condition_2 ? matrix_accumulator_21 : matrix_accumulator_116;

    matrix_fragment_12 = !condition_2 ? matrix_accumulator_22 : matrix_fragment_15;

    matrix_accumulator_23 = !condition_2 ? matrix_fragment_38 : matrix_fragment_57;

    matrix_fragment_37 = uint32_t(int32_t(matrix_fragment_36) >> 0x1fu);

    matrix_fragment_15 = !condition_2 ? matrix_fragment_15 : matrix_accumulator_22;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_37 =
        uint32_t(combine_words(matrix_fragment_37, 0u) << 0x6u >> 32) + matrix_fragment_36;

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_21 = !condition_2 ? matrix_accumulator_116 : matrix_accumulator_21;

    matrix_fragment_38 = !condition_2 ? matrix_fragment_57 : matrix_fragment_38;

    index_word_149 = uint32_t(combine_words(index_word_149, 0u) << 0x6u >> 32) + packed_half_0;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_37 = matrix_fragment_37 & 0xffc0u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    index_word_149 = index_word_149 & 0xffc0u;

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_37 = matrix_fragment_36 + (0u - matrix_fragment_37);

    matrix_fragment_13 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_13),
                                        select_half_operand<1, false, false>(matrix_fragment_13));

    index_word_149 = packed_half_0 + (0u - index_word_149);

    matrix_fragment_12 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_12),
                                        select_half_operand<1, false, false>(matrix_fragment_12));

    matrix_fragment_14 = permute_signed(matrix_fragment_37, 0u, 0x8880u);

    packed_half_0 = permute_signed(index_word_149, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x5410u);

    matrix_fragment_13 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    packed_half_0 = permute_signed(packed_half_0, 0u, 0x7710u);

    matrix_fragment_13 = uint32_t(uint32_t(matrix_fragment_13) >> 0xau);

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0xau);

    matrix_accumulator_20 = matrix_fragment_13 & 0x1fu;

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_14 = packed_half_0 & 0x1fu;

    matrix_accumulator_22 = matrix_fragment_37 + matrix_accumulator_20;

    packed_half_0 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    matrix_fragment_14 = index_word_149 + matrix_fragment_14;

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_accumulator_20 = matrix_accumulator_22 & 0xe0u;

    packed_half_0 = reciprocal_bits(packed_half_0);

    matrix_fragment_15 = matrix_fragment_14 & 0xe0u;

    matrix_fragment_36 = 0u - matrix_accumulator_20;

    matrix_fragment_13 = reciprocal_bits(matrix_fragment_12);

    index_word_147 = uint32_t(combine_words(index_word_147, 0u) << 0x6u >> 32) + address_word_150;

    matrix_fragment_15 = 0u - matrix_fragment_15;

    index_word_145 = uint32_t(combine_words(index_word_145, 0u) << 0x6u >> 32) + address_word_148;

    matrix_fragment_36 = permute_signed(matrix_fragment_36, 0u, 0x7710u);

    index_word_147 = index_word_147 & 0xffc0u;

    index_word_145 = index_word_145 & 0xffc0u;

    matrix_accumulator_20 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_37 = matrix_fragment_37 + matrix_fragment_36;

    index_word_147 = address_word_150 + (0u - index_word_147);

    index_word_145 = address_word_148 + (0u - index_word_145);

    index_word_149 = index_word_149 + matrix_accumulator_20;

    matrix_accumulator_116 = pack_half(matrix_fragment_13, packed_half_0);

    matrix_fragment_37 = matrix_fragment_37 & 0xffffu;

    packed_half_0 = permute_signed(index_word_147, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(index_word_145, 0u, 0x8880u);

    index_word_149 = index_word_149 & 0xffffu;

    matrix_fragment_15 = permute_signed(matrix_fragment_37, 0u, 0x8880u);

    packed_half_0 = permute_signed(packed_half_0, 0u, 0x7710u);

    matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    matrix_fragment_13 = permute_signed(index_word_149, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0xau);

    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_12) >> 0xau);

    packed_half_0 = packed_half_0 & 0x1fu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_13), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_12 = matrix_fragment_12 & 0x1fu;

    matrix_fragment_38 = index_word_147 + packed_half_0;

    matrix_fragment_49 = transpose_half_tile(matrix_fragment_49);

    matrix_fragment_39 = index_word_145 + matrix_fragment_12;

    packed_half_0 = matrix_fragment_14 & 0xffffu;

    matrix_fragment_51 = transpose_half_tile(matrix_fragment_51);

    matrix_fragment_14 = matrix_fragment_38 & 0xe0u;

    matrix_fragment_12 = matrix_accumulator_22 & 0xffffu;

    matrix_fragment_48 = transpose_half_tile(matrix_fragment_48);

    matrix_accumulator_22 = matrix_fragment_39 & 0xe0u;

    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_12) >> 0x5u);

    matrix_fragment_14 = 0u - matrix_fragment_14;

    matrix_accumulator_22 = 0u - matrix_accumulator_22;

    matrix_accumulator_20 = transpose_half_tile(matrix_accumulator_106);

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0x5u);

    condition_1 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_fragment_12 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_14 = permute_signed(matrix_accumulator_22, 0u, 0x7710u);

    matrix_accumulator_106 = permute_signed(matrix_fragment_15, matrix_accumulator_106, 0x7632u);

    address_word_143 = uint32_t(combine_words(address_word_143, 0u) << 0x6u >> 32) + index_word_144;

    condition_0 = (packed_half_0 & 0x1u) != 0;

    packed_half_0 = permute_signed(matrix_fragment_13, packed_half_0, 0x7632u);

    matrix_accumulator_106 = condition_1 ? matrix_accumulator_106 : matrix_fragment_15;

    address_word_143 = address_word_143 & 0xffc0u;

    index_word_147 = index_word_147 + matrix_fragment_12;

    index_word_145 = index_word_145 + matrix_fragment_14;

    packed_half_0 = condition_0 ? packed_half_0 : matrix_fragment_13;

    matrix_fragment_47 = transpose_half_tile(matrix_fragment_47);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    index_word_144 = index_word_144 + (0u - address_word_143);

    index_word_147 = index_word_147 & 0xffffu;

    index_word_145 = index_word_145 & 0xffffu;

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = transpose_half_tile(matrix_fragment_46);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = permute_signed(index_word_147, 0u, 0x8880u);

    matrix_accumulator_21 = transpose_half_tile(matrix_fragment_50);

    matrix_fragment_51 = permute_signed(index_word_145, 0u, 0x8880u);

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_48 = transpose_half_tile(matrix_fragment_45);

    matrix_fragment_50 = permute_signed(index_word_144, 0u, 0x8880u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_49), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_50 = permute_signed(matrix_fragment_50, 0u, 0x7710u);

    matrix_fragment_41 = transpose_half_tile(matrix_fragment_44);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_51), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    matrix_fragment_50 = uint32_t(uint32_t(matrix_fragment_50) >> 0xau);

    address_word_133 = uint32_t(combine_words(address_word_133, 0u) << 0x6u >> 32) + index_word_142;

    matrix_accumulator_107 = transpose_half_tile(matrix_accumulator_107);

    matrix_fragment_57 = matrix_fragment_50 & 0x1fu;

    matrix_accumulator_105 = transpose_half_tile(matrix_accumulator_105);

    matrix_accumulator_103 = transpose_half_tile(matrix_accumulator_103);

    matrix_accumulator_101 = transpose_half_tile(matrix_accumulator_101);

    address_word_133 = address_word_133 & 0xffc0u;

    matrix_accumulator_104 = transpose_half_tile(matrix_accumulator_104);

    matrix_accumulator_102 = transpose_half_tile(matrix_accumulator_102);

    matrix_accumulator_100 = transpose_half_tile(matrix_accumulator_100);

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = index_word_144 + matrix_fragment_57;

    index_word_142 = index_word_142 + (0u - address_word_133);

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    matrix_fragment_38 = matrix_fragment_38 & 0xffffu;

    matrix_fragment_39 = matrix_fragment_39 & 0xffffu;

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_47 = matrix_fragment_56 & 0xe0u;

    matrix_fragment_50 = permute_signed(index_word_142, 0u, 0x8880u);

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = uint32_t(uint32_t(matrix_fragment_38) >> 0x5u);

    matrix_fragment_39 = uint32_t(uint32_t(matrix_fragment_39) >> 0x5u);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_50 = permute_signed(matrix_fragment_50, 0u, 0x7710u);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_57 = 0u - matrix_fragment_47;

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    condition_0 = (matrix_fragment_38 & 0x1u) != 0;

    condition_1 = (matrix_fragment_39 & 0x1u) != 0;

    matrix_fragment_92 = permute_signed(matrix_fragment_49, matrix_fragment_92, 0x7632u);

    matrix_fragment_48 = permute_signed(matrix_fragment_51, matrix_fragment_48, 0x7632u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_50 = uint32_t(uint32_t(matrix_fragment_50) >> 0xau);

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = condition_0 ? matrix_fragment_92 : matrix_fragment_49;

    matrix_fragment_41 = condition_1 ? matrix_fragment_48 : matrix_fragment_51;

    matrix_fragment_51 = matrix_fragment_50 & 0x1fu;

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_42 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_46 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_72 = index_word_142 + matrix_fragment_51;

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_14 = matrix_fragment_72 & 0xe0u;

    matrix_fragment_51 = 0u - matrix_fragment_14;

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    index_word_144 = index_word_144 + matrix_fragment_57;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_67 = uint32_t(int64_t(int32_t(index_word_151)) * (1ll << 0x1eu) >> 32) + 0x38u;

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x6u >> 32) + index_word_132;

    matrix_fragment_58 = uint32_t(int32_t(matrix_fragment_67) >> 0x1fu);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = permute_signed(matrix_fragment_51, 0u, 0x7710u);

    index_word_144 = index_word_144 & 0xffffu;

    matrix_fragment_58 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x6u >> 32) + matrix_fragment_67;

    packed_half_3 = packed_half_3 & 0xffc0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = permute_signed(index_word_144, 0u, 0x8880u);

    index_word_142 = index_word_142 + matrix_fragment_51;

    matrix_fragment_66 = matrix_fragment_58 & 0xffc0u;

    index_word_132 = index_word_132 + (0u - packed_half_3);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_64));

    index_word_142 = index_word_142 & 0xffffu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_57), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    matrix_fragment_67 = matrix_fragment_67 + (0u - matrix_fragment_66);

    packed_half_3 = matrix_fragment_56 & 0xffffu;

    matrix_fragment_56 = permute_signed(index_word_132, 0u, 0x8880u);

    matrix_fragment_65 = permute_signed(index_word_142, 0u, 0x8880u);

    matrix_fragment_56 = permute_signed(matrix_fragment_56, 0u, 0x7710u);

    matrix_fragment_57 = permute_signed(matrix_fragment_67, 0u, 0x8880u);

    matrix_fragment_66 = uint32_t(uint32_t(matrix_fragment_56) >> 0xau);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_65), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_65 = result_word;
    }

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    matrix_fragment_93 = matrix_fragment_66 & 0x1fu;

    matrix_fragment_73 = uint32_t(uint32_t(matrix_fragment_57) >> 0xau);

    packed_half_3 = uint32_t(uint32_t(packed_half_3) >> 0x5u);

    matrix_fragment_66 = matrix_fragment_73 & 0x1fu;

    matrix_fragment_93 = index_word_132 + matrix_fragment_93;

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    condition_0 = (packed_half_3 & 0x1u) != 0;

    matrix_fragment_89 = matrix_fragment_67 + matrix_fragment_66;

    matrix_fragment_66 = matrix_fragment_93 & 0xe0u;

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    packed_half_3 = permute_signed(matrix_fragment_64, packed_half_3, 0x7632u);

    matrix_fragment_72 = matrix_fragment_72 & 0xffffu;

    matrix_fragment_73 = 0u - matrix_fragment_66;

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = condition_0 ? packed_half_3 : matrix_fragment_64;

    matrix_fragment_64 = uint32_t(uint32_t(matrix_fragment_72) >> 0x5u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_73 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    condition_0 = (matrix_fragment_64 & 0x1u) != 0;

    matrix_fragment_88 = permute_signed(matrix_fragment_65, matrix_fragment_88, 0x7632u);

    matrix_fragment_72 = matrix_fragment_89 & 0xe0u;

    index_word_132 = index_word_132 + matrix_fragment_73;

    matrix_fragment_88 = condition_0 ? matrix_fragment_88 : matrix_fragment_65;

    matrix_fragment_73 = index_word_151 & 0x7ffffffcu;

    matrix_fragment_72 = 0u - matrix_fragment_72;

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_73 = 0u - matrix_fragment_73 + address_word_2;

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_72 = permute_signed(matrix_fragment_72, 0u, 0x7710u);

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_72 = matrix_fragment_67 + matrix_fragment_72;

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = matrix_fragment_72 & 0xffffu;

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_76 = address_word_159 * 0x20u;

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_73 = matrix_fragment_73 * 0x2u + matrix_fragment_76;

    matrix_fragment_76 = uint32_t(uint32_t(matrix_fragment_76) >> 0x1u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    index_word_151 = index_word_151 & 0xfffffffcu;

    matrix_fragment_86 = matrix_fragment_76 + 0x4u;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_87 = matrix_fragment_76 + 0x8u;

    matrix_fragment_79 = uint32_t(int32_t(matrix_fragment_73) >> 0x1u);

    matrix_fragment_76 = matrix_fragment_76 + 0xcu;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_95 = permute_signed(matrix_fragment_72, 0u, 0x8880u);

    matrix_fragment_91 = matrix_fragment_86 + address_word_2 + (0u - index_word_151);

    split_address_words(matrix_fragment_78, matrix_fragment_79,
                        int64_t(int32_t(matrix_fragment_79)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_accumulator_103 = matrix_fragment_87 + address_word_2 + (0u - index_word_151);

    matrix_fragment_93 = matrix_fragment_93 & 0xffffu;

    index_word_151 = matrix_fragment_76 + address_word_2 + (0u - index_word_151);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_95), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    matrix_accumulator_102 = matrix_fragment_89 & 0xffffu;

    split_address_words(matrix_fragment_90, matrix_fragment_91,
                        uint64_t(uint32_t(matrix_fragment_91)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_89 = uint32_t(uint32_t(matrix_fragment_93) >> 0x5u);

    matrix_fragment_86 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_78, matrix_fragment_79) + 0xc8420ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_94, matrix_fragment_95,
                        uint64_t(uint32_t(matrix_accumulator_103)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    matrix_fragment_87 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xc8420ull, memory_ranges,
        error_counts);

    split_address_words(address_word_152, address_word_153,
                        uint64_t(uint32_t(index_word_151)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_152, address_word_153));

    condition_0 = (matrix_fragment_89 & 0x1u) != 0;

    matrix_fragment_89 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_94, matrix_fragment_95) + 0xc8420ull, memory_ranges,
        error_counts);

    matrix_fragment_90 = load_aligned_word<CheckBounds>(
        combine_words(address_word_152, address_word_153) + 0xc8420ull, memory_ranges,
        error_counts);

    index_word_132 = index_word_132 & 0xffffu;

    matrix_fragment_77 = permute_signed(index_word_132, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_77), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_100 = result_word;
    }

    matrix_fragment_78 = uint32_t(uint32_t(matrix_accumulator_102) >> 0x5u);

    condition_1 = (matrix_fragment_78 & 0x1u) != 0;

    matrix_fragment_78 = permute_signed(matrix_accumulator_101, matrix_fragment_78, 0x7632u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_93 = condition_1 ? matrix_fragment_78 : matrix_accumulator_101;

    matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_79 = permute_signed(matrix_accumulator_100, matrix_fragment_79, 0x7632u);

    matrix_fragment_4 = transpose_half_tile(matrix_fragment_4);

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_91 = condition_0 ? matrix_fragment_79 : matrix_accumulator_100;

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_19),
                                             select_half_operand<0, false, false>(packed_half_0));

    packed_half_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    packed_half_18 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    matrix_fragment_81 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                             select_half_operand<0, false, false>(packed_half_0));

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                   select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_80 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                                             select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = transpose_half_tile(matrix_fragment_6);

    matrix_accumulator_9 = transpose_half_tile(matrix_accumulator_9);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_29 = transpose_half_tile(matrix_accumulator_29);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_31 = transpose_half_tile(matrix_fragment_31);

    matrix_accumulator_33 = transpose_half_tile(matrix_accumulator_33);

    matrix_accumulator_11 = transpose_half_tile(matrix_accumulator_11);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_accumulator_8 = transpose_half_tile(matrix_accumulator_8);

    matrix_accumulator_28 = transpose_half_tile(matrix_accumulator_28);

    matrix_fragment_30 = transpose_half_tile(matrix_fragment_30);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_accumulator_32 = transpose_half_tile(matrix_accumulator_32);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_19),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    packed_half_18 = transpose_half_tile(matrix_accumulator_10);

    packed_half_19 = transpose_half_tile(matrix_fragment_34);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<0, false, false>(matrix_accumulator_106));

    matrix_accumulator_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                                         select_half_operand<2, false, false>(0u));

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                      select_half_operand<2, false, false>(matrix_accumulator_9));

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                   select_half_operand<2, false, false>(packed_half_17));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(matrix_accumulator_33));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                   select_half_operand<2, false, false>(matrix_accumulator_11));

    packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_19),
                                   select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_16, packed_half_17, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(packed_half_3));

    uniform_index_word_5 = 0u;

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<0, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_18, packed_half_19, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_10));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_16, packed_half_17, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    packed_half_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                                         select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(matrix_fragment_91));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_18, packed_half_19, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_fragment_88));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_16, packed_half_17, matrix_fragment_66,
                             matrix_fragment_67);

    uniform_index_word_8 = 0x400u;

    uniform_index_word_8 = (uniform_index_word_5 << 0x18u) + uniform_index_word_8;

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_72,
                             matrix_fragment_73);

    packed_half_3 = (address_word_159 << 0x9u) + uniform_index_word_8;

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_18, packed_half_19, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_16, packed_half_17, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_fragment_40 = address_word_2 * 0x10u + packed_half_3;

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_accumulator_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                         select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_18, packed_half_19, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_accumulator_8));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_40, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_40 + 0x3000u, matrix_accumulator_8, matrix_accumulator_9,
        matrix_accumulator_10, matrix_accumulator_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_40 + 0x1000u, packed_half_16,
                                        packed_half_17, packed_half_18, packed_half_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_40 + 0x2000u, matrix_fragment_4, matrix_fragment_5,
        matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                          select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                          select_half_operand<2, false, false>(packed_half_0));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                          select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                          select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(matrix_accumulator_22));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                   select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_accumulator_22));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_40, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_40 + 0x1000u, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_40 + 0x2000u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_40 + 0x3000u, matrix_fragment_24, matrix_fragment_25,
        packed_half_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_12);

    packed_half_0 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_20 = unpack_fp8_bits(packed_half_16);

    matrix_accumulator_21 = unpack_fp8_bits(packed_half_17);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_9);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_5);

    packed_half_26 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_accumulator_8);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    packed_half_16 = unpack_fp8_bits(packed_half_16 >> 16);

    packed_half_17 = unpack_fp8_bits(packed_half_17 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_15);

    matrix_accumulator_23 = unpack_fp8_bits(packed_half_19);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_accumulator_22 = unpack_fp8_bits(packed_half_18);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_88 = unpack_fp8_bits(matrix_accumulator_10);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    packed_half_18 = unpack_fp8_bits(packed_half_18 >> 16);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    packed_half_19 = unpack_fp8_bits(packed_half_19 >> 16);

    matrix_accumulator_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_28 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_34 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_35 = unpack_fp8_bits(matrix_accumulator_11);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_36 = unpack_fp8_bits(matrix_accumulator_11 >> 16);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    split_address_words(address_word_156, address_word_157,
                        combine_words(address_word_156, address_word_157) + 0xb8620u);

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_17),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    packed_half_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_18),
                                         select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_19),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    uniform_index_word_5 = 0u;

    __syncthreads();
    do {
        split_address_words(packed_half_16, packed_half_17,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_156, address_word_157));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_16, packed_half_17) - 0x200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_16, packed_half_17), matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

        packed_half_0 = (address_word_2 << 0x4u) + uniform_index_word_8;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(address_word_156, address_word_157,
                            combine_words(address_word_156, address_word_157) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0, matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
            matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_8 = uniform_index_word_8 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_9;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3000u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(packed_half_26, matrix_fragment_27, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_50, matrix_fragment_51, packed_half_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_14,
                                 matrix_fragment_15);
    } while (!uniform_condition_0);
    uniform_index_word_5 = blockIdx.y;

    uniform_index_word_8 = parameters[parameter_optional_control_11];

    matrix_fragment_30 = threadIdx.x;

    packed_half_0 = address_word_159 * 0x80u;

    uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);

    packed_half_3 = threadIdx.z;

    uniform_index_word_9 = uniform_index_word_6 * uniform_index_word_11 + uniform_address_word_7;

    matrix_fragment_24 = parameters[parameter_output_low_2];
    matrix_fragment_25 = parameters[parameter_output_high_3];

    packed_half_17 = uniform_index_word_9;

    packed_half_17 = packed_half_17 * 0x400u + packed_half_0;

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_index_word_8;

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x4u) && !uniform_condition_0;

    uniform_condition_3 = uniform_condition_0 & uniform_condition_1;

    split_address_words(packed_half_16, packed_half_17,
                        int64_t(int32_t(packed_half_17)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));
    if (uniform_condition_3) {
        packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(0u));

        packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_31 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                          select_half_operand<2, false, false>(packed_half_18));

        matrix_accumulator_20 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                          select_half_operand<2, false, false>(packed_half_19));

        split_address_words(packed_half_18, packed_half_19,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, packed_half_17));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_accumulator_22 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        store_four_words_streaming<CheckBounds>(combine_words(packed_half_18, packed_half_19),
                                                matrix_accumulator_20, matrix_accumulator_21,
                                                matrix_accumulator_22, matrix_accumulator_23,
                                                memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                   select_half_operand<2, false, false>(0u));

    packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                         select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        packed_half_19 = permute_signed(packed_half_19, matrix_accumulator_8, 0x5410u);

        packed_half_18 = permute_signed(packed_half_18, matrix_accumulator_9, 0x5410u);

        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, packed_half_17));

        packed_half_17 = permute_signed(matrix_accumulator_22, matrix_fragment_41, 0x5410u);

        packed_half_16 = permute_signed(matrix_accumulator_20, matrix_accumulator_21, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x1000ull, packed_half_16,
            packed_half_17, packed_half_18, packed_half_19, memory_ranges, error_counts);
    }
    uniform_index_word_11 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;

    uniform_index_word_6 = uniform_index_word_6 + 0x1u;

    condition_0 = (address_word_159 | matrix_fragment_30 | packed_half_3) != 0;

    uniform_address_word_7 = uniform_address_word_7 + uniform_index_word_11;

    uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                                  select_half_operand<2, false, false>(0u));

    packed_half_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                   select_half_operand<2, false, false>(0u));

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x8u) && !uniform_condition_0;

    matrix_accumulator_9 = uniform_address_word_7;

    matrix_accumulator_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(0u));

    uniform_condition_1 = uniform_condition_0 & uniform_condition_1;

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 = matrix_accumulator_9 * 0x400u + packed_half_0;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                                  select_half_operand<2, false, false>(0u));

    packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                   select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                        int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));

    packed_half_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_1) {
        packed_half_18 = permute_signed(packed_half_18, matrix_fragment_7, 0x5410u);

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));

        packed_half_19 = permute_signed(packed_half_19, matrix_fragment_4, 0x5410u);

        packed_half_17 = permute_signed(packed_half_17, packed_half_0, 0x5410u);

        packed_half_16 = permute_signed(packed_half_16, packed_half_3, 0x5410u);

        store_four_words_streaming<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                                packed_half_16, packed_half_17, packed_half_18,
                                                packed_half_19, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                          select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_4, 0x5410u);

        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_13, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_accumulator_10, matrix_accumulator_11, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, packed_half_3) + 0x1000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_index_word_5 = blockIdx.x;

    uniform_index_word_6 = parameters[parameter_publication_optional_low_16];
    uniform_address_word_7 = parameters[parameter_publication_optional_high_17];

    uniform_index_word_5 = uniform_index_word_5 + uniform_address_word_4;

    split_address_words(uniform_index_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_index_word_6, uniform_address_word_7));

    split_address_words(address_word_2, packed_half_3,
                        combine_words(uniform_index_word_6, uniform_address_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__
__maxnreg__(128) void phoenix_window_attention_channels_256_upsample(KernelArguments parameters) {
    run_window_attention_channels_256_upsample<false>(parameters.words, nullptr, nullptr);
}
