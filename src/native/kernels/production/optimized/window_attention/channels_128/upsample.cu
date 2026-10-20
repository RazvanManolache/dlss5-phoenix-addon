// DLSS5 Phoenix optimized kernel: window_attention/channels_128/upsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (__CUDA_ARCH__ >= 900)
#define DLSS5_TILE_SHARED_BYTES 9216
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
    parameter_scalar_control_10 = 10,
    parameter_optional_control_11 = 11,
    parameter_publication_optional_low_16 = 16,
    parameter_publication_optional_high_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_128_upsample(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t address_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t packed_half_2 = 0;
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
    uint32_t matrix_accumulator_32 = 0;
    uint32_t matrix_accumulator_33 = 0;
    uint32_t matrix_accumulator_34 = 0;
    uint32_t matrix_accumulator_35 = 0;
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
    uint32_t matrix_accumulator_52 = 0;
    uint32_t matrix_accumulator_53 = 0;
    uint32_t matrix_accumulator_54 = 0;
    uint32_t matrix_accumulator_55 = 0;
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
    uint32_t matrix_fragment_68 = 0;
    uint32_t matrix_fragment_69 = 0;
    uint32_t matrix_accumulator_70 = 0;
    uint32_t matrix_accumulator_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t matrix_fragment_74 = 0;
    uint32_t matrix_fragment_75 = 0;
    uint32_t matrix_fragment_76 = 0;
    uint32_t matrix_fragment_77 = 0;
    uint32_t matrix_fragment_78 = 0;
    uint32_t packed_half_79 = 0;
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
    uint32_t matrix_fragment_92 = 0;
    uint32_t matrix_fragment_93 = 0;
    uint32_t matrix_fragment_94 = 0;
    uint32_t matrix_fragment_95 = 0;
    uint32_t matrix_fragment_96 = 0;
    uint32_t matrix_fragment_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t matrix_fragment_100 = 0;
    uint32_t matrix_fragment_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
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
    uint32_t packed_half_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t index_word_138 = 0;
    uint32_t tile_word_139 = 0;
    uint32_t index_word_140 = 0;
    uint32_t index_word_141 = 0;
    uint32_t index_word_142 = 0;
    uint32_t index_word_143 = 0;
    uint32_t address_word_144 = 0;
    uint32_t index_word_145 = 0;
    uint32_t address_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t address_word_148 = 0;
    uint32_t address_word_149 = 0;
    uint32_t address_word_150 = 0;
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
    uint32_t float_bits_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t float_bits_163 = 0;
    uint32_t reused_word_164 = 0;
    uint32_t float_bits_165 = 0;
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

    address_word_0 = threadIdx.x & 31u;

    uniform_parameter_word_21 = blockIdx.y;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_accumulator_105 = threadIdx.y;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    uniform_index_word_10 = parameters[parameter_scalar_control_10];
    uniform_index_word_11 = parameters[parameter_optional_control_11];

    matrix_fragment_40 = threadIdx.x & 31u;

    uniform_index_word_9 = blockIdx.x;

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_address_word_4 = uniform_address_word_4 + 0x1u;

    packed_half_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_address_word_4, 0u) << 0x1u >> 32) + uniform_address_word_4;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_address_word_4)) * (1ll << 0x1fu) >> 32) + 0x3u;

    uniform_index_word_8 = (uniform_parameter_word_21 << 0x3u) + uniform_index_word_11;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_10));

    uniform_address_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x1u >> 32) + uniform_index_word_8;

    index_word_143 = address_word_0;

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_address_word_7, 0u) << 0x2u >> 32) + uniform_index_word_6;

    uniform_index_word_6 = uniform_index_word_15 + 0x1u;

    uniform_index_word_9 = (uniform_index_word_9 << 0x3u) + uniform_index_word_10;

    uniform_address_word_7 = 0u - uniform_index_word_5 + uniform_index_word_15;

    matrix_fragment_4 = uint32_t(int32_t(index_word_143) >> 0x2u);

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_9, 0u) << 0x1u >> 32) + uniform_index_word_9;

    uniform_condition_0 = uint32_t(uniform_address_word_7) != 0u;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1fu) >> 32) + 0x3u;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x2u >> 32) + matrix_fragment_4;

    uniform_parameter_word_13 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    matrix_fragment_27 = address_word_0;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    matrix_fragment_5 = matrix_fragment_5 & 0xfffffffcu;

    uniform_index_word_18 = uniform_index_word_14 & 0xfffffffcu;

    matrix_fragment_6 = uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                        uniform_index_word_12;

    uniform_index_word_15 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_6;

    matrix_fragment_28 = matrix_fragment_4 + uniform_parameter_word_13 + (0u - matrix_fragment_5);

    if (uniform_condition_0) {
        uniform_index_word_10 = uniform_address_word_7 + 0xffffffffu;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_index_word_19 = uniform_index_word_15 & 0xfffffffcu;

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(uniform_index_word_18);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);
    }

    split_address_words(packed_half_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_accumulator_105)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_9) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_8;

    condition_1 = int32_t(matrix_fragment_28) >= int32_t(uniform_index_word_19);

    condition_0 = int32_t(matrix_fragment_6) > -1 && !condition_0;

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(combine_words(uniform_index_word_11, 0u) << 0x3u >> 32) +
                                uniform_index_word_10;
    }

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        combine_words(packed_half_2, matrix_accumulator_3) + 0x18200u);

    condition_1 = int32_t(matrix_fragment_28) > -1 && !condition_1;

    uniform_address_word_7 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x2u >> 32) + uniform_index_word_9;

    uniform_index_word_15 = uniform_index_word_15 << (0x2u & 31u);

    uniform_index_word_14 = uniform_index_word_14 << (0x1u & 31u);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x3u);
    }

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    condition_0 = condition_0 & condition_1;

    uniform_address_word_4 = 0u;

    matrix_fragment_26 = uniform_index_word_12;

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
            matrix_fragment_27 =
                uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_26;

            matrix_fragment_28 = matrix_fragment_28 * 0x4u;

            index_word_143 = index_word_143 & 0xfffffffcu;
        }

        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        if (condition_0) {
            matrix_fragment_14 = parameters[parameter_input_low_0];
            matrix_fragment_15 = parameters[parameter_input_high_1];

            matrix_fragment_27 = matrix_fragment_27 * uniform_index_word_15 + matrix_fragment_28;
        }

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_18, matrix_fragment_19),
                                            matrix_fragment_36, matrix_fragment_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        if (condition_0) {
            matrix_fragment_13 = matrix_fragment_27 + address_word_0 + (0u - index_word_143);
        }

        address_word_0 = matrix_fragment_40;

        index_word_143 = address_word_0;

        matrix_fragment_27 = address_word_0;

        address_word_144 = uint32_t(int32_t(index_word_143) >> 0x2u);

        matrix_fragment_20 = uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                             uniform_index_word_12;

        matrix_fragment_12 =
            uint32_t(combine_words(address_word_144, 0u) << 0x2u >> 32) + address_word_144;

        if (condition_0) {
            split_address_words(matrix_fragment_14, matrix_fragment_15,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_14, matrix_fragment_15));
        }

        condition_3 = int32_t(matrix_fragment_20) >= int32_t(uniform_index_word_18);

        matrix_fragment_21 = matrix_fragment_12 & 0xfffffffcu;

        matrix_fragment_12 = uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                             uniform_parameter_word_20;

        matrix_fragment_28 =
            address_word_144 + uniform_parameter_word_13 + (0u - matrix_fragment_21);

        condition_3 = int32_t(matrix_fragment_20) > -1 && !condition_3;

        condition_1 = int32_t(matrix_fragment_12) >= int32_t(uniform_index_word_18);

        condition_2 = int32_t(matrix_fragment_28) >= int32_t(uniform_index_word_19);

        condition_1 = int32_t(matrix_fragment_12) > -1 && !condition_1;

        matrix_fragment_12 = 0u;

        condition_2 = int32_t(matrix_fragment_28) > -1 && !condition_2;

        condition_1 = condition_1 & condition_2;

        if (condition_0) {
            matrix_fragment_12 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
        }

        condition_0 = condition_3 & condition_2;

        if (condition_1) {
            matrix_fragment_30 = matrix_fragment_26 + uniform_index_word_18;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            matrix_fragment_13 =
                uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_26;

            matrix_accumulator_32 = matrix_fragment_28 * 0x4u;

            matrix_fragment_31 =
                uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_30;

            matrix_fragment_29 = matrix_fragment_28 * 0x4u;
        }

        if (condition_0) {
            matrix_fragment_30 = matrix_fragment_26 + uniform_index_word_18;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + 0x2u;
        }

        if (condition_0) {
            matrix_fragment_22 = parameters[parameter_input_low_0];
            matrix_fragment_23 = parameters[parameter_input_high_1];
        }

        if (condition_1) {
            matrix_fragment_14 = matrix_fragment_31 + 0x2u;
        }

        if (condition_0) {
            matrix_fragment_15 = matrix_fragment_28 * 0x4u;

            matrix_fragment_30 =
                uint32_t(int64_t(int32_t(matrix_fragment_27)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_30;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 * uniform_index_word_15 + matrix_accumulator_32;

            matrix_fragment_31 = index_word_143 & 0xfffffffcu;

            matrix_fragment_29 = matrix_fragment_14 * uniform_index_word_15 + matrix_fragment_29;

            matrix_fragment_14 = index_word_143 & 0xfffffffcu;

            matrix_fragment_24 = parameters[parameter_input_low_0];
            matrix_fragment_25 = parameters[parameter_input_high_1];
        }

        if (condition_0) {
            matrix_fragment_15 = matrix_fragment_30 * uniform_index_word_15 + matrix_fragment_15;

            matrix_fragment_30 = index_word_143 & 0xfffffffcu;
        }

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19) - 0x200ull, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, memory_ranges,
            error_counts);

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + address_word_0 + (0u - matrix_fragment_14);
        }

        if (condition_0) {
            matrix_fragment_15 = matrix_fragment_15 + address_word_0 + (0u - matrix_fragment_30);
        }

        if (condition_1) {
            matrix_fragment_29 = matrix_fragment_29 + address_word_0 + (0u - matrix_fragment_31);

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }

        matrix_fragment_13 = 0u;

        if (condition_0) {
            split_address_words(matrix_fragment_22, matrix_fragment_23,
                                int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_22, matrix_fragment_23));
        }

        matrix_fragment_14 = 0;
        matrix_fragment_15 = 0;

        if (condition_1) {
            matrix_fragment_13 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_20, matrix_fragment_21), memory_ranges, error_counts);
        }

        if (condition_0) {
            matrix_fragment_14 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_22, matrix_fragment_23), memory_ranges, error_counts);
        }

        if (condition_1) {
            split_address_words(matrix_fragment_24, matrix_fragment_25,
                                int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_24, matrix_fragment_25));

            matrix_fragment_15 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25), memory_ranges, error_counts);
        }

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_index_word_10 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x1000u);

        matrix_fragment_26 = matrix_fragment_26 + uniform_index_word_14;

        uniform_index_word_5 = uniform_index_word_10;

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                                 matrix_fragment_5);
    } while (!uniform_condition_0);
    matrix_fragment_12 = address_word_0 * 0x2u;

    matrix_fragment_14 = uint32_t(uint32_t(address_word_0) >> 0x1u);

    uniform_index_word_14 = parameters[parameter_tensor_height_8];
    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_fragment_17 = index_word_143 & 0x7ffffffcu;

    matrix_fragment_16 = matrix_accumulator_105 * 0x20u;

    matrix_fragment_13 = matrix_fragment_12 & 0x8u;

    uniform_condition_5 = false;

    matrix_fragment_15 = matrix_fragment_14 & 0x4u;

    address_word_148 = parameters[parameter_trained_weights_low_4];
    address_word_149 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_12 = matrix_fragment_13 | 0x13u & address_word_0;

    matrix_fragment_17 = 0u - matrix_fragment_17 + address_word_0;

    matrix_fragment_19 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x8u;

    matrix_fragment_14 = matrix_fragment_12 | matrix_fragment_15;

    matrix_fragment_12 = matrix_fragment_15 | 0x13u & address_word_0;

    matrix_accumulator_52 = matrix_fragment_17 * 0x2u + matrix_fragment_16;

    matrix_fragment_16 = uint32_t(uint32_t(matrix_fragment_16) >> 0x1u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    matrix_fragment_13 = matrix_fragment_12 | matrix_fragment_13;

    uniform_index_word_10 = uniform_index_word_14 & 0xfffffffcu;

    matrix_fragment_12 = uint32_t(int32_t(matrix_fragment_19) >> 0x1fu);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);

    matrix_fragment_15 = matrix_fragment_16 + 0x4u;

    matrix_fragment_20 = index_word_143 & 0xfffffffcu;

    uniform_condition_3 = uint32_t(uniform_index_word_10) != uint32_t(0x4u);

    matrix_accumulator_59 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x4u >> 32) + matrix_fragment_19;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_15) >> 0x1fu);

    matrix_accumulator_32 =
        uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x20u;

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_14;

    matrix_fragment_17 = matrix_fragment_16 + 0x8u;

    matrix_accumulator_107 = matrix_fragment_16 + 0xcu;

    matrix_fragment_67 = matrix_fragment_15 + address_word_0 + (0u - matrix_fragment_20);

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_15;

    matrix_fragment_16 = matrix_accumulator_59 & 0xfff0u;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    matrix_fragment_15 = uint32_t(int32_t(matrix_accumulator_32) >> 0x1fu);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    matrix_fragment_18 = address_word_0;

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_30 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x28u;

    uniform_condition_1 = int32_t(uniform_address_word_7) >= int32_t(uniform_index_word_11);

    matrix_fragment_28 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x38u;

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_0;
    }

    matrix_accumulator_63 =
        uint32_t(combine_words(matrix_fragment_15, 0u) << 0x4u >> 32) + matrix_accumulator_32;

    uniform_index_word_18 = uniform_index_word_15 & 0xfffffffcu;

    matrix_fragment_23 = uint32_t(int32_t(matrix_fragment_18) >> 0x6u);

    uniform_condition_0 = true;

    matrix_fragment_27 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x5u >> 32) + matrix_fragment_19;

    uniform_condition_1 =
        int32_t(uniform_index_word_9) > int32_t(0u - 0x4u) && !uniform_condition_1;

    matrix_fragment_16 = matrix_fragment_19 + (0u - matrix_fragment_16);

    matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_30) >> 0x1fu);

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_28) >> 0x1fu);

    uniform_condition_2 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    matrix_fragment_69 = matrix_fragment_17 + address_word_0 + (0u - matrix_fragment_20);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    matrix_fragment_29 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x30u;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_2 | uniform_condition_1);

    matrix_fragment_17 = matrix_accumulator_63 & 0xfff0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_18 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x1u >> 32) + matrix_fragment_23;

    uniform_condition_4 = uint32_t(uniform_index_word_18) != uint32_t(0x4u);

    matrix_fragment_76 =
        uint32_t(combine_words(matrix_fragment_25, 0u) << 0x4u >> 32) + matrix_fragment_30;

    uniform_index_word_12 = uniform_condition_5 ? uniform_address_word_7 : 0u;

    matrix_fragment_12 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x4u >> 32) + matrix_fragment_28;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_fragment_24 = uint32_t(int32_t(matrix_fragment_29) >> 0x1fu);

    uniform_index_word_5 = 0u;

    matrix_fragment_26 =
        uint32_t(combine_words(matrix_fragment_15, 0u) << 0x5u >> 32) + matrix_accumulator_32;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_60 = result_word;
    }

    matrix_fragment_18 = matrix_fragment_18 & 0x3ffffffeu;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11;
    }

    matrix_fragment_15 = matrix_fragment_76 & 0xfff0u;

    uniform_index_word_12 = !uniform_condition_4 ? uniform_index_word_12 : uniform_address_word_7;

    matrix_accumulator_32 = matrix_accumulator_32 + (0u - matrix_fragment_17);

    matrix_fragment_17 = matrix_fragment_12 & 0xfff0u;

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    packed_half_79 =
        uint32_t(combine_words(matrix_fragment_24, 0u) << 0x4u >> 32) + matrix_fragment_29;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_25 =
        uint32_t(combine_words(matrix_fragment_25, 0u) << 0x5u >> 32) + matrix_fragment_30;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_fragment_23 = matrix_fragment_23 + (0u - matrix_fragment_18);

    matrix_fragment_30 = matrix_fragment_30 + (0u - matrix_fragment_15);

    matrix_fragment_22 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x5u >> 32) + matrix_fragment_28;

    if (uniform_condition_3) {
        uniform_index_word_12 = uniform_index_word_6 + 0x1u;
    }

    matrix_fragment_18 = packed_half_79 & 0xfff0u;

    uniform_condition_5 = false;

    matrix_fragment_28 = matrix_fragment_28 + (0u - matrix_fragment_17);

    matrix_fragment_15 = 0x8u ^ matrix_fragment_13;

    if (uniform_condition_3) {
        uniform_condition_2 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_17 = 0x10u ^ matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_70 = result_word;
    }

    matrix_fragment_13 = 0x18u ^ matrix_fragment_13;

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_2;
    }

    matrix_fragment_24 =
        uint32_t(combine_words(matrix_fragment_24, 0u) << 0x5u >> 32) + matrix_fragment_29;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    matrix_fragment_29 = matrix_fragment_29 + (0u - matrix_fragment_18);

    matrix_accumulator_107 = matrix_accumulator_107 + address_word_0 + (0u - matrix_fragment_20);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    condition_0 = uniform_condition_0;

    condition_1 = (address_word_0 & 0x10u) != 0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_33 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    uniform_condition_2 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    matrix_accumulator_53 = !condition_1 ? matrix_fragment_18 : matrix_fragment_21;

    matrix_fragment_40 = !condition_1 ? matrix_fragment_44 : matrix_fragment_19;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_fragment_44 = !condition_1 ? matrix_fragment_19 : matrix_fragment_44;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_55 = result_word;
    }

    uniform_index_word_12 = uniform_condition_5 ? uniform_address_word_7 : 0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_58 = result_word;
    }

    index_word_142 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x10u;

    if (uniform_condition_3) {
        uniform_condition_6 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_46 = !condition_1 ? matrix_fragment_48 : matrix_fragment_31;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    matrix_fragment_45 = !condition_1 ? matrix_fragment_43 : matrix_fragment_36;

    index_word_140 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x18u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    uniform_index_word_12 = !uniform_condition_4 ? uniform_index_word_12 : uniform_address_word_7;

    matrix_fragment_4 = uniform_index_word_5;

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_accumulator_82 = !condition_1 ? matrix_fragment_37 : matrix_fragment_10;

    matrix_fragment_15 = !condition_1 ? matrix_fragment_21 : matrix_fragment_18;

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    uniform_condition_0 = true;

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    matrix_fragment_14 = !condition_1 ? matrix_fragment_10 : matrix_fragment_37;

    matrix_fragment_75 = !condition_1 ? matrix_fragment_73 : matrix_fragment_8;

    matrix_fragment_73 = !condition_1 ? matrix_fragment_8 : matrix_fragment_73;

    uniform_condition_2 = uniform_condition_0 & (uniform_condition_2 | uniform_condition_1);

    if (!condition_0) {
        matrix_fragment_7 = uniform_index_word_5;
    }

    matrix_accumulator_71 = !condition_1 ? matrix_fragment_9 : matrix_fragment_50;

    matrix_fragment_21 = !condition_1 ? matrix_fragment_50 : matrix_fragment_9;

    if (condition_0) {
        matrix_fragment_5 = matrix_fragment_4 * 0x4u + matrix_accumulator_105;
    }

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (!condition_0) {
        matrix_fragment_4 = uniform_index_word_5;
    }

    matrix_fragment_18 = uint32_t(int32_t(matrix_accumulator_59) >> 0x4u);

    if (condition_0) {
        matrix_fragment_5 = matrix_fragment_5 * 0x80u;
    }

    matrix_accumulator_62 = !condition_1 ? matrix_accumulator_70 : matrix_accumulator_55;

    matrix_accumulator_70 = !condition_1 ? matrix_accumulator_55 : matrix_accumulator_70;

    if (uniform_condition_2) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    if (condition_0) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_0) {
        matrix_fragment_5 = uniform_index_word_5;
    }

    matrix_accumulator_54 = !condition_1 ? matrix_accumulator_60 : matrix_fragment_11;

    index_word_141 = uint32_t(int32_t(index_word_142) >> 0x1fu);

    matrix_fragment_48 = !condition_1 ? matrix_fragment_31 : matrix_fragment_48;

    matrix_fragment_19 = !condition_1 ? matrix_fragment_51 : matrix_fragment_6;

    matrix_fragment_17 = !condition_1 ? matrix_fragment_6 : matrix_fragment_51;

    if (!condition_0) {
        matrix_fragment_6 = uniform_index_word_5;
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_accumulator_59 =
        uint32_t(combine_words(matrix_accumulator_59, 0u) << 0x1u >> 32) + matrix_fragment_18;

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_accumulator_60 = !condition_1 ? matrix_fragment_11 : matrix_accumulator_60;

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    condition_0 = uniform_condition_2;

    uniform_index_word_12 = uniform_address_word_7 + 0x1u;

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_accumulator_107)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_55 = matrix_accumulator_59 & 0x3ffffffeu;

    uniform_condition_5 = false;

    matrix_fragment_43 = !condition_1 ? matrix_fragment_36 : matrix_fragment_43;

    matrix_fragment_47 = !condition_1 ? matrix_fragment_49 : matrix_fragment_38;

    matrix_accumulator_56 = !condition_1 ? matrix_accumulator_58 : matrix_fragment_39;

    matrix_fragment_74 = !condition_1 ? matrix_fragment_72 : matrix_fragment_41;

    matrix_fragment_8 = uniform_index_word_5;

    if (!uniform_condition_2) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_6;
    }

    matrix_fragment_31 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x20100ull, memory_ranges,
        error_counts);

    matrix_fragment_36 = uint32_t(int32_t(matrix_accumulator_63) >> 0x4u);

    if (!uniform_condition_2) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    uniform_condition_2 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_11);

    matrix_accumulator_55 = matrix_fragment_18 + (0u - matrix_accumulator_55);

    uniform_condition_0 = true;

    matrix_fragment_18 = uint32_t(combine_words(index_word_141, 0u) << 0x4u >> 32) + index_word_142;

    uniform_condition_2 =
        int32_t(uniform_index_word_9) > int32_t(0u - 0x8u) && !uniform_condition_2;

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    uniform_condition_6 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_76) >> 0x4u);

    matrix_fragment_10 = uint32_t(int32_t(packed_half_79) >> 0x4u);

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_8 * 0x4u + matrix_accumulator_105;
    }

    matrix_accumulator_63 =
        uint32_t(combine_words(matrix_accumulator_63, 0u) << 0x1u >> 32) + matrix_fragment_36;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_6 | uniform_condition_2);

    matrix_fragment_76 =
        uint32_t(combine_words(matrix_fragment_76, 0u) << 0x1u >> 32) + matrix_fragment_11;

    packed_half_79 = uint32_t(combine_words(packed_half_79, 0u) << 0x1u >> 32) + matrix_fragment_10;

    split_address_words(matrix_fragment_66, matrix_fragment_67,
                        uint64_t(uint32_t(matrix_fragment_67)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_fragment_37 = uint32_t(int32_t(matrix_fragment_18) >> 0x4u);

    matrix_fragment_49 = !condition_1 ? matrix_fragment_38 : matrix_fragment_49;

    matrix_fragment_13 = !condition_1 ? matrix_fragment_77 : matrix_fragment_42;

    matrix_accumulator_58 = !condition_1 ? matrix_fragment_39 : matrix_accumulator_58;

    matrix_fragment_72 = !condition_1 ? matrix_fragment_41 : matrix_fragment_72;

    matrix_fragment_77 = !condition_1 ? matrix_fragment_42 : matrix_fragment_77;

    matrix_accumulator_65 = uint32_t(int32_t(matrix_accumulator_52) >> 0x1u);

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_9 * 0x80u;
    }

    matrix_fragment_41 = matrix_accumulator_63 & 0x3ffffffeu;

    split_address_words(matrix_fragment_38, matrix_fragment_39,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    matrix_fragment_42 = matrix_fragment_76 & 0x3ffffffeu;

    packed_half_79 = packed_half_79 & 0x3ffffffeu;

    matrix_accumulator_59 = uint32_t(int32_t(index_word_140) >> 0x1fu);

    matrix_accumulator_81 = uint32_t(int32_t(matrix_fragment_12) >> 0x4u);

    split_address_words(matrix_fragment_68, matrix_fragment_69,
                        uint64_t(uint32_t(matrix_fragment_69)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_57 = !condition_1 ? matrix_fragment_20 : matrix_accumulator_33;

    matrix_accumulator_61 = !condition_1 ? matrix_accumulator_33 : matrix_fragment_20;

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x1u >> 32) + matrix_fragment_37;

    matrix_fragment_78 = !condition_1 ? matrix_accumulator_35 : matrix_accumulator_34;

    matrix_fragment_20 = !condition_1 ? matrix_accumulator_34 : matrix_accumulator_35;

    split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                        int64_t(int32_t(matrix_accumulator_65)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_34 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_66, matrix_fragment_67) + 0x20100ull, memory_ranges,
        error_counts);

    matrix_fragment_41 = matrix_fragment_36 + (0u - matrix_fragment_41);

    matrix_accumulator_63 =
        uint32_t(combine_words(matrix_accumulator_59, 0u) << 0x4u >> 32) + index_word_140;

    if (condition_0) {
        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));
    }

    matrix_accumulator_80 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1u >> 32) + matrix_accumulator_81;

    uniform_parameter_word_13 = uniform_condition_5 ? uniform_index_word_12 : 0u;

    matrix_fragment_42 = matrix_fragment_11 + (0u - matrix_fragment_42);

    matrix_fragment_36 = matrix_fragment_10 + (0u - packed_half_79);

    matrix_fragment_12 = matrix_fragment_8 & 0x3ffffffeu;

    matrix_accumulator_35 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_68, matrix_fragment_69) + 0x20100ull, memory_ranges,
        error_counts);

    if (!condition_0) {
        matrix_fragment_8 = uniform_index_word_5;

        matrix_fragment_9 = uniform_index_word_5;

        matrix_fragment_10 = uniform_index_word_5;

        matrix_fragment_11 = uniform_index_word_5;
    }

    uniform_index_word_5 = 0u;

    matrix_accumulator_52 = uint32_t(int32_t(matrix_accumulator_63) >> 0x4u);

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11;
    }

    uniform_parameter_word_13 =
        !uniform_condition_4 ? uniform_parameter_word_13 : uniform_index_word_12;

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_accumulator_33 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x20100ull, memory_ranges,
        error_counts);

    if (condition_0) {
        split_address_words(matrix_fragment_50, matrix_fragment_51,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));
    }

    condition_2 = uniform_condition_0;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_parameter_word_13 + uniform_index_word_5;
    }

    matrix_fragment_76 =
        uint32_t(combine_words(matrix_accumulator_63, 0u) << 0x1u >> 32) + matrix_accumulator_52;

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_50, matrix_fragment_51), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    condition_0 = (address_word_0 & 0x4u) != 0;

    packed_half_79 = matrix_fragment_76 & 0x3ffffffeu;

    matrix_fragment_39 = matrix_fragment_37 + (0u - matrix_fragment_12);

    matrix_fragment_12 = uniform_index_word_5;

    matrix_fragment_76 = !condition_0 ? matrix_fragment_44 : matrix_fragment_15;

    matrix_fragment_38 = matrix_accumulator_52 + (0u - packed_half_79);

    matrix_fragment_44 = !condition_0 ? matrix_fragment_15 : matrix_fragment_44;

    matrix_accumulator_52 = !condition_0 ? matrix_fragment_46 : matrix_accumulator_57;

    matrix_fragment_46 = !condition_0 ? matrix_accumulator_57 : matrix_fragment_46;

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_12 * 0x4u + matrix_accumulator_105;
    }

    matrix_accumulator_57 = !condition_0 ? matrix_fragment_43 : matrix_fragment_78;

    matrix_fragment_43 = !condition_0 ? matrix_fragment_78 : matrix_fragment_43;

    split_address_words(matrix_fragment_78, packed_half_79,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_15 * 0x80u;
    }

    matrix_fragment_50 = !condition_0 ? matrix_fragment_48 : matrix_accumulator_61;

    index_word_145 = uint32_t(int32_t(index_word_143) >> 0x1fu);

    matrix_fragment_48 = !condition_0 ? matrix_accumulator_61 : matrix_fragment_48;

    if (condition_2) {
        split_address_words(matrix_fragment_78, packed_half_79,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_78, packed_half_79));
    }

    matrix_accumulator_61 = !condition_0 ? matrix_fragment_45 : matrix_fragment_20;

    matrix_fragment_45 = !condition_0 ? matrix_fragment_20 : matrix_fragment_45;

    matrix_fragment_20 =
        uint32_t(combine_words(index_word_145, 0u) << 0x4u >> 32) + address_word_144;

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_3) {
        uniform_parameter_word_13 = uniform_index_word_6 + 0x1u;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_78, packed_half_79));
    }

    packed_half_79 = matrix_fragment_20 & 0xfff0u;

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_parameter_word_13) >= int32_t(uniform_index_word_10);
    }

    uniform_condition_6 = false;

    packed_half_79 = address_word_144 + (0u - packed_half_79);

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    uniform_condition_0 = true;

    matrix_fragment_20 = permute_signed(packed_half_79, 0u, 0x8880u);

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    matrix_fragment_78 = !condition_0 ? matrix_accumulator_58 : matrix_fragment_73;

    matrix_accumulator_58 = !condition_0 ? matrix_fragment_73 : matrix_accumulator_58;

    matrix_fragment_20 = permute_signed(matrix_fragment_20, 0u, 0x7710u);

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_2);

    matrix_fragment_73 = permute_signed(matrix_fragment_16, 0u, 0x8880u);

    matrix_fragment_94 = !condition_0 ? matrix_fragment_74 : matrix_accumulator_71;

    matrix_fragment_51 = !condition_0 ? matrix_fragment_49 : matrix_fragment_14;

    matrix_fragment_88 = !condition_0 ? matrix_accumulator_54 : matrix_fragment_13;

    matrix_fragment_74 = !condition_0 ? matrix_accumulator_71 : matrix_fragment_74;

    matrix_fragment_20 = uint32_t(uint32_t(matrix_fragment_20) >> 0xdu);

    matrix_fragment_49 = !condition_0 ? matrix_fragment_14 : matrix_fragment_49;

    matrix_accumulator_54 = !condition_0 ? matrix_fragment_13 : matrix_accumulator_54;

    matrix_accumulator_71 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    matrix_accumulator_80 = matrix_accumulator_80 & 0x3ffffffeu;

    if (!condition_2) {
        matrix_fragment_12 = uniform_index_word_5;

        matrix_fragment_13 = uniform_index_word_5;

        matrix_fragment_14 = uniform_index_word_5;

        matrix_fragment_15 = uniform_index_word_5;
    }

    matrix_fragment_20 = matrix_fragment_20 & 0x3u;

    uniform_index_word_5 = 0u;

    matrix_accumulator_71 = uint32_t(uint32_t(matrix_accumulator_71) >> 0xdu);

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;
    }

    matrix_fragment_37 = matrix_accumulator_81 + (0u - matrix_accumulator_80);

    matrix_fragment_86 = !condition_0 ? matrix_accumulator_60 : matrix_fragment_77;

    if (!uniform_condition_4) {
        uniform_index_word_12 = uniform_condition_6 ? uniform_index_word_12 : 0u;
    }

    if (uniform_condition_0) {
        uniform_index_word_18 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_index_word_19 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_accumulator_80 = !condition_0 ? matrix_fragment_40 : matrix_accumulator_53;

    matrix_accumulator_60 = !condition_0 ? matrix_fragment_77 : matrix_accumulator_60;

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    matrix_fragment_40 = !condition_0 ? matrix_accumulator_53 : matrix_fragment_40;

    condition_2 = uniform_condition_0;

    matrix_accumulator_71 = matrix_accumulator_71 & 0x3u;

    matrix_fragment_77 = packed_half_79 + matrix_fragment_20;

    matrix_accumulator_53 = !condition_0 ? matrix_fragment_47 : matrix_accumulator_82;

    matrix_fragment_47 = !condition_0 ? matrix_accumulator_82 : matrix_fragment_47;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_accumulator_82 = !condition_0 ? matrix_accumulator_56 : matrix_fragment_75;

    matrix_accumulator_56 = !condition_0 ? matrix_fragment_75 : matrix_accumulator_56;

    matrix_fragment_75 = matrix_fragment_16 + matrix_accumulator_71;

    matrix_fragment_20 = matrix_fragment_77 & 0xfcu;

    matrix_fragment_90 = !condition_0 ? matrix_accumulator_62 : matrix_fragment_19;

    matrix_fragment_73 = permute_signed(matrix_accumulator_32, 0u, 0x8880u);

    matrix_fragment_96 = uniform_index_word_5;

    matrix_accumulator_62 = !condition_0 ? matrix_fragment_19 : matrix_accumulator_62;

    matrix_fragment_19 = matrix_fragment_75 & 0xfcu;

    matrix_fragment_20 = 0u - matrix_fragment_20;

    matrix_fragment_73 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    if (condition_2) {
        matrix_accumulator_71 = matrix_fragment_96 * 0x4u + matrix_accumulator_105;
    }

    matrix_fragment_92 = !condition_0 ? matrix_fragment_72 : matrix_fragment_21;

    matrix_fragment_84 = !condition_0 ? matrix_accumulator_70 : matrix_fragment_17;

    matrix_fragment_72 = !condition_0 ? matrix_fragment_21 : matrix_fragment_72;

    matrix_accumulator_81 = 0u - matrix_fragment_19;

    matrix_fragment_96 = permute_signed(matrix_fragment_20, 0u, 0x7710u);

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_accumulator_70 = !condition_0 ? matrix_fragment_17 : matrix_accumulator_70;

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        combine_words(uniform_index_word_18, uniform_index_word_19));

    matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_73) >> 0xdu);

    if (condition_2) {
        matrix_fragment_19 = matrix_accumulator_71 * 0x80u;
    }

    matrix_accumulator_81 = permute_signed(matrix_accumulator_81, 0u, 0x7710u);

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_17 = matrix_fragment_17 & 0x3u;

    if (condition_2) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    matrix_accumulator_71 = packed_half_79 + matrix_fragment_96;

    matrix_fragment_73 = matrix_fragment_16 + matrix_accumulator_81;

    packed_half_79 = matrix_fragment_18 & 0xfff0u;

    matrix_fragment_87 = matrix_accumulator_32 + matrix_fragment_17;

    if (!condition_2) {
        matrix_fragment_16 = uniform_index_word_5;

        matrix_fragment_17 = uniform_index_word_5;

        matrix_fragment_18 = uniform_index_word_5;

        matrix_fragment_19 = uniform_index_word_5;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    matrix_accumulator_83 = permute_signed(matrix_fragment_30, 0u, 0x8880u);

    matrix_accumulator_83 = permute_signed(matrix_accumulator_83, 0u, 0x7710u);

    matrix_fragment_85 = permute_signed(matrix_fragment_29, 0u, 0x8880u);

    packed_half_79 = index_word_142 + (0u - packed_half_79);

    matrix_accumulator_83 = uint32_t(uint32_t(matrix_accumulator_83) >> 0xdu);

    matrix_accumulator_81 = matrix_fragment_87 & 0xfcu;

    matrix_fragment_85 = permute_signed(matrix_fragment_85, 0u, 0x7710u);

    matrix_fragment_89 = permute_signed(packed_half_79, 0u, 0x8880u);

    matrix_accumulator_83 = matrix_accumulator_83 & 0x3u;

    matrix_fragment_91 = 0u - matrix_accumulator_81;

    matrix_fragment_85 = uint32_t(uint32_t(matrix_fragment_85) >> 0xdu);

    matrix_accumulator_81 = matrix_accumulator_63 & 0xfff0u;

    matrix_accumulator_63 = permute_signed(matrix_fragment_89, 0u, 0x7710u);

    matrix_fragment_89 = matrix_fragment_30 + matrix_accumulator_83;

    matrix_fragment_96 = matrix_fragment_85 & 0x3u;

    matrix_fragment_85 = permute_signed(matrix_fragment_91, 0u, 0x7710u);

    matrix_accumulator_83 = index_word_140 + (0u - matrix_accumulator_81);

    matrix_fragment_21 = matrix_fragment_89 & 0xfcu;

    matrix_fragment_91 = matrix_fragment_29 + matrix_fragment_96;

    matrix_accumulator_32 = matrix_accumulator_32 + matrix_fragment_85;

    matrix_accumulator_63 = uint32_t(uint32_t(matrix_accumulator_63) >> 0xdu);

    matrix_fragment_85 = permute_signed(matrix_accumulator_83, 0u, 0x8880u);

    matrix_fragment_93 = 0u - matrix_fragment_21;

    matrix_fragment_95 = permute_signed(matrix_fragment_28, 0u, 0x8880u);

    matrix_accumulator_81 = matrix_fragment_91 & 0xfcu;

    matrix_fragment_20 = matrix_accumulator_63 & 0x3u;

    matrix_fragment_21 = permute_signed(matrix_fragment_85, 0u, 0x7710u);

    matrix_accumulator_63 = permute_signed(matrix_fragment_95, 0u, 0x7710u);

    matrix_fragment_93 = permute_signed(matrix_fragment_93, 0u, 0x7710u);

    matrix_fragment_96 = 0u - matrix_accumulator_81;

    matrix_fragment_21 = uint32_t(uint32_t(matrix_fragment_21) >> 0xdu);

    matrix_accumulator_81 = packed_half_79 + matrix_fragment_20;

    matrix_fragment_85 = uint32_t(uint32_t(matrix_accumulator_63) >> 0xdu);

    matrix_accumulator_63 = matrix_fragment_30 + matrix_fragment_93;

    matrix_fragment_30 = matrix_fragment_21 & 0x3u;

    matrix_fragment_20 = matrix_accumulator_81 & 0xfcu;

    matrix_fragment_85 = matrix_fragment_85 & 0x3u;

    matrix_fragment_21 = permute_signed(matrix_fragment_77, 0u, 0x8880u);

    matrix_fragment_77 = matrix_accumulator_83 + matrix_fragment_30;

    matrix_fragment_95 = 0u - matrix_fragment_20;

    matrix_fragment_93 = matrix_fragment_28 + matrix_fragment_85;

    matrix_fragment_20 = address_word_0;

    matrix_fragment_85 = matrix_fragment_77 & 0xfcu;

    matrix_fragment_30 = permute_signed(matrix_fragment_95, 0u, 0x7710u);

    matrix_fragment_96 = permute_signed(matrix_fragment_96, 0u, 0x7710u);

    matrix_fragment_75 = permute_signed(matrix_fragment_75, 0u, 0x8880u);

    matrix_fragment_20 = uint32_t(int32_t(matrix_fragment_20) >> 0x7u);

    matrix_fragment_85 = 0u - matrix_fragment_85;

    matrix_fragment_21 = uint32_t(int32_t(matrix_fragment_21) >> 0x2u);

    matrix_fragment_30 = packed_half_79 + matrix_fragment_30;

    matrix_fragment_29 = matrix_fragment_29 + matrix_fragment_96;

    packed_half_79 = permute_signed(matrix_fragment_87, 0u, 0x8880u);

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

    matrix_fragment_96 = (matrix_fragment_20 << 0x2u) + uniform_index_word_8;

    matrix_fragment_21 = permute_signed(matrix_fragment_21, 0u, 0x9910u);

    matrix_fragment_20 = permute_signed(matrix_fragment_85, 0u, 0x7710u);

    packed_half_79 = uint32_t(int32_t(packed_half_79) >> 0x2u);

    matrix_fragment_26 = uint32_t(int32_t(matrix_fragment_26) >> 0x5u);

    matrix_fragment_75 = permute_signed(matrix_fragment_75, 0u, 0x9910u);

    matrix_fragment_89 = permute_signed(matrix_fragment_89, 0u, 0x8880u);

    matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_27) >> 0x5u);

    matrix_fragment_20 = matrix_accumulator_83 + matrix_fragment_20;

    matrix_fragment_21 = matrix_fragment_21 + matrix_fragment_96;

    matrix_accumulator_83 = permute_signed(packed_half_79, 0u, 0x9910u);

    matrix_fragment_96 = (matrix_fragment_26 << 0x2u) + uniform_index_word_8;

    matrix_fragment_27 = (matrix_fragment_27 << 0x2u) + uniform_index_word_8;

    matrix_fragment_26 = matrix_fragment_75;

    packed_half_79 = matrix_fragment_89;

    condition_5 = int32_t(matrix_fragment_21) >= int32_t(0u);

    condition_4 = int32_t(matrix_fragment_21) >= int32_t(uniform_index_word_14);

    matrix_fragment_27 = matrix_fragment_26 + matrix_fragment_27;

    matrix_fragment_21 = uint32_t(int32_t(packed_half_79) >> 0x2u);

    matrix_fragment_26 = permute_signed(matrix_accumulator_81, 0u, 0x8880u);

    packed_half_79 = permute_signed(matrix_fragment_21, 0u, 0x9910u);

    matrix_fragment_91 = permute_signed(matrix_fragment_91, 0u, 0x8880u);

    matrix_fragment_21 = matrix_fragment_26;

    index_word_141 = uint32_t(combine_words(index_word_141, 0u) << 0x5u >> 32) + index_word_142;

    matrix_fragment_26 = packed_half_79;

    packed_half_79 = uint32_t(int32_t(matrix_fragment_24) >> 0x5u);

    matrix_fragment_24 = uint32_t(int32_t(matrix_fragment_21) >> 0x2u);

    matrix_accumulator_81 = matrix_fragment_91;

    matrix_fragment_21 = uint32_t(int32_t(index_word_141) >> 0x5u);

    matrix_fragment_77 = permute_signed(matrix_fragment_77, 0u, 0x8880u);

    matrix_fragment_75 = matrix_accumulator_83;

    matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_25) >> 0x5u);

    matrix_fragment_24 = permute_signed(matrix_fragment_24, 0u, 0x9910u);

    matrix_fragment_21 = (matrix_fragment_21 << 0x2u) + uniform_index_word_8;

    matrix_accumulator_81 = uint32_t(int32_t(matrix_accumulator_81) >> 0x2u);

    index_word_138 =
        uint32_t(combine_words(matrix_accumulator_59, 0u) << 0x5u >> 32) + index_word_140;

    matrix_fragment_25 = (matrix_fragment_25 << 0x2u) + uniform_index_word_8;

    matrix_accumulator_59 = matrix_fragment_77;

    matrix_fragment_75 = matrix_fragment_75 + matrix_fragment_96;

    matrix_accumulator_81 = permute_signed(matrix_accumulator_81, 0u, 0x9910u);

    matrix_fragment_24 = matrix_fragment_24 + matrix_fragment_21;

    matrix_fragment_96 = (packed_half_79 << 0x2u) + uniform_index_word_8;

    matrix_fragment_21 = uint32_t(int32_t(index_word_138) >> 0x5u);

    packed_half_79 = permute_signed(matrix_fragment_93, 0u, 0x8880u);

    matrix_accumulator_71 = matrix_accumulator_71 & 0xffffu;

    matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_22) >> 0x5u);

    matrix_fragment_26 = matrix_fragment_26 + matrix_fragment_25;

    matrix_fragment_22 = uint32_t(int32_t(matrix_accumulator_59) >> 0x2u);

    matrix_fragment_25 = matrix_accumulator_81;

    matrix_accumulator_59 = (matrix_fragment_21 << 0x2u) + uniform_index_word_8;

    matrix_fragment_21 = uint32_t(int32_t(packed_half_79) >> 0x2u);

    matrix_accumulator_71 = permute_signed(matrix_accumulator_71, 0u, 0x8880u);

    matrix_fragment_100 = (matrix_fragment_77 << 0x2u) + uniform_index_word_8;

    matrix_fragment_25 = matrix_fragment_25 + matrix_fragment_96;

    matrix_fragment_73 = matrix_fragment_73 & 0xffffu;

    matrix_fragment_77 = permute_signed(matrix_fragment_21, 0u, 0x9910u);

    matrix_fragment_96 = (matrix_fragment_23 << 0x2u) + uniform_index_word_9;

    matrix_fragment_21 = matrix_accumulator_71;

    matrix_fragment_23 = permute_signed(matrix_fragment_73, 0u, 0x8880u);

    matrix_fragment_98 = (matrix_accumulator_55 << 0x2u) + uniform_index_word_9;

    matrix_fragment_96 = matrix_fragment_21 + matrix_fragment_96;

    condition_3 = int32_t(matrix_fragment_27) >= int32_t(0u);

    condition_0 = int32_t(matrix_fragment_27) >= int32_t(uniform_index_word_14);

    matrix_fragment_98 = matrix_fragment_23 + matrix_fragment_98;

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_6);

    condition_2 = int32_t(matrix_fragment_96) >= int32_t(uniform_index_word_15);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    condition_6 = int32_t(matrix_fragment_96) < int32_t(0u) || condition_2;

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_22 = permute_signed(matrix_fragment_22, 0u, 0x9910u);

    matrix_fragment_27 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_4 = !condition_6 ? matrix_fragment_4 : 0u;

    matrix_fragment_23 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_6 = !condition_6 ? matrix_fragment_6 : 0u;

    matrix_fragment_22 = matrix_fragment_22 + matrix_accumulator_59;

    matrix_accumulator_55 = !condition_6 ? matrix_fragment_27 : 0u;

    matrix_fragment_27 = !condition_4 ? matrix_fragment_4 : 0u;

    matrix_fragment_23 = !condition_6 ? matrix_fragment_23 : 0u;

    matrix_accumulator_59 = !condition_4 ? matrix_fragment_6 : 0u;

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_7);

    condition_6 = int32_t(matrix_fragment_98) >= int32_t(uniform_index_word_15);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_7 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_accumulator_32 = matrix_accumulator_32 & 0xffffu;

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    condition_6 = int32_t(matrix_fragment_98) < int32_t(0u) || condition_6;

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_32 = permute_signed(matrix_accumulator_32, 0u, 0x8880u);

    matrix_fragment_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_41 = (matrix_fragment_41 << 0x2u) + uniform_index_word_9;

    matrix_fragment_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_23 = !condition_4 ? matrix_fragment_23 : 0u;

    matrix_accumulator_55 = !condition_4 ? matrix_accumulator_55 : 0u;

    matrix_accumulator_63 = matrix_accumulator_63 & 0xffffu;

    matrix_fragment_76 = !condition_6 ? matrix_fragment_76 : 0u;

    matrix_accumulator_57 = !condition_6 ? matrix_accumulator_57 : 0u;

    matrix_fragment_86 = !condition_6 ? matrix_fragment_86 : 0u;

    matrix_fragment_92 = !condition_6 ? matrix_fragment_92 : 0u;

    matrix_accumulator_32 = matrix_accumulator_32 + matrix_fragment_41;

    matrix_accumulator_63 = permute_signed(matrix_accumulator_63, 0u, 0x8880u);

    matrix_fragment_6 = condition_5 ? matrix_fragment_23 : 0u;

    matrix_fragment_7 = condition_5 ? matrix_fragment_27 : 0u;

    matrix_fragment_4 = condition_5 ? matrix_accumulator_55 : 0u;

    matrix_fragment_5 = condition_5 ? matrix_accumulator_59 : 0u;

    condition_5 = int32_t(matrix_fragment_26) >= int32_t(0u);

    condition_6 = int32_t(matrix_fragment_26) >= int32_t(uniform_index_word_14);

    matrix_fragment_76 = !condition_0 ? matrix_fragment_76 : 0u;

    matrix_accumulator_57 = !condition_0 ? matrix_accumulator_57 : 0u;

    matrix_fragment_23 = !condition_0 ? matrix_fragment_86 : 0u;

    matrix_fragment_92 = !condition_0 ? matrix_fragment_92 : 0u;

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_10);

    condition_0 = int32_t(matrix_accumulator_32) >= int32_t(uniform_index_word_15);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_accumulator_61 = (matrix_fragment_42 << 0x2u) + uniform_index_word_9;

    matrix_fragment_42 = matrix_accumulator_63;

    condition_0 = int32_t(matrix_accumulator_32) < int32_t(0u) || condition_0;

    matrix_accumulator_52 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_53 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_41 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_42 = matrix_fragment_42 + matrix_accumulator_61;

    condition_4 = int32_t(matrix_fragment_75) >= int32_t(uniform_index_word_14);

    matrix_accumulator_52 = !condition_0 ? matrix_accumulator_52 : 0u;

    matrix_accumulator_53 = !condition_0 ? matrix_accumulator_53 : 0u;

    matrix_fragment_41 = !condition_0 ? matrix_fragment_41 : 0u;

    matrix_fragment_90 = !condition_0 ? matrix_fragment_90 : 0u;

    condition_0 = int32_t(matrix_fragment_42) >= int32_t(uniform_index_word_15);

    matrix_fragment_26 = condition_3 ? matrix_fragment_76 : 0u;

    matrix_fragment_8 = condition_3 ? matrix_accumulator_57 : 0u;

    matrix_fragment_23 = condition_3 ? matrix_fragment_23 : 0u;

    matrix_fragment_10 = condition_3 ? matrix_fragment_92 : 0u;

    matrix_accumulator_52 = !condition_4 ? matrix_accumulator_52 : 0u;

    matrix_accumulator_53 = !condition_4 ? matrix_accumulator_53 : 0u;

    matrix_fragment_41 = !condition_4 ? matrix_fragment_41 : 0u;

    matrix_fragment_90 = !condition_4 ? matrix_fragment_90 : 0u;

    condition_3 = int32_t(matrix_fragment_25) >= int32_t(0u);

    condition_4 = int32_t(matrix_fragment_25) >= int32_t(uniform_index_word_14);

    matrix_fragment_95 = matrix_fragment_93 & 0xfcu;

    condition_0 = int32_t(matrix_fragment_42) < int32_t(0u) || condition_0;

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_42 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_30 = matrix_fragment_30 & 0xffffu;

    matrix_fragment_50 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_95 = 0u - matrix_fragment_95;

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_30 = permute_signed(matrix_fragment_30, 0u, 0x8880u);

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_39 = (matrix_fragment_39 << 0x2u) + uniform_index_word_9;

    matrix_fragment_42 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_85 = permute_signed(matrix_fragment_95, 0u, 0x7710u);

    condition_2 = int32_t(matrix_fragment_75) >= int32_t(0u);

    matrix_fragment_50 = !condition_0 ? matrix_fragment_50 : 0u;

    matrix_fragment_9 = !condition_0 ? matrix_fragment_9 : 0u;

    matrix_fragment_78 = !condition_0 ? matrix_fragment_78 : 0u;

    matrix_fragment_42 = !condition_0 ? matrix_fragment_42 : 0u;

    matrix_fragment_30 = matrix_fragment_30 + matrix_fragment_39;

    matrix_fragment_28 = matrix_fragment_28 + matrix_fragment_85;

    matrix_fragment_11 = condition_2 ? matrix_accumulator_52 : 0u;

    matrix_accumulator_53 = condition_2 ? matrix_accumulator_53 : 0u;

    matrix_fragment_25 = condition_2 ? matrix_fragment_41 : 0u;

    matrix_fragment_27 = condition_2 ? matrix_fragment_90 : 0u;

    matrix_fragment_50 = !condition_6 ? matrix_fragment_50 : 0u;

    matrix_accumulator_32 = !condition_6 ? matrix_fragment_9 : 0u;

    matrix_fragment_78 = !condition_6 ? matrix_fragment_78 : 0u;

    matrix_fragment_42 = !condition_6 ? matrix_fragment_42 : 0u;

    condition_2 = int32_t(matrix_fragment_24) >= int32_t(0u);

    condition_0 = int32_t(matrix_fragment_24) >= int32_t(uniform_index_word_14);

    condition_6 = int32_t(matrix_fragment_30) >= int32_t(uniform_index_word_15);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_29 = matrix_fragment_29 & 0xffffu;

    matrix_fragment_28 = matrix_fragment_28 & 0xffffu;

    matrix_fragment_20 = matrix_fragment_20 & 0xffffu;

    condition_6 = int32_t(matrix_fragment_30) < int32_t(0u) || condition_6;

    matrix_fragment_39 = (matrix_fragment_36 << 0x2u) + uniform_index_word_9;

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_30 = permute_signed(matrix_fragment_29, 0u, 0x8880u);

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_36 = permute_signed(matrix_fragment_28, 0u, 0x8880u);

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_20 = permute_signed(matrix_fragment_20, 0u, 0x8880u);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_29 = (matrix_fragment_38 << 0x2u) + uniform_index_word_9;

    matrix_fragment_28 = matrix_fragment_30;

    matrix_fragment_37 = (matrix_fragment_37 << 0x2u) + uniform_index_word_9;

    matrix_fragment_30 = matrix_fragment_36;

    matrix_fragment_9 = !condition_6 ? matrix_fragment_9 : 0u;

    matrix_fragment_12 = !condition_6 ? matrix_fragment_12 : 0u;

    matrix_fragment_24 = !condition_6 ? matrix_fragment_24 : 0u;

    matrix_fragment_14 = !condition_6 ? matrix_fragment_14 : 0u;

    matrix_fragment_20 = matrix_fragment_20 + matrix_fragment_29;

    matrix_fragment_50 = condition_5 ? matrix_fragment_50 : 0u;

    matrix_accumulator_32 = condition_5 ? matrix_accumulator_32 : 0u;

    matrix_fragment_78 = condition_5 ? matrix_fragment_78 : 0u;

    matrix_fragment_42 = condition_5 ? matrix_fragment_42 : 0u;

    condition_5 = int32_t(matrix_fragment_22) >= int32_t(0u);

    condition_6 = int32_t(matrix_fragment_22) >= int32_t(uniform_index_word_14);

    matrix_fragment_39 = matrix_fragment_28 + matrix_fragment_39;

    matrix_fragment_37 = matrix_fragment_30 + matrix_fragment_37;

    matrix_fragment_9 = !condition_0 ? matrix_fragment_9 : 0u;

    matrix_fragment_12 = !condition_0 ? matrix_fragment_12 : 0u;

    matrix_fragment_24 = !condition_0 ? matrix_fragment_24 : 0u;

    matrix_fragment_28 = !condition_0 ? matrix_fragment_14 : 0u;

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_13);

    condition_0 = int32_t(matrix_fragment_20) >= int32_t(uniform_index_word_15);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_22 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    condition_0 = int32_t(matrix_fragment_20) < int32_t(0u) || condition_0;

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_30 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_36 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_22 = !condition_0 ? matrix_fragment_22 : 0u;

    matrix_fragment_13 = !condition_0 ? matrix_fragment_13 : 0u;

    matrix_fragment_30 = !condition_0 ? matrix_fragment_30 : 0u;

    uniform_index_word_8 = 0u;

    matrix_fragment_36 = !condition_0 ? matrix_fragment_36 : 0u;

    matrix_fragment_14 = condition_2 ? matrix_fragment_9 : 0u;

    matrix_fragment_15 = condition_2 ? matrix_fragment_28 : 0u;

    matrix_fragment_20 = condition_2 ? matrix_fragment_24 : 0u;

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_22 = !condition_6 ? matrix_fragment_22 : 0u;

    matrix_fragment_13 = !condition_6 ? matrix_fragment_13 : 0u;

    matrix_fragment_30 = !condition_6 ? matrix_fragment_30 : 0u;

    matrix_fragment_36 = !condition_6 ? matrix_fragment_36 : 0u;

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_19);

    matrix_accumulator_71 = matrix_fragment_77;

    condition_0 = int32_t(matrix_fragment_39) >= int32_t(uniform_index_word_15);

    condition_6 = int32_t(matrix_fragment_37) >= int32_t(uniform_index_word_15);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_21 = matrix_accumulator_71 + matrix_fragment_100;

    matrix_accumulator_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_accumulator_62));

    condition_0 = int32_t(matrix_fragment_39) < int32_t(0u) || condition_0;

    condition_6 = int32_t(matrix_fragment_37) < int32_t(0u) || condition_6;

    matrix_accumulator_34 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_accumulator_70));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_12 = condition_2 ? matrix_fragment_12 : 0u;

    matrix_fragment_22 = condition_5 ? matrix_fragment_22 : 0u;

    matrix_fragment_13 = condition_5 ? matrix_fragment_13 : 0u;

    matrix_fragment_30 = condition_5 ? matrix_fragment_30 : 0u;

    matrix_fragment_36 = condition_5 ? matrix_fragment_36 : 0u;

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_fragment_48));

    condition_5 = int32_t(matrix_fragment_21) >= int32_t(uniform_index_word_14);

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_47 = !condition_0 ? matrix_fragment_47 : 0u;

    matrix_accumulator_62 = !condition_0 ? matrix_accumulator_62 : 0u;

    matrix_accumulator_34 = !condition_6 ? matrix_accumulator_34 : 0u;

    matrix_fragment_31 = !condition_6 ? matrix_fragment_31 : 0u;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = !condition_0 ? matrix_fragment_46 : 0u;

    matrix_fragment_28 = !condition_0 ? matrix_fragment_28 : 0u;

    matrix_fragment_24 = !condition_6 ? matrix_fragment_24 : 0u;

    matrix_fragment_29 = !condition_6 ? matrix_fragment_29 : 0u;

    condition_2 = int32_t(matrix_fragment_21) >= int32_t(0u);

    matrix_fragment_47 = !condition_4 ? matrix_fragment_47 : 0u;

    matrix_accumulator_62 = !condition_4 ? matrix_accumulator_62 : 0u;

    matrix_accumulator_34 = !condition_5 ? matrix_accumulator_34 : 0u;

    matrix_fragment_31 = !condition_5 ? matrix_fragment_31 : 0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    uniform_index_word_5 = 0x400u;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = (uniform_index_word_8 << 0x18u) + uniform_index_word_5;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_46 = !condition_4 ? matrix_fragment_46 : 0u;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = !condition_4 ? matrix_fragment_28 : 0u;

    matrix_fragment_24 = !condition_5 ? matrix_fragment_24 : 0u;

    matrix_fragment_29 = !condition_5 ? matrix_fragment_29 : 0u;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 = condition_3 ? matrix_fragment_47 : 0u;

    matrix_accumulator_62 = condition_3 ? matrix_accumulator_62 : 0u;

    matrix_accumulator_34 = condition_2 ? matrix_accumulator_34 : 0u;

    matrix_fragment_31 = condition_2 ? matrix_fragment_31 : 0u;

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_4 = condition_3 ? matrix_fragment_46 : 0u;

    tile_word_139 = (matrix_accumulator_105 << 0x9u) + uniform_index_word_8;

    matrix_fragment_6 = condition_3 ? matrix_fragment_28 : 0u;

    matrix_fragment_5 = condition_2 ? matrix_fragment_24 : 0u;

    matrix_fragment_7 = condition_2 ? matrix_fragment_29 : 0u;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_34 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_104 = address_word_0 * 0x10u + tile_word_139;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_104, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    uniform_index_word_12 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_13 = parameters[parameter_trained_weights_high_5];

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x800u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1000u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1800u, matrix_fragment_4, matrix_fragment_5,
        matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_70 = 0;
    matrix_accumulator_71 = 0;

    matrix_fragment_72 = 0;
    matrix_fragment_73 = 0;

    matrix_fragment_74 = 0;
    matrix_fragment_75 = 0;

    matrix_fragment_76 = 0;
    matrix_fragment_77 = 0;

    matrix_fragment_78 = 0;
    packed_half_79 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    matrix_fragment_84 = 0;
    matrix_fragment_85 = 0;

    matrix_fragment_86 = 0;
    matrix_fragment_87 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    matrix_fragment_92 = 0;
    matrix_fragment_93 = 0;

    matrix_fragment_94 = 0;
    matrix_fragment_95 = 0;

    matrix_fragment_96 = 0;
    matrix_fragment_97 = 0;

    matrix_fragment_98 = 0;
    matrix_fragment_99 = 0;

    matrix_fragment_100 = 0;
    matrix_fragment_101 = 0;

    split_address_words(matrix_fragment_102, matrix_fragment_103,
                        combine_words(uniform_index_word_12, uniform_parameter_word_13));

    uniform_index_word_5 = 0u;

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            uint64_t(uint32_t(matrix_accumulator_105)) *
                                    uint64_t(uint32_t(0x4000u)) +
                                combine_words(matrix_fragment_102, matrix_fragment_103));

        split_address_words(matrix_accumulator_32, matrix_accumulator_33,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33), matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x200ull,
            matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x1000ull,
            matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x1200ull,
            matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x2000ull,
            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x2200ull,
            matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x3000ull,
            matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x3200ull,
            matrix_fragment_16, matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
            memory_ranges, error_counts);

        matrix_accumulator_106 = (address_word_0 << 0x4u) + uniform_index_word_8;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x800u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_106 + 0x1000u,
                                            matrix_accumulator_52, matrix_accumulator_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_106 + 0x1800u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_106 + 0x200u,
                                            matrix_accumulator_60, matrix_accumulator_61,
                                            matrix_accumulator_62, matrix_accumulator_63,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_106 + 0x1a00u,
                                            matrix_accumulator_32, matrix_accumulator_33,
                                            matrix_accumulator_34, matrix_accumulator_35,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x1200u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_36,
                                 matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_36,
                                 matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_38,
                                 matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_44, matrix_fragment_45, 0u,
                                 0u);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_46,
                                 matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_44,
                                 matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_46,
                                 matrix_fragment_47, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x400u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_60,
            matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_24,
            matrix_fragment_25, matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(
            matrix_accumulator_112, matrix_accumulator_113, matrix_accumulator_60,
            matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_26,
            matrix_fragment_27, matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(
            matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_60,
            matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
            matrix_fragment_21, matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(
            matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_60,
            matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_22,
            matrix_fragment_23, matrix_accumulator_118, matrix_accumulator_119);

        split_address_words(matrix_accumulator_62, matrix_accumulator_63,
                            uint64_t(uint32_t(matrix_accumulator_105)) *
                                    uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_fragment_102, matrix_fragment_103));

        split_address_words(matrix_accumulator_62, matrix_accumulator_63,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_62, matrix_accumulator_63));

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_38,
                                 matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_36,
                                 matrix_fragment_37);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_62, matrix_accumulator_63) + 0x10000ull,
            matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_32, matrix_accumulator_33,
                                 matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_20,
                                 matrix_fragment_21, matrix_accumulator_54, matrix_accumulator_55);

        multiply_accumulate_half(
            matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_22,
            matrix_fragment_23, matrix_accumulator_122, matrix_accumulator_123);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_62, matrix_accumulator_63) + 0x10200ull,
            matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_accumulator_32,
                                 matrix_accumulator_33, matrix_accumulator_34,
                                 matrix_accumulator_35, matrix_fragment_24, matrix_fragment_25,
                                 matrix_accumulator_128, matrix_accumulator_129);

        multiply_accumulate_half(
            matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_26,
            matrix_fragment_27, matrix_accumulator_116, matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_106 + 0x600u,
                                            matrix_accumulator_32, matrix_accumulator_33,
                                            matrix_accumulator_34, matrix_accumulator_35,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x1400u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x1c00u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0xe00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_112, matrix_accumulator_113, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_14,
            matrix_fragment_15, matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(
            matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_12,
            matrix_fragment_13, matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(
            matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_16,
            matrix_fragment_17, matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(
            matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_18,
            matrix_fragment_19, matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_112), 0x44004400u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_113), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                                 matrix_fragment_49);

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_52,
                                 matrix_accumulator_53);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x1600u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_54,
                                 matrix_accumulator_55);

        matrix_fragment_9 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_28 = 0x2b28u;

        matrix_fragment_11 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_9),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_58,
                                 matrix_accumulator_59);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9),
            select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_106 + 0x1e00u, matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_124,
                                 matrix_accumulator_125);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x44004400u);

        matrix_accumulator_112 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_11),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_9 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x44004400u);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        matrix_fragment_30 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_60,
                                 matrix_accumulator_61);

        matrix_fragment_11 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_31 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_58), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_114,
                                 matrix_accumulator_115);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_11), 0x3b283b28u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_accumulator_113 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_accumulator_110 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_9 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_59), 0x44004400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_30),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_56,
                                 matrix_accumulator_57);

        matrix_accumulator_118 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_accumulator_111 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_fragment_11));

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_52,
                                 matrix_accumulator_53);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_11), 0x3b283b28u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_60), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_48,
                                 matrix_fragment_49);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_11 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_accumulator_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_11),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        matrix_fragment_26 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_56), 0x44004400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_15 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_108), 0x44004400u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_57), 0x44004400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_14 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_13 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_14),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_54,
                                 matrix_accumulator_55);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_13),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14),
            select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

        matrix_fragment_17 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_26),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_24 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

        matrix_fragment_17 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13),
            select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

        matrix_fragment_15 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26),
            select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

        matrix_fragment_14 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_26 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_17),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_16 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_14),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_accumulator_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_31),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_17 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_accumulator_115 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14),
            select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_16),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_accumulator_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31),
            select_half_operand<2, false, false>(matrix_accumulator_32), 0x3b283b28u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_18 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_17),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_8 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_accumulator_119 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_accumulator_32));

        matrix_accumulator_124 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_accumulator_62));

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_26),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17),
            select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_125 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26),
            select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

        matrix_fragment_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_108 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_9 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                                select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_accumulator_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                          select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                          select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_18 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_54), 0x44004400u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_55), 0x44004400u);

        matrix_fragment_31 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_fragment_11 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                                select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                          select_half_operand<2, false, false>(matrix_accumulator_32));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_accumulator_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_18 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_accumulator_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_18),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                          select_half_operand<2, false, false>(matrix_accumulator_32));

        matrix_accumulator_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_33),
            select_half_operand<0, false, true>(matrix_fragment_28), 0x37283728u);

        matrix_fragment_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_accumulator_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_accumulator_32), 0x3b283b28u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_33),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_13 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                                select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_12 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

        matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_accumulator_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_accumulator_58));

        matrix_accumulator_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_accumulator_61));

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                          select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_27 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                          select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_19 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                          select_half_operand<2, false, false>(matrix_fragment_28));

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_100,
                                 matrix_fragment_101);

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_76,
                                 matrix_fragment_77);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_74,
                                 matrix_fragment_75);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_94,
                                 matrix_fragment_95);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_fragment_78, packed_half_79, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_78,
                                 packed_half_79);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_72,
                                 matrix_fragment_73);

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_70,
                                 matrix_accumulator_71);

        split_address_words(matrix_fragment_102, matrix_fragment_103,
                            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x400u);

        uniform_index_word_5 = uniform_index_word_9;
    } while (!uniform_condition_0);
    matrix_accumulator_65 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x20000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_accumulator_107)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_fragment_66 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_66, matrix_fragment_67) + 0x20000ull, memory_ranges,
        error_counts);

    matrix_fragment_68 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_68, matrix_fragment_69) + 0x20000ull, memory_ranges,
        error_counts);

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x20000ull, memory_ranges,
        error_counts);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = 0u;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_104, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = 0x400u;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x800u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    split_address_words(packed_half_2, matrix_accumulator_3,
                        combine_words(packed_half_2, matrix_accumulator_3) + 0x14200u);

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_100),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    uniform_index_word_12 = (uniform_index_word_8 << 0x18u) + uniform_index_word_5;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1000u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    uniform_index_word_5 = 0u;

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    uniform_index_word_8 = uniform_index_word_12;

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1800u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104, matrix_fragment_100, matrix_fragment_101,
        matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    packed_half_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                   select_half_operand<2, false, false>(matrix_fragment_22));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x800u, matrix_fragment_92, matrix_fragment_93,
        matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1000u, matrix_fragment_84, matrix_fragment_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_104 + 0x1800u,
                                         matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                         packed_half_79, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_12);

    matrix_accumulator_34 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_11);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_35 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, matrix_accumulator_3));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19) - 0x200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_18, matrix_fragment_19),
                                            matrix_fragment_48, matrix_fragment_49,
                                            matrix_fragment_50, matrix_fragment_51, memory_ranges,
                                            error_counts);

        matrix_fragment_26 = (address_word_0 << 0x4u) + uniform_index_word_8;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(packed_half_2, matrix_accumulator_3,
                            combine_words(packed_half_2, matrix_accumulator_3) + 0x1000u);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_26, matrix_accumulator_52,
                                            matrix_accumulator_53, matrix_accumulator_54,
                                            matrix_accumulator_55, shared_bytes, memory_ranges,
                                            error_counts);

        uniform_index_word_8 = uniform_index_word_8 + 0x200u;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_26 + 0x800u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_9;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_26 + 0x1000u,
                                            matrix_accumulator_60, matrix_accumulator_61,
                                            matrix_accumulator_62, matrix_accumulator_63,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_26 + 0x1800u, matrix_accumulator_64, matrix_accumulator_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_44, matrix_fragment_45,
                                 matrix_fragment_6, matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_20, matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_48, matrix_fragment_49,
                                 matrix_fragment_22, matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_24, matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_44, matrix_fragment_45,
                                 matrix_fragment_4, matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_12, matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_48, matrix_fragment_49,
                                 matrix_fragment_28, matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_30, matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_44, matrix_fragment_45,
                                 matrix_fragment_8, matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_14, matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35,
                                 matrix_accumulator_60, matrix_accumulator_61,
                                 matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                                 matrix_fragment_49, matrix_accumulator_34, matrix_accumulator_35);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_accumulator_64,
                                 matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_accumulator_64,
                                 matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_64,
                                 matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_64,
                                 matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_40,
                                 matrix_fragment_41);
    } while (!uniform_condition_0);
    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                         select_half_operand<2, false, false>(0u));

    __syncthreads();

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                  select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                        uint64_t(uint32_t(matrix_accumulator_105)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                         select_half_operand<2, false, false>(0u));

    split_address_words(address_word_146, address_word_147,
                        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x20c00u);

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_accumulator_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                          select_half_operand<2, false, false>(packed_half_2));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x800u, matrix_fragment_4, matrix_fragment_5,
        matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_accumulator_34 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_104, matrix_fragment_20,
                                         matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(packed_half_2));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1000u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_accumulator_34, matrix_accumulator_35, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_104 + 0x1800u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_34 = 0;
    matrix_accumulator_35 = 0;

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    packed_half_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    packed_half_135 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_accumulator_132 = 0;
    packed_half_133 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    uniform_index_word_5 = 0u;

    uniform_index_word_8 = uniform_index_word_12;

    __syncthreads();
    do {
        split_address_words(address_word_150, packed_half_151,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_146, address_word_147));

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_150, packed_half_151) - 0xa00ull, matrix_accumulator_64,
            matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_150, packed_half_151) - 0x800ull, matrix_fragment_68,
            matrix_fragment_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_150, packed_half_151) - 0x600ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_150, packed_half_151) - 0x400ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, packed_half_79, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_150, packed_half_151) - 0x200ull, matrix_accumulator_80,
            matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(address_word_150, packed_half_151),
                                            matrix_fragment_84, matrix_fragment_85,
                                            matrix_fragment_86, matrix_fragment_87, memory_ranges,
                                            error_counts);

        packed_half_152 = (address_word_0 << 0x4u) + uniform_index_word_8;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(address_word_146, address_word_147,
                            combine_words(address_word_146, address_word_147) + 0x3000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_8 = uniform_index_word_8 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x800u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_9;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x1000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x1800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, packed_half_133, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_64, matrix_accumulator_65,
                                 matrix_accumulator_132, packed_half_133);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_64, matrix_accumulator_65,
                                 matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_66, matrix_fragment_67,
                                 matrix_accumulator_106, matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_68, matrix_fragment_69,
                                 matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_134, packed_half_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_70, matrix_accumulator_71,
                                 matrix_accumulator_134, packed_half_135);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_70, matrix_accumulator_71,
                                 matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_72, matrix_fragment_73,
                                 matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, packed_half_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_74, matrix_fragment_75,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_76, matrix_fragment_77,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, packed_half_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, packed_half_79, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, packed_half_79, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_78, packed_half_79,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_32, matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_56, matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_82, matrix_accumulator_83,
                                 matrix_accumulator_34, matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_82, matrix_accumulator_83,
                                 matrix_accumulator_58, matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_82, matrix_accumulator_83,
                                 matrix_accumulator_122, matrix_accumulator_123);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_84, matrix_fragment_85,
                                 matrix_accumulator_124, matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_126, matrix_accumulator_127);
    } while (!uniform_condition_0);
    matrix_fragment_100 = threadIdx.y;

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                             select_half_operand<2, false, false>(packed_half_135));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    packed_half_79 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                         select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    condition_0 = (address_word_0 & 0x1u) != 0;

    condition_2 = (address_word_0 & 0x2u) != 0;

    split_address_words(matrix_fragment_98, matrix_fragment_99,
                        uint64_t(uint32_t(matrix_fragment_100)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    condition_3 = (address_word_0 & 0x8u) != 0;

    matrix_accumulator_35 = transpose_half_tile(matrix_accumulator_35);

    index_word_145 = uint32_t(combine_words(index_word_145, 0u) << 0x5u >> 32) + address_word_144;

    index_word_141 = index_word_141 & 0xffffffe0u;

    matrix_fragment_98 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_98, matrix_fragment_99) + 0x34200ull, memory_ranges,
        error_counts);

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    address_word_147 = matrix_fragment_100 * 0x800u;

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    split_address_words(address_word_146, address_word_147,
                        uint64_t(uint32_t(address_word_147)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_13 = transpose_half_tile(matrix_fragment_13);

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    split_address_words(address_word_146, address_word_147,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_146, address_word_147));

    address_word_150 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    __syncthreads();

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_37 = transpose_half_tile(matrix_fragment_37);

    matrix_accumulator_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_39 = transpose_half_tile(matrix_fragment_39);

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_12 = transpose_half_tile(matrix_fragment_12);

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_36 = transpose_half_tile(matrix_fragment_36);

    matrix_accumulator_83 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_38 = transpose_half_tile(matrix_fragment_38);

    packed_half_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_59 = transpose_half_tile(matrix_accumulator_59);

    matrix_accumulator_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_121 = transpose_half_tile(matrix_accumulator_121);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_58 = transpose_half_tile(matrix_accumulator_58);

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_125 = transpose_half_tile(matrix_accumulator_125);

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                         select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                                     select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_57 = transpose_half_tile(matrix_accumulator_57);

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_61 = transpose_half_tile(matrix_accumulator_61);

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_63 = transpose_half_tile(matrix_accumulator_63);

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                        select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_122 = transpose_half_tile(matrix_accumulator_122);

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    tile_word_139 = address_word_0 * 0x10u + tile_word_139;

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_81 = result_word;
    }

    packed_half_79 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                         select_half_operand<2, false, false>(matrix_fragment_31));

    uniform_index_word_5 = blockIdx.y;

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_102), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    uniform_index_word_8 = parameters[parameter_optional_control_11];

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_93), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    packed_half_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(packed_half_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_152), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_157 = result_word;
    }

    matrix_accumulator_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                         select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_150), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_83 = result_word;
    }

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_102), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_153 = result_word;
    }

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                       select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_93), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_154 = result_word;
    }

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                     select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_99), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_103 = result_word;
    }

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(matrix_fragment_95));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_155 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_151), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_90 = result_word;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2c200ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2c400ull, matrix_fragment_68,
        matrix_fragment_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2c600ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                         select_half_operand<2, false, false>(packed_half_153));

    packed_half_153 = permute_signed(matrix_fragment_92, matrix_fragment_92, 0x5432u);

    matrix_fragment_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2c800ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, packed_half_79, memory_ranges, error_counts);

    matrix_fragment_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_99 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    address_word_150 = add_half_pairs(select_half_operand<2, false, false>(address_word_150),
                                      select_half_operand<2, false, false>(matrix_accumulator_83));

    packed_half_154 = permute_signed(matrix_fragment_93, matrix_fragment_93, 0x5432u);

    matrix_fragment_103 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_fragment_103));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_99), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_156 = result_word;
    }

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                         select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(packed_half_155));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2ca00ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2cc00ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x4100410u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_95 = permute_signed(matrix_fragment_94, matrix_fragment_94, 0x5432u);

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(packed_half_154));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2ce00ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    packed_half_154 = half_to_float_bits((matrix_fragment_92 & 65535u) * 65537u);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                     select_half_operand<2, false, false>(matrix_fragment_95));

    packed_half_160 = half_to_float_bits((matrix_fragment_92 >> 16) * 65537u);

    packed_half_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_101), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_158 = result_word;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2d000ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(packed_half_156));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_156 = result_word;
    }

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                                         select_half_operand<2, false, false>(packed_half_158));

    packed_half_158 = permute_signed(matrix_fragment_103, matrix_fragment_103, 0x5432u);

    matrix_fragment_103 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                                         select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x4100410u);

    float_bits_165 = half_to_float_bits((packed_half_155 & 65535u) * 65537u);

    packed_half_158 = half_to_float_bits((matrix_fragment_103 & 65535u) * 65537u);

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_103 >> 16) * 65537u);

    float_bits_161 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    float_bits_163 = half_to_float_bits((packed_half_155 >> 16) * 65537u);

    packed_half_158 = rsqrt_bits(packed_half_158);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(packed_half_156));

    packed_half_153 = half_to_float_bits((float_bits_161 & 65535u) * 65537u);

    packed_half_156 = half_to_float_bits((float_bits_161 >> 16) * 65537u);

    packed_half_157 = rsqrt_bits(matrix_fragment_103);

    packed_half_155 = rsqrt_bits(float_bits_165);

    reused_word_164 = rsqrt_bits(float_bits_163);

    packed_half_154 = rsqrt_bits(packed_half_154);

    matrix_fragment_103 = rsqrt_bits(packed_half_160);

    packed_half_153 = rsqrt_bits(packed_half_153);

    packed_half_156 = rsqrt_bits(packed_half_156);

    packed_half_157 = pack_half(packed_half_157, packed_half_158);

    packed_half_155 = pack_half(reused_word_164, packed_half_155);

    matrix_fragment_103 = pack_half(matrix_fragment_103, packed_half_154);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_157));

    packed_half_157 = permute_signed(matrix_fragment_102, matrix_fragment_102, 0x5432u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(packed_half_155));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_153 = pack_half(packed_half_156, packed_half_153);

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(packed_half_155));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                       select_half_operand<2, false, false>(packed_half_157));

    packed_half_2 = permute_signed(matrix_fragment_99, matrix_fragment_99, 0x5432u);

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(packed_half_133));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_133 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_152), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_159 = result_word;
    }

    matrix_accumulator_129 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_99 = permute_signed(matrix_fragment_96, matrix_fragment_96, 0x5432u);

    packed_half_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_2 = pack_half(0u, matrix_fragment_98);

    matrix_accumulator_136 = half_to_float_bits((packed_half_133 & 65535u) * 65537u);

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                        select_half_operand<2, false, false>(0u));

    packed_half_155 = half_to_float_bits((packed_half_133 >> 16) * 65537u);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_6 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_133 = rsqrt_bits(matrix_accumulator_136);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_130 = rsqrt_bits(packed_half_155);

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(0u));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_136 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_151), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 = half_to_float_bits((matrix_accumulator_136 & 65535u) * 65537u);

    matrix_accumulator_132 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 = half_to_float_bits((matrix_accumulator_136 >> 16) * 65537u);

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(matrix_accumulator_132));

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                          select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_131 = rsqrt_bits(matrix_accumulator_134);

    packed_half_135 = pack_half(matrix_accumulator_130, packed_half_133);

    matrix_accumulator_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = rsqrt_bits(matrix_accumulator_136);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_101), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    matrix_accumulator_137 = permute_signed(packed_half_152, packed_half_152, 0x5432u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_135));

    packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = half_to_float_bits((matrix_fragment_7 & 65535u) * 65537u);

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_26 = half_to_float_bits((matrix_fragment_7 >> 16) * 65537u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(packed_half_133));

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_3 = rsqrt_bits(matrix_accumulator_3);

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_150), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_133 = result_word;
    }

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_135));

    matrix_accumulator_130 = pack_half(matrix_fragment_6, matrix_accumulator_131);

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_28 = half_to_float_bits((packed_half_152 & 65535u) * 65537u);

    matrix_fragment_24 = permute_signed(packed_half_151, packed_half_151, 0x5432u);

    packed_half_152 = half_to_float_bits((packed_half_152 >> 16) * 65537u);

    packed_half_135 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_28 = rsqrt_bits(matrix_fragment_28);

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                                         select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    packed_half_151 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_151), 0x4100410u);

    matrix_fragment_27 = rsqrt_bits(packed_half_152);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_29 = pack_half(matrix_fragment_26, matrix_accumulator_3);

    matrix_accumulator_3 = half_to_float_bits((packed_half_151 & 65535u) * 65537u);

    matrix_fragment_26 = permute_signed(matrix_fragment_101, matrix_fragment_101, 0x5432u);

    packed_half_151 = half_to_float_bits((packed_half_151 >> 16) * 65537u);

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(address_word_150),
                                     select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                                         select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_accumulator_3 = rsqrt_bits(matrix_accumulator_3);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_48 = rsqrt_bits(packed_half_151);

    matrix_accumulator_130 = pack_half(matrix_fragment_27, matrix_fragment_28);

    matrix_fragment_28 = permute_signed(packed_half_133, packed_half_133, 0x5432u);

    matrix_accumulator_131 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                     select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101), 0x4100410u);

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x4100410u);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_30 = half_to_float_bits((matrix_fragment_101 & 65535u) * 65537u);

    packed_half_133 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_133), 0x4100410u);

    matrix_accumulator_128 = half_to_float_bits((matrix_accumulator_129 & 65535u) * 65537u);

    matrix_accumulator_129 = half_to_float_bits((matrix_accumulator_129 >> 16) * 65537u);

    matrix_fragment_31 = rsqrt_bits(matrix_fragment_30);

    matrix_fragment_48 = pack_half(matrix_fragment_48, matrix_accumulator_3);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_101 = half_to_float_bits((matrix_fragment_101 >> 16) * 65537u);

    matrix_fragment_30 = half_to_float_bits((packed_half_133 & 65535u) * 65537u);

    packed_half_133 = half_to_float_bits((packed_half_133 >> 16) * 65537u);

    matrix_accumulator_128 = rsqrt_bits(matrix_accumulator_128);

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_129 = rsqrt_bits(matrix_accumulator_129);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = rsqrt_bits(matrix_fragment_30);

    matrix_fragment_48 = rsqrt_bits(packed_half_133);

    matrix_fragment_50 = rsqrt_bits(matrix_fragment_101);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_128 = pack_half(matrix_accumulator_129, matrix_accumulator_128);

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(0u));

    packed_half_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_3 = pack_half(matrix_fragment_48, matrix_accumulator_3);

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_50 = pack_half(matrix_fragment_50, matrix_fragment_31);

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_accumulator_52 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_accumulator_52 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_accumulator_3 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_74,
                             matrix_fragment_75);

    packed_half_2 = permute_signed(packed_half_2, matrix_accumulator_3, 0x5410u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, packed_half_79, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_78,
                             packed_half_79);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_23 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_101 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_22 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_97 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_50 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                          select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_51 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_accumulator_52 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_53 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    packed_half_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_79),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_75 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_74 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_64 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_54 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_55 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_78 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_accumulator_65 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_83 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    packed_half_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x3e473e47u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_fragment_17 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_66 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_69 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_77 = (packed_half_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_67 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_70 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_73 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_71 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_72 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = address_word_0 * 0x4u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_95 = matrix_fragment_19 & 0x1cu;

    matrix_fragment_19 = condition_0 ? matrix_accumulator_3 : matrix_fragment_16;

    matrix_fragment_20 = condition_0 ? matrix_fragment_17 : matrix_fragment_18;

    matrix_accumulator_3 = condition_0 ? matrix_fragment_16 : matrix_accumulator_3;

    matrix_fragment_18 = condition_0 ? matrix_fragment_18 : matrix_fragment_17;

    matrix_fragment_95 =
        uint32_t(int64_t(int32_t(address_word_0)) * (1ll << 0x1du) >> 32) + matrix_fragment_95;

    matrix_fragment_16 = !condition_2 ? matrix_fragment_19 : matrix_fragment_20;

    matrix_fragment_17 = !condition_2 ? matrix_fragment_20 : matrix_fragment_19;

    matrix_fragment_21 = !condition_2 ? matrix_accumulator_3 : matrix_fragment_18;

    matrix_fragment_20 = !condition_2 ? matrix_fragment_18 : matrix_accumulator_3;

    matrix_fragment_94 = 0x1u ^ matrix_fragment_95;

    matrix_fragment_18 = 0x2u ^ matrix_fragment_95;

    matrix_fragment_19 = 0x3u ^ matrix_fragment_95;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(matrix_fragment_18), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    matrix_accumulator_3 = !condition_3 ? matrix_fragment_16 : matrix_fragment_21;

    packed_half_79 = !condition_3 ? matrix_fragment_21 : matrix_fragment_16;

    matrix_accumulator_82 = !condition_3 ? matrix_fragment_17 : matrix_fragment_20;

    matrix_fragment_84 = !condition_3 ? matrix_fragment_20 : matrix_fragment_17;

    matrix_fragment_85 = !condition_1 ? matrix_accumulator_3 : matrix_accumulator_82;

    matrix_fragment_86 = !condition_1 ? packed_half_79 : matrix_fragment_84;

    matrix_accumulator_3 = !condition_1 ? matrix_accumulator_82 : matrix_accumulator_3;

    packed_half_79 = !condition_1 ? matrix_fragment_84 : packed_half_79;

    matrix_fragment_85 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_85 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_85 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_85 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_85),
                                        select_half_operand<1, false, false>(matrix_fragment_85));

    matrix_fragment_85 = permute_signed(matrix_fragment_85, matrix_fragment_85, 0x5410u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x4100410u);

    matrix_accumulator_3 = half_to_float_bits((matrix_fragment_85 & 65535u) * 65537u);

    matrix_fragment_85 = half_to_float_bits((matrix_fragment_85 >> 16) * 65537u);

    matrix_fragment_17 = address_word_144 + 0x8u;

    matrix_accumulator_3 = reciprocal_bits(matrix_accumulator_3);

    matrix_fragment_16 = uint32_t(int32_t(matrix_fragment_17) >> 0x1fu);

    matrix_fragment_86 = reciprocal_bits(matrix_fragment_85);

    matrix_fragment_20 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x5u >> 32) + matrix_fragment_17;

    matrix_fragment_20 = matrix_fragment_20 & 0xffffffe0u;

    index_word_145 = index_word_145 & 0xffffffe0u;

    matrix_fragment_91 = matrix_fragment_17 + (0u - matrix_fragment_20);

    matrix_fragment_86 = pack_half(matrix_fragment_86, matrix_accumulator_3);

    matrix_accumulator_3 = address_word_144 + (0u - index_word_145);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_91), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_21 = transpose_half_tile(matrix_accumulator_34);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    matrix_fragment_85 = transpose_half_tile(matrix_fragment_9);

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_84 = transpose_half_tile(matrix_fragment_8);

    matrix_accumulator_35 = transpose_half_tile(matrix_accumulator_33);

    matrix_fragment_87 = transpose_half_tile(matrix_accumulator_32);

    packed_half_79 = permute_signed(matrix_fragment_91, matrix_fragment_91, 0x5410u);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_82 = permute_signed(matrix_fragment_88, matrix_fragment_88, 0x5410u);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_accumulator_34 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_85 = index_word_138 & 0xffffffe0u;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_20 = transpose_half_tile(matrix_fragment_10);

    index_word_141 = index_word_142 + (0u - index_word_141);

    matrix_fragment_87 = index_word_140 + (0u - matrix_fragment_85);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(index_word_141), "r"(0x1fu), "r"(__activemask()));
        index_word_141 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_87), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_89 = transpose_half_tile(matrix_accumulator_120);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_91 = transpose_half_tile(matrix_accumulator_124);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_84 = permute_signed(index_word_141, index_word_141, 0x5410u);

    matrix_fragment_85 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5410u);

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                      select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_88 = transpose_half_tile(matrix_accumulator_56);

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_90 = transpose_half_tile(matrix_accumulator_60);

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_92 = transpose_half_tile(matrix_accumulator_62);

    matrix_fragment_93 = transpose_half_tile(matrix_accumulator_126);

    matrix_accumulator_65 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_59));

    packed_half_79 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                         select_half_operand<2, false, false>(packed_half_79));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_79),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                      select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_34,
                             matrix_accumulator_35);

    matrix_fragment_96 = matrix_fragment_100 * 0x20u;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_57 = index_word_143 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_97 = uint32_t(uint32_t(matrix_fragment_96) >> 0x1u);

    matrix_accumulator_57 = 0u - matrix_accumulator_57 + address_word_0;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_99 = matrix_fragment_97 + 0x8u;

    matrix_fragment_101 = matrix_fragment_97 + 0x4u;

    matrix_accumulator_56 = index_word_143 & 0xfffffffcu;

    matrix_accumulator_57 = matrix_accumulator_57 * 0x2u + matrix_fragment_96;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_accumulator_57 = uint32_t(int32_t(matrix_accumulator_57) >> 0x1u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_97 = matrix_fragment_97 + 0xcu;

    matrix_fragment_11 = matrix_fragment_99 + address_word_0 + (0u - matrix_accumulator_56);

    matrix_fragment_9 = matrix_fragment_101 + address_word_0 + (0u - matrix_accumulator_56);

    split_address_words(matrix_accumulator_118, matrix_accumulator_119,
                        int64_t(int32_t(matrix_accumulator_57)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_accumulator_53 = matrix_fragment_97 + address_word_0 + (0u - matrix_accumulator_56);

    matrix_accumulator_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_52, matrix_accumulator_53,
                        uint64_t(uint32_t(matrix_accumulator_53)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    matrix_accumulator_118 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_118, matrix_accumulator_119) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_98 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_52, matrix_accumulator_53) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_accumulator_33));

    packed_half_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                   select_half_operand<2, false, false>(matrix_fragment_38));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_139 + 0x800u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_accumulator_34, matrix_accumulator_35, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                      select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_139, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_139, matrix_fragment_76,
                                         matrix_fragment_77, matrix_fragment_78, packed_half_79,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_139 + 0x800u, matrix_accumulator_80, matrix_accumulator_81,
        matrix_accumulator_82, matrix_accumulator_83, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_113 = matrix_fragment_100 * 0x100u;

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        uint64_t(uint32_t(matrix_accumulator_113)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_148, address_word_149));

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_112, matrix_accumulator_113));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x34210ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x34410ull,
        matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x35210ull,
        matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x35410ull,
        matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x36210ull,
        matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x36410ull,
        matrix_fragment_68, matrix_fragment_69, matrix_accumulator_70, matrix_accumulator_71,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x37210ull,
        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x37410ull,
        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, packed_half_79, memory_ranges,
        error_counts);

    matrix_accumulator_119 = (address_word_0 << 0x4u) + uniform_index_word_12;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119, matrix_accumulator_80, matrix_accumulator_81,
        matrix_accumulator_82, matrix_accumulator_83, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_84 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_85 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_index_word_8;

    uniform_index_word_9 = uniform_index_word_6 * uniform_index_word_11 + uniform_address_word_7;

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x4u) && !uniform_condition_0;

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_80,
                             matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0x800u, matrix_fragment_84, matrix_fragment_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_80,
                             matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_80,
                             matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_8 = unpack_fp8_bits(matrix_accumulator_32);

    matrix_fragment_9 = unpack_fp8_bits(matrix_accumulator_33);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_80,
                             matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0x200u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0xa00u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_34);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_35);

    matrix_accumulator_34 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

    matrix_accumulator_35 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_34,
                             matrix_accumulator_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0xc00u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_accumulator_34, matrix_accumulator_35, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_80,
                             matrix_accumulator_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0x400u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_124,
                             matrix_accumulator_125);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0xe00u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_80,
                             matrix_accumulator_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_119 + 0x600u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_accumulator_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_63));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_124,
                             matrix_accumulator_125);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                       select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_accumulator_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_81), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_62 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_60), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_61 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_65 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_63), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_62));

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_120,
                             matrix_accumulator_121);

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                       select_half_operand<2, false, false>(matrix_accumulator_61));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_81), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_124,
                             matrix_accumulator_125);

    uniform_condition_3 = uniform_condition_0 & uniform_condition_1;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_78, packed_half_79, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_38 = parameters[parameter_output_low_2];
    matrix_fragment_39 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_67 = uniform_index_word_9;

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_78, packed_half_79, matrix_accumulator_52,
                             matrix_accumulator_53);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    matrix_fragment_72 = matrix_fragment_100 * 0x80u;

    matrix_accumulator_61 = matrix_fragment_67 * 0x200u + matrix_fragment_72;

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_38, matrix_fragment_39,
                        int64_t(int32_t(matrix_accumulator_61)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_38, matrix_fragment_39));
    if (uniform_condition_3) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_accumulator_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        split_address_words(matrix_accumulator_32, matrix_accumulator_33,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_9 = matrix_accumulator_34;

        matrix_fragment_11 = matrix_accumulator_55;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_32, matrix_accumulator_33), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_accumulator_57 = permute_signed(matrix_accumulator_57, matrix_fragment_10, 0x5410u);

        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));

        matrix_accumulator_59 = permute_signed(matrix_accumulator_59, matrix_fragment_8, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_accumulator_56, matrix_fragment_11, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_58, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = matrix_accumulator_59;

        matrix_fragment_11 = matrix_accumulator_57;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x800ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2d200ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2d400ull, matrix_accumulator_32,
        matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2d600ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2d800ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2da00ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2dc00ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2de00ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_146, address_word_147) + 0x2e000ull, matrix_fragment_68,
        matrix_fragment_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_16 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x5u >> 32) + matrix_fragment_17;

    packed_half_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                    select_half_operand<2, false, false>(packed_half_79));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_74 = permute_signed(matrix_accumulator_81, matrix_accumulator_81, 0x5432u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_80 = permute_signed(packed_half_79, packed_half_79, 0x5432u);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_fragment_74));

    packed_half_79 = add_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                                    select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81), 0x4100410u);

    packed_half_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x4100410u);

    matrix_fragment_74 = half_to_float_bits((matrix_accumulator_81 & 65535u) * 65537u);

    matrix_accumulator_81 = half_to_float_bits((matrix_accumulator_81 >> 16) * 65537u);

    matrix_fragment_75 = half_to_float_bits((packed_half_79 & 65535u) * 65537u);

    packed_half_79 = half_to_float_bits((packed_half_79 >> 16) * 65537u);

    matrix_fragment_74 = rsqrt_bits(matrix_fragment_74);

    matrix_accumulator_81 = rsqrt_bits(matrix_accumulator_81);

    matrix_fragment_75 = rsqrt_bits(matrix_fragment_75);

    matrix_accumulator_80 = rsqrt_bits(packed_half_79);

    matrix_accumulator_82 = pack_half(matrix_accumulator_81, matrix_fragment_74);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_80 = pack_half(matrix_accumulator_80, matrix_fragment_75);

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_40 = permute_signed(matrix_fragment_40, matrix_fragment_75, 0x5410u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = permute_signed(matrix_fragment_41, matrix_fragment_42, 0x5410u);

    matrix_fragment_42 = permute_signed(matrix_fragment_44, matrix_fragment_43, 0x5410u);

    matrix_fragment_43 = permute_signed(matrix_fragment_45, matrix_fragment_46, 0x5410u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_9 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x4100410u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_76 & 65535u) * 65537u);

    matrix_fragment_10 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_76 = half_to_float_bits((matrix_fragment_76 >> 16) * 65537u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_fragment_9 = rsqrt_bits(matrix_fragment_76);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_73 & 65535u) * 65537u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_73 >> 16) * 65537u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_34,
                             matrix_accumulator_35);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_fragment_8 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_accumulator_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_32 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3c203c20u);

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_32 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3e473e47u);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_9 = pack_half(matrix_fragment_73, matrix_fragment_10);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_9 = permute_signed(matrix_accumulator_105, matrix_fragment_10, 0x5410u);

    matrix_accumulator_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_35),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_10 = permute_signed(matrix_accumulator_108, matrix_fragment_11, 0x5410u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_34 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34), 0x3c203c20u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_33 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3c203c20u);

    matrix_accumulator_34 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34), 0x3e473e47u);

    matrix_accumulator_35 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_35), 0x3c203c20u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_33 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3e473e47u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_35 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_35), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_11 = permute_signed(matrix_accumulator_109, matrix_fragment_40, 0x5410u);

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_accumulator_52 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_accumulator_53 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3c203c20u);

    matrix_accumulator_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3e473e47u);

    matrix_accumulator_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3e473e47u);

    matrix_accumulator_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_54 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3c203c20u);

    matrix_accumulator_55 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3e473e47u);

    matrix_accumulator_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3e473e47u);

    matrix_fragment_4 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_fragment_7 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_47 = (matrix_accumulator_54 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_fragment_24 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_26 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_fragment_28 = (matrix_accumulator_32 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_29 = (matrix_accumulator_33 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_27 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_fragment_31 = (matrix_accumulator_34 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                           select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_30 = (matrix_accumulator_35 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_33 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                       select_half_operand<2, false, false>(matrix_fragment_31));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_40 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_accumulator_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_9 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_33 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_10 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_8 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_34 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_35 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_36 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_37 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_32 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_33 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_11 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_43 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_2 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_39 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    matrix_fragment_44 = (matrix_accumulator_52 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_42 = (matrix_accumulator_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = (matrix_accumulator_55 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_38 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_40 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_43 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_41 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_55 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_48 = condition_0 ? matrix_fragment_49 : matrix_fragment_46;

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_46 = condition_0 ? matrix_fragment_46 : matrix_fragment_49;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_49 = condition_0 ? matrix_accumulator_54 : matrix_accumulator_53;

    matrix_accumulator_53 = condition_0 ? matrix_accumulator_53 : matrix_accumulator_54;

    matrix_accumulator_52 = !condition_2 ? matrix_fragment_48 : matrix_fragment_49;

    matrix_accumulator_55 = !condition_2 ? matrix_fragment_46 : matrix_accumulator_53;

    matrix_fragment_49 = !condition_2 ? matrix_fragment_49 : matrix_fragment_48;

    matrix_fragment_46 = !condition_2 ? matrix_accumulator_53 : matrix_fragment_46;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_52), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_52 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_55), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_55 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_18), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_48 = !condition_3 ? matrix_accumulator_52 : matrix_accumulator_55;

    matrix_accumulator_53 = !condition_3 ? matrix_accumulator_55 : matrix_accumulator_52;

    matrix_accumulator_57 = !condition_3 ? matrix_fragment_49 : matrix_fragment_46;

    matrix_accumulator_54 = !condition_3 ? matrix_fragment_46 : matrix_fragment_49;

    matrix_fragment_46 = matrix_fragment_16 & 0xffffffe0u;

    matrix_accumulator_56 = !condition_1 ? matrix_fragment_48 : matrix_accumulator_57;

    matrix_accumulator_59 = !condition_1 ? matrix_accumulator_53 : matrix_accumulator_54;

    matrix_fragment_48 = !condition_1 ? matrix_accumulator_57 : matrix_fragment_48;

    matrix_accumulator_53 = !condition_1 ? matrix_accumulator_54 : matrix_accumulator_53;

    matrix_fragment_17 = matrix_fragment_17 + (0u - matrix_fragment_46);

    matrix_fragment_46 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_49 = uint32_t(int32_t(matrix_fragment_46) >> 0x1fu);

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_49 =
        uint32_t(combine_words(matrix_fragment_49, 0u) << 0x5u >> 32) + matrix_fragment_46;

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_49 = matrix_fragment_49 & 0xffffffe0u;

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_56),
                       select_half_operand<1, false, false>(matrix_accumulator_56));

    matrix_fragment_49 = matrix_fragment_46 + (0u - matrix_fragment_49);

    matrix_accumulator_56 = permute_signed(matrix_accumulator_56, matrix_accumulator_56, 0x5410u);

    matrix_accumulator_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56), 0x4100410u);

    matrix_fragment_18 = half_to_float_bits((matrix_accumulator_56 & 65535u) * 65537u);

    matrix_accumulator_56 = half_to_float_bits((matrix_accumulator_56 >> 16) * 65537u);

    matrix_fragment_18 = reciprocal_bits(matrix_fragment_18);

    matrix_fragment_19 = reciprocal_bits(matrix_accumulator_56);

    matrix_fragment_16 = pack_half(matrix_fragment_19, matrix_fragment_18);

    matrix_fragment_19 = uint32_t(int64_t(int32_t(index_word_143)) * (1ll << 0x1eu) >> 32) + 0x10u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_fragment_18 = uint32_t(int32_t(matrix_fragment_19) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    matrix_fragment_18 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x5u >> 32) + matrix_fragment_19;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_18 = matrix_fragment_18 & 0xffffffe0u;

    matrix_fragment_19 = matrix_fragment_19 + (0u - matrix_fragment_18);

    matrix_fragment_46 = permute_signed(matrix_fragment_17, matrix_fragment_17, 0x5410u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_accumulator_3 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5410u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_49 = permute_signed(matrix_fragment_49, matrix_fragment_49, 0x5410u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_48 = permute_signed(matrix_fragment_17, matrix_fragment_17, 0x5410u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                          select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_accumulator_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                      select_half_operand<2, false, false>(matrix_accumulator_35));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_139 + 0x1000u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = index_word_143 & 0x7ffffffcu;

    index_word_143 = index_word_143 & 0xfffffffcu;

    matrix_fragment_101 = matrix_fragment_101 + address_word_0 + (0u - index_word_143);

    matrix_fragment_99 = matrix_fragment_99 + address_word_0 + (0u - index_word_143);

    matrix_fragment_97 = matrix_fragment_97 + address_word_0 + (0u - index_word_143);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                             matrix_fragment_17);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_5 = 0u - matrix_accumulator_3 + address_word_0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = matrix_fragment_5 * 0x2u + matrix_fragment_96;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_92, matrix_fragment_93, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_96) >> 0x1u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_22,
                             matrix_fragment_23);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_139, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_24,
                             matrix_fragment_25);

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_139 + 0x1800u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_accumulator_34, matrix_accumulator_35, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_99)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_accumulator_53 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_101)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    split_address_words(packed_half_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_fragment_97)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_139 + 0x800u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_52 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x38210ull, memory_ranges,
        error_counts);

    matrix_accumulator_54 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, matrix_accumulator_3) + 0x38210ull, memory_ranges,
        error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x34210ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x34410ull,
        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x35210ull,
        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x35410ull,
        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x36210ull,
        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x36410ull,
        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x37210ull,
        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x37410ull,
        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
        error_counts);

    uniform_index_word_9 = 0u;

    uniform_index_word_8 = 0x400u;

    uniform_index_word_8 = (uniform_index_word_9 << 0x18u) + uniform_index_word_8;

    matrix_accumulator_55 = (address_word_0 << 0x4u) + uniform_index_word_8;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55, matrix_accumulator_56, matrix_accumulator_57,
        matrix_accumulator_58, matrix_accumulator_59, shared_bytes, memory_ranges, error_counts);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_44);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_45);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x800u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_44 >> 16);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_45 >> 16);

    uniform_index_word_11 = uniform_index_word_6 * uniform_index_word_11 + uniform_index_word_11;

    uniform_index_word_6 = uniform_index_word_6 + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_10);

    uniform_address_word_7 = uniform_address_word_7 + uniform_index_word_11;

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x8u) && !uniform_condition_0;

    uniform_condition_1 = uniform_condition_0 & uniform_condition_1;

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_36, matrix_fragment_37, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_46);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_47);

    matrix_fragment_46 = unpack_fp8_bits(matrix_fragment_46 >> 16);

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_47 = unpack_fp8_bits(matrix_fragment_47 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_20, matrix_fragment_21, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_accumulator_32);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_33);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x200u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_46,
                             matrix_fragment_47);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xa00u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_accumulator_34);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_accumulator_34 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

    matrix_accumulator_35 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, packed_half_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_34,
                             matrix_accumulator_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x400u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_24, matrix_fragment_25, packed_half_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_64,
                             matrix_accumulator_65);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xe00u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x600u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, packed_half_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_9 = threadIdx.x;

    matrix_fragment_8 = threadIdx.z;

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_4, matrix_fragment_5, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_56,
                             matrix_accumulator_57);

    condition_0 = (matrix_fragment_100 | matrix_fragment_8 | matrix_fragment_9) != 0;

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_5 = uniform_address_word_7;

    matrix_fragment_5 = matrix_fragment_5 * 0x200u + matrix_fragment_72;

    split_address_words(packed_half_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));
    if (uniform_condition_1) {
        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, matrix_accumulator_3));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_6, 0x5410u);

        split_address_words(packed_half_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, matrix_accumulator_3));

        matrix_fragment_37 = permute_signed(matrix_fragment_37, matrix_fragment_4, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_12, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_36, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = matrix_fragment_37;

        matrix_fragment_7 = matrix_fragment_13;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, matrix_accumulator_3) + 0x800ull, matrix_fragment_4,
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

    split_address_words(packed_half_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_6, uniform_address_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(165) void phoenix_window_attention_channels_128_upsample(KernelArguments parameters) {
    run_window_attention_channels_128_upsample<false>(parameters.words, nullptr, nullptr);
}
