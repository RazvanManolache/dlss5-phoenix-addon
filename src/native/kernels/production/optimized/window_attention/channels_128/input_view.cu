// DLSS5 Phoenix optimized kernel: window_attention/channels_128/input_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
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
    parameter_tensor_height_8 = 8,
    parameter_tensor_width_9 = 9,
    parameter_optional_control_10 = 10,
    parameter_optional_control_11 = 11,
    parameter_publication_optional_low_16 = 16,
    parameter_publication_optional_high_17 = 17,
    parameter_optional_control_20 = 20,
    parameter_optional_control_21 = 21,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_128_input_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
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
    uint32_t matrix_accumulator_54 = 0;
    uint32_t matrix_accumulator_55 = 0;
    uint32_t matrix_accumulator_56 = 0;
    uint32_t matrix_accumulator_57 = 0;
    uint32_t matrix_accumulator_58 = 0;
    uint32_t matrix_accumulator_59 = 0;
    uint32_t matrix_accumulator_60 = 0;
    uint32_t matrix_accumulator_61 = 0;
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
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
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
    uint32_t packed_half_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t packed_half_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t address_word_138 = 0;
    uint32_t parameter_word_139 = 0;
    uint32_t address_word_140 = 0;
    uint32_t index_word_141 = 0;
    uint32_t index_word_142 = 0;
    uint32_t tile_word_143 = 0;
    uint32_t index_word_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t index_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t address_word_148 = 0;
    uint32_t address_word_149 = 0;
    uint32_t address_word_150 = 0;
    uint32_t address_word_151 = 0;
    uint32_t address_word_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t float_bits_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_parameter_word_8 = 0;
    uint32_t uniform_parameter_word_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
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

    address_word_138 = threadIdx.x & 31u;

    matrix_fragment_8 = parameters[parameter_optional_control_20];
    matrix_fragment_9 = parameters[parameter_optional_control_21];

    uniform_index_word_6 = parameters[parameter_optional_control_10];
    uniform_parameter_word_7 = parameters[parameter_optional_control_11];

    matrix_fragment_31 = matrix_fragment_31 & 0xffffffdfu;

    packed_half_2 = blockIdx.y;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    address_word_140 = threadIdx.y;

    parameter_word_139 = parameters[parameter_tensor_height_8];

    uniform_index_word_4 = blockIdx.x;

    packed_half_0 = parameters[parameter_tensor_width_9];

    condition_0 = int32_t(matrix_fragment_8) > int32_t(0u);

    condition_1 = int32_t(matrix_fragment_9) > int32_t(0u);

    matrix_fragment_4 = address_word_138;

    index_word_144 = address_word_138;

    matrix_accumulator_3 = (packed_half_2 << 0x3u) + uniform_parameter_word_7;

    matrix_fragment_5 = condition_0 ? matrix_fragment_8 : parameter_word_139;

    address_word_147 = uint32_t(int32_t(index_word_144) >> 0x2u);

    matrix_fragment_12 = address_word_140 * 0x2u;

    matrix_fragment_6 =
        uint32_t(int64_t(int32_t(matrix_fragment_4)) * (1ll << 0x1cu) >> 32) + matrix_accumulator_3;

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_index_word_6;

    matrix_fragment_4 =
        uint32_t(combine_words(address_word_147, 0u) << 0x2u >> 32) + address_word_147;

    matrix_fragment_14 = matrix_fragment_12 + 0x1u;

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(matrix_fragment_5);

    matrix_fragment_4 = matrix_fragment_4 & 0xfffffffcu;

    matrix_fragment_7 = condition_1 ? matrix_fragment_9 : packed_half_0;

    matrix_fragment_20 = matrix_fragment_6 + 0x2u;

    condition_0 = int32_t(matrix_fragment_6) < int32_t(0u) || condition_0;

    matrix_fragment_16 = address_word_147 + uniform_index_word_4 + (0u - matrix_fragment_4);

    matrix_fragment_26 = matrix_fragment_6 + 0x4u;

    condition_2 = int32_t(matrix_fragment_20) >= int32_t(matrix_fragment_5);

    condition_3 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_0;

    condition_5 = int32_t(matrix_fragment_16) >= int32_t(matrix_fragment_7);

    condition_0 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_0;

    matrix_fragment_24 = matrix_fragment_16 + 0x4u;

    condition_2 = int32_t(matrix_fragment_20) < int32_t(0u) || condition_2;

    condition_1 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) || condition_3;

    condition_5 = int32_t(matrix_fragment_16) > -1 && !condition_5;

    matrix_fragment_28 = matrix_fragment_6 + 0x6u;

    condition_6 = int32_t(matrix_fragment_24) >= int32_t(matrix_fragment_7);

    matrix_fragment_13 = matrix_fragment_7 * 0x4u;

    if (condition_3) {
        matrix_fragment_31 = 0x20u | matrix_fragment_31;
    }

    condition_3 = condition_0 & (condition_1 | condition_5);

    matrix_fragment_31 = matrix_fragment_31 & 0xfffffffeu;

    condition_6 = int32_t(matrix_fragment_24) > -1 && !condition_6;

    condition_0 = condition_0 & (condition_1 | condition_6);

    if (condition_3) {
        matrix_fragment_31 = 0x1u | matrix_fragment_31;
    }

    condition_3 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_2;

    condition_2 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_2;

    matrix_fragment_31 = matrix_fragment_31 & 0xffffffbfu;

    matrix_fragment_25 = uint32_t(condition_0);

    if (condition_3) {
        matrix_fragment_31 = 0x40u | matrix_fragment_31;
    }

    condition_3 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) || condition_3;

    condition_1 = condition_2 & (condition_3 | condition_5);

    condition_2 = condition_2 & (condition_3 | condition_6);

    condition_3 = int32_t(matrix_fragment_26) >= int32_t(matrix_fragment_5);

    matrix_fragment_27 = uint32_t(condition_1) << 1;

    matrix_fragment_23 = uint32_t(condition_2) << 2;

    condition_3 = int32_t(matrix_fragment_26) < int32_t(0u) || condition_3;

    condition_1 = (matrix_fragment_31 & 0x20u) != 0;

    condition_2 = (matrix_fragment_31 & 0x1u) != 0;

    condition_4 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_3;

    matrix_fragment_31 = matrix_fragment_31 & 0xffffffefu;

    condition_0 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_3;

    condition_1 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) && !condition_1;

    if (condition_2) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];

        matrix_fragment_17 = index_word_144 & 0xfffffffcu;

        matrix_fragment_21 = matrix_fragment_16 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_31 = 0x10u | matrix_fragment_31;
    }

    condition_4 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) || condition_4;

    if (condition_2) {
        matrix_fragment_19 = index_word_144 & 0xfffffffcu;
    }

    matrix_fragment_31 = matrix_fragment_31 & 0xfffffff7u;

    if (condition_2) {
        matrix_fragment_10 = parameters[parameter_input_low_0];
        matrix_fragment_11 = parameters[parameter_input_high_1];
    }

    condition_3 = condition_0 & (condition_4 | condition_5);

    condition_4 = condition_0 & (condition_4 | condition_6);

    condition_0 = int32_t(matrix_fragment_28) >= int32_t(matrix_fragment_5);

    matrix_fragment_29 = uint32_t(condition_3) << 3;

    if (condition_2) {
        matrix_fragment_4 = 0u - matrix_fragment_17 + address_word_138;

        matrix_fragment_19 = 0u - matrix_fragment_19 + address_word_138;
    }

    condition_0 = int32_t(matrix_fragment_28) < int32_t(0u) || condition_0;

    condition_3 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_0;

    condition_0 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_0;

    if (condition_3) {
        matrix_fragment_31 = 0x8u | matrix_fragment_31;
    }

    condition_3 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) || condition_3;

    condition_5 = condition_0 & (condition_3 | condition_5);

    condition_6 = condition_0 & (condition_3 | condition_6);

    condition_0 = uint32_t(matrix_fragment_5) != uint32_t(0x1u);

    condition_3 = (matrix_fragment_31 & 0x40u) != 0;

    matrix_fragment_30 = uint32_t(condition_5) << 5;

    matrix_fragment_15 = condition_0 ? matrix_fragment_6 : 0u;

    if (condition_2) {
        matrix_fragment_6 = matrix_fragment_16 * 0x4u;
    }

    matrix_fragment_20 = condition_0 ? matrix_fragment_20 : 0u;

    matrix_fragment_26 = condition_0 ? matrix_fragment_26 : 0u;

    matrix_fragment_28 = condition_0 ? matrix_fragment_28 : 0u;

    if (condition_2) {
        matrix_fragment_18 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_15;
    }

    condition_0 = int32_t(matrix_fragment_25) != int32_t(0u);

    if (condition_2) {
        matrix_fragment_22 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_15;

        matrix_fragment_6 = !condition_1 ? matrix_fragment_6 : 0u;

        matrix_fragment_17 = matrix_fragment_13 * matrix_fragment_18 + matrix_fragment_4;
    }

    matrix_fragment_4 = 0u;

    if (condition_2) {
        matrix_fragment_22 = matrix_fragment_13 * matrix_fragment_22 + matrix_fragment_19;

        matrix_fragment_19 = !condition_1 ? matrix_fragment_21 : 0u;
    }

    condition_3 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) && !condition_3;

    if (condition_2) {
        matrix_fragment_17 = matrix_fragment_6 + matrix_fragment_17;
    }

    matrix_fragment_6 = 0u;

    condition_5 = (matrix_fragment_31 & 0x10u) != 0;

    if (condition_2) {
        matrix_fragment_19 = matrix_fragment_19 + matrix_fragment_22;

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_0) {
        matrix_fragment_17 = matrix_fragment_24 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_4 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges, error_counts);
    }

    if (condition_0) {
        matrix_fragment_18 = matrix_fragment_24 * 0x4u;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (condition_0) {
        matrix_fragment_34 = !condition_1 ? matrix_fragment_17 : 0u;

        matrix_fragment_41 = !condition_1 ? matrix_fragment_18 : 0u;
    }

    condition_1 = int32_t(matrix_fragment_27) != int32_t(0u);

    if (condition_2) {
        matrix_fragment_6 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), memory_ranges, error_counts);
    }

    condition_2 = int32_t(matrix_fragment_23) != int32_t(0u);

    if (condition_1) {
        matrix_fragment_17 = matrix_fragment_16 * 0x4u;

        matrix_fragment_9 = matrix_fragment_16 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_36 = matrix_fragment_24 * 0x4u;

        matrix_fragment_10 = matrix_fragment_24 * 0x4u;
    }

    if (condition_1) {
        matrix_fragment_8 = !condition_3 ? matrix_fragment_17 : 0u;

        matrix_fragment_17 = !condition_3 ? matrix_fragment_9 : 0u;
    }

    if (condition_2) {
        matrix_fragment_36 = !condition_3 ? matrix_fragment_36 : 0u;

        matrix_fragment_46 = !condition_3 ? matrix_fragment_10 : 0u;
    }

    condition_3 = int32_t(matrix_fragment_29) != int32_t(0u);

    condition_5 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) && !condition_5;

    if (condition_4) {
        matrix_fragment_40 = matrix_fragment_24 * 0x4u;

        matrix_fragment_11 = matrix_fragment_24 * 0x4u;

        matrix_fragment_40 = !condition_5 ? matrix_fragment_40 : 0u;
    }

    if (condition_3) {
        matrix_fragment_9 = matrix_fragment_16 * 0x4u;

        matrix_fragment_10 = matrix_fragment_16 * 0x4u;
    }

    if (condition_4) {
        matrix_accumulator_57 = !condition_5 ? matrix_fragment_11 : 0u;
    }

    if (condition_3) {
        matrix_fragment_9 = !condition_5 ? matrix_fragment_9 : 0u;

        matrix_fragment_49 = !condition_5 ? matrix_fragment_10 : 0u;
    }

    condition_5 = int32_t(matrix_fragment_30) != int32_t(0u);

    matrix_fragment_18 = uint32_t(condition_0);

    condition_0 = (matrix_fragment_31 & 0x8u) != 0;

    if (condition_6) {
        matrix_fragment_42 = matrix_fragment_24 * 0x4u;

        matrix_fragment_24 = matrix_fragment_24 * 0x4u;
    }

    condition_0 = uint32_t(matrix_fragment_7) == uint32_t(0x1u) && !condition_0;

    if (condition_5) {
        matrix_fragment_35 = matrix_fragment_16 * 0x4u;

        matrix_fragment_16 = matrix_fragment_16 * 0x4u;
    }

    if (condition_6) {
        matrix_fragment_42 = !condition_0 ? matrix_fragment_42 : 0u;

        matrix_fragment_52 = !condition_0 ? matrix_fragment_24 : 0u;
    }

    if (condition_5) {
        matrix_fragment_35 = !condition_0 ? matrix_fragment_35 : 0u;

        matrix_fragment_48 = !condition_0 ? matrix_fragment_16 : 0u;
    }

    condition_0 = int32_t(matrix_fragment_18) != int32_t(0u);

    if (condition_1) {
        matrix_fragment_7 = index_word_144 & 0xfffffffcu;

        matrix_fragment_11 = index_word_144 & 0xfffffffcu;

        matrix_fragment_10 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_20;
    }

    if (condition_3) {
        matrix_fragment_21 = index_word_144 & 0xfffffffcu;
    }

    if (condition_1) {
        matrix_fragment_16 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_20;
    }

    if (condition_3) {
        matrix_fragment_25 = index_word_144 & 0xfffffffcu;
    }

    if (condition_1) {
        matrix_fragment_7 = 0u - matrix_fragment_7 + address_word_138;

        matrix_fragment_11 = 0u - matrix_fragment_11 + address_word_138;
    }

    if (condition_3) {
        matrix_fragment_18 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_26;
    }

    if (condition_0) {
        matrix_fragment_19 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_15;

        matrix_fragment_23 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_15;
    }

    if (condition_3) {
        matrix_fragment_24 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_26;
    }

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_20;
    }

    if (condition_4) {
        matrix_fragment_29 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_26;
    }

    if (condition_3) {
        matrix_fragment_21 = 0u - matrix_fragment_21 + address_word_138;
    }

    if (condition_5) {
        matrix_fragment_22 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_28;
    }

    if (condition_3) {
        matrix_fragment_25 = 0u - matrix_fragment_25 + address_word_138;
    }

    if (condition_5) {
        matrix_fragment_30 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_28;
    }

    if (condition_2) {
        matrix_fragment_20 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_20;
    }

    if (condition_6) {
        matrix_fragment_31 = matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_28;
    }

    if (condition_4) {
        matrix_fragment_26 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_26;
    }

    if (condition_6) {
        matrix_fragment_32 = matrix_fragment_5 * matrix_fragment_14 + matrix_fragment_28;
    }

    if (condition_1) {
        matrix_fragment_5 = matrix_fragment_13 * matrix_fragment_10 + matrix_fragment_7;
    }

    if (condition_0) {
        matrix_fragment_7 = index_word_144 & 0xfffffffcu;
    }

    if (condition_1) {
        matrix_fragment_16 = matrix_fragment_13 * matrix_fragment_16 + matrix_fragment_11;
    }

    if (condition_2) {
        matrix_fragment_11 = index_word_144 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_18 = matrix_fragment_13 * matrix_fragment_18 + matrix_fragment_21;
    }

    if (condition_0) {
        matrix_fragment_21 = index_word_144 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_24 = matrix_fragment_13 * matrix_fragment_24 + matrix_fragment_25;
    }

    if (condition_5) {
        matrix_fragment_25 = index_word_144 & 0xfffffffcu;
    }

    if (condition_0) {
        matrix_fragment_10 = 0u - matrix_fragment_7 + address_word_138;
    }

    if (condition_2) {
        matrix_fragment_12 = 0u - matrix_fragment_11 + address_word_138;
    }

    if (condition_5) {
        matrix_fragment_27 = index_word_144 & 0xfffffffcu;

        matrix_fragment_25 = 0u - matrix_fragment_25 + address_word_138;
    }

    if (condition_0) {
        matrix_fragment_14 = 0u - matrix_fragment_21 + address_word_138;

        matrix_fragment_39 = matrix_fragment_13 * matrix_fragment_19 + matrix_fragment_10;
    }

    if (condition_2) {
        matrix_fragment_7 = index_word_144 & 0xfffffffcu;

        matrix_fragment_43 = matrix_fragment_13 * matrix_fragment_15 + matrix_fragment_12;
    }

    if (condition_4) {
        matrix_fragment_11 = index_word_144 & 0xfffffffcu;

        matrix_fragment_19 = index_word_144 & 0xfffffffcu;
    }

    if (condition_6) {
        matrix_fragment_15 = index_word_144 & 0xfffffffcu;

        matrix_fragment_21 = index_word_144 & 0xfffffffcu;
    }

    if (condition_5) {
        matrix_fragment_27 = 0u - matrix_fragment_27 + address_word_138;

        matrix_fragment_38 = matrix_fragment_13 * matrix_fragment_22 + matrix_fragment_25;
    }

    if (condition_2) {
        matrix_fragment_10 = 0u - matrix_fragment_7 + address_word_138;
    }

    if (condition_0) {
        matrix_fragment_44 = matrix_fragment_13 * matrix_fragment_23 + matrix_fragment_14;
    }

    if (condition_4) {
        matrix_fragment_12 = 0u - matrix_fragment_11 + address_word_138;

        matrix_fragment_22 = 0u - matrix_fragment_19 + address_word_138;
    }

    if (condition_6) {
        matrix_fragment_14 = 0u - matrix_fragment_15 + address_word_138;

        matrix_fragment_28 = 0u - matrix_fragment_21 + address_word_138;
    }

    if (condition_5) {
        matrix_fragment_37 = matrix_fragment_13 * matrix_fragment_30 + matrix_fragment_27;
    }

    if (condition_1) {
        matrix_fragment_5 = matrix_fragment_8 + matrix_fragment_5;
    }

    if (condition_2) {
        matrix_fragment_53 = matrix_fragment_13 * matrix_fragment_20 + matrix_fragment_10;
    }

    if (condition_1) {
        matrix_fragment_7 = matrix_fragment_17 + matrix_fragment_16;
    }

    if (condition_4) {
        matrix_accumulator_55 = matrix_fragment_13 * matrix_fragment_29 + matrix_fragment_12;
    }

    if (condition_3) {
        matrix_fragment_45 = matrix_fragment_9 + matrix_fragment_18;
    }

    if (condition_4) {
        matrix_fragment_50 = matrix_fragment_13 * matrix_fragment_26 + matrix_fragment_22;
    }

    if (condition_0) {
        matrix_fragment_16 = parameters[parameter_input_low_0];
        matrix_fragment_17 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_accumulator_59 = matrix_fragment_13 * matrix_fragment_31 + matrix_fragment_14;
    }

    if (condition_5) {
        matrix_fragment_51 = matrix_fragment_48 + matrix_fragment_37;
    }

    if (condition_6) {
        matrix_accumulator_61 = matrix_fragment_13 * matrix_fragment_32 + matrix_fragment_28;
    }

    if (condition_3) {
        matrix_fragment_49 = matrix_fragment_49 + matrix_fragment_24;
    }

    if (condition_5) {
        matrix_fragment_47 = matrix_fragment_35 + matrix_fragment_38;
    }

    if (condition_0) {
        matrix_fragment_37 = matrix_fragment_39 + matrix_fragment_34;

        matrix_fragment_12 = parameters[parameter_input_low_0];
        matrix_fragment_13 = parameters[parameter_input_high_1];
    }

    if (condition_1) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];

        matrix_fragment_10 = parameters[parameter_input_low_0];
        matrix_fragment_11 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_14 = parameters[parameter_input_low_0];
        matrix_fragment_15 = parameters[parameter_input_high_1];

        matrix_fragment_18 = parameters[parameter_input_low_0];
        matrix_fragment_19 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_20 = parameters[parameter_input_low_0];
        matrix_fragment_21 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_22 = parameters[parameter_input_low_0];
        matrix_fragment_23 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_24 = parameters[parameter_input_low_0];
        matrix_fragment_25 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_26 = parameters[parameter_input_low_0];
        matrix_fragment_27 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_28 = parameters[parameter_input_low_0];
        matrix_fragment_29 = parameters[parameter_input_high_1];

        matrix_fragment_32 = parameters[parameter_input_low_0];
        matrix_fragment_33 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_fragment_30 = parameters[parameter_input_low_0];
        matrix_fragment_31 = parameters[parameter_input_high_1];

        matrix_fragment_34 = parameters[parameter_input_low_0];
        matrix_fragment_35 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_39 = matrix_fragment_43 + matrix_fragment_36;

        matrix_fragment_43 = matrix_fragment_53 + matrix_fragment_46;
    }

    if (condition_0) {
        matrix_fragment_41 = matrix_fragment_44 + matrix_fragment_41;
    }

    if (condition_4) {
        matrix_fragment_53 = matrix_accumulator_55 + matrix_fragment_40;
    }

    if (condition_6) {
        matrix_accumulator_55 = matrix_accumulator_59 + matrix_fragment_42;
    }

    if (condition_4) {
        matrix_accumulator_57 = matrix_fragment_50 + matrix_accumulator_57;
    }

    if (condition_6) {
        matrix_accumulator_59 = matrix_accumulator_61 + matrix_fragment_52;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));
    }

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    if (condition_0) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    if (condition_0) {
        matrix_fragment_36 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), memory_ranges, error_counts);
    }

    if (condition_1) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_fragment_5 = 0u;

    if (condition_1) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    matrix_fragment_7 = 0u;

    if (condition_1) {
        matrix_fragment_5 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges, error_counts);
    }

    if (condition_2) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    if (condition_1) {
        matrix_fragment_7 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), memory_ranges, error_counts);
    }

    if (condition_2) {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    if (condition_0) {
        matrix_fragment_38 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), memory_ranges, error_counts);
    }

    if (condition_3) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    if (condition_2) {
        matrix_fragment_37 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_fragment_47)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    if (condition_2) {
        matrix_fragment_39 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19), memory_ranges, error_counts);
    }

    if (condition_3) {
        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));

        matrix_fragment_40 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_26, matrix_fragment_27,
                            int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_26, matrix_fragment_27));
    }

    if (condition_3) {
        matrix_fragment_42 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25), memory_ranges, error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_28, matrix_fragment_29));
    }

    if (condition_5) {
        matrix_fragment_41 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23), memory_ranges, error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_32, matrix_fragment_33,
                            int64_t(int32_t(matrix_accumulator_57)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));
    }

    if (condition_5) {
        matrix_fragment_43 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_26, matrix_fragment_27), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_30, matrix_fragment_31,
                            int64_t(int32_t(matrix_accumulator_55)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_30, matrix_fragment_31));
    }

    if (condition_4) {
        matrix_fragment_44 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_34, matrix_fragment_35,
                            int64_t(int32_t(matrix_accumulator_59)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_34, matrix_fragment_35));
    }

    if (condition_4) {
        matrix_fragment_46 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_32, matrix_fragment_33), memory_ranges, error_counts);
    }

    if (condition_6) {
        matrix_fragment_45 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_30, matrix_fragment_31), memory_ranges, error_counts);

        matrix_fragment_47 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_34, matrix_fragment_35), memory_ranges, error_counts);
    }

    matrix_fragment_9 = min(int32_t(0u), int32_t(uniform_index_word_6));

    uniform_index_word_6 = 0u;

    uniform_index_word_5 = 0x400u;

    matrix_fragment_9 = 0u - matrix_fragment_9 + packed_half_0;

    condition_0 = uint32_t(matrix_fragment_9) != 0u;

    uniform_index_word_6 = (uniform_index_word_6 << 0x18u) + uniform_index_word_5;

    tile_word_143 = (address_word_140 << 0x9u) + uniform_index_word_6;

    tile_word_143 = address_word_138 * 0x10u + tile_word_143;

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_9 + 0xffffffffu;
    }

    uniform_parameter_word_8 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_9 = parameters[parameter_trained_weights_high_5];

    if (condition_0) {
        matrix_fragment_8 = uint32_t(int32_t(matrix_fragment_9) >> 0x1fu);

        matrix_fragment_9 =
            uint32_t(combine_words(matrix_fragment_8, 0u) << 0x3u >> 32) + matrix_fragment_9;
    }

    matrix_fragment_8 = uint32_t(int32_t(matrix_accumulator_3) >> 0x1fu);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);

    if (condition_0) {
        matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_9) >> 0x3u);
    }

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_8, 0u) << 0x2u >> 32) + matrix_accumulator_3;

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_4;

    address_word_145 = 0u;

    if (condition_0) {
        address_word_145 = matrix_fragment_9 * packed_half_2 + packed_half_2;
    }

    packed_half_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_fragment_68 = 0;
    matrix_fragment_69 = 0;

    matrix_fragment_72 = 0;
    matrix_fragment_73 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_fragment_70 = 0;
    matrix_fragment_71 = 0;

    matrix_accumulator_76 = 0;
    matrix_accumulator_77 = 0;

    matrix_fragment_80 = 0;
    matrix_fragment_81 = 0;

    matrix_fragment_92 = 0;
    matrix_fragment_93 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_fragment_82 = 0;
    matrix_fragment_83 = 0;

    matrix_fragment_86 = 0;
    matrix_fragment_87 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    matrix_fragment_84 = 0;
    matrix_fragment_85 = 0;

    matrix_fragment_64 = 0;
    matrix_fragment_65 = 0;

    matrix_fragment_66 = 0;
    matrix_fragment_67 = 0;

    matrix_fragment_74 = 0;
    matrix_fragment_75 = 0;

    index_word_142 = address_word_140 * 0x20u;

    split_address_words(matrix_fragment_94, matrix_fragment_95,
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9));

    index_word_146 = uint32_t(int32_t(matrix_fragment_8) >> 0x2u);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_4 = 0u;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_40,
                                         matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_44,
                                         matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                         shared_bytes, memory_ranges, error_counts);

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            uint64_t(uint32_t(address_word_140)) * uint64_t(uint32_t(0x4000u)) +
                                combine_words(matrix_fragment_94, matrix_fragment_95));

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_28, matrix_fragment_29),
                                            matrix_fragment_36, matrix_fragment_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x1000ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x1200ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x2000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x2200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x3000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x3200ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        matrix_fragment_96 = (address_word_138 << 0x4u) + uniform_index_word_6;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x800u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_96 + 0x1000u,
                                            matrix_fragment_52, matrix_fragment_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_96 + 0x1800u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x200u, matrix_accumulator_60, matrix_accumulator_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1a00u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1200u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_36, matrix_fragment_37, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_38, matrix_fragment_39, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_36, matrix_fragment_37, 0u,
                                 0u);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_44,
                                 matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_46, matrix_fragment_47, 0u,
                                 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x400u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_38,
                                 matrix_fragment_39, 0u, 0u);

        split_address_words(matrix_accumulator_58, matrix_accumulator_59,
                            uint64_t(uint32_t(address_word_140)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_fragment_94, matrix_fragment_95));

        split_address_words(matrix_accumulator_58, matrix_accumulator_59,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_58, matrix_accumulator_59));

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_62,
                                 matrix_fragment_63, matrix_fragment_34, matrix_fragment_35,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_62,
                                 matrix_fragment_63, matrix_fragment_20, matrix_fragment_21,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_62,
                                 matrix_fragment_63, matrix_fragment_22, matrix_fragment_23,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_36,
                                 matrix_fragment_37);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_58, matrix_accumulator_59) + 0x10000ull,
            matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_100,
                                 matrix_fragment_101);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_58, matrix_accumulator_59) + 0x10200ull,
            matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x600u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1400u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1c00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_110,
                                 matrix_accumulator_111);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0xe00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_62,
                                 matrix_fragment_63);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_100,
                                 matrix_fragment_101);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_52,
                                 matrix_fragment_53);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_103), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1600u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_60,
                                 matrix_accumulator_61);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_102), 0x44004400u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_fragment_24 = 0x2b28u;

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_96 + 0x1e00u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_100), 0x44004400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_101), 0x44004400u);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_110,
                                 matrix_accumulator_111);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_31 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0x44004400u);

        matrix_fragment_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_fragment_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_52,
                                 matrix_fragment_53);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_accumulator_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_48,
                                 matrix_fragment_49);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_108), 0x44004400u);

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_accumulator_119 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

        matrix_accumulator_108 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_50,
                                 matrix_fragment_51);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

        matrix_fragment_15 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0x44004400u);

        matrix_fragment_14 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_46,
                                 matrix_fragment_47);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_14),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0x44004400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14),
            select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_54,
                                 matrix_accumulator_55);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

        matrix_fragment_8 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_16 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_31),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_accumulator_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_10,
                                 matrix_fragment_11);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_16),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_31 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16),
            select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

        matrix_fragment_30 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_13 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_17 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_5 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_32 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_30),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        matrix_fragment_17 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

        matrix_fragment_9 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_7 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_5),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30),
            select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

        matrix_fragment_11 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_fragment_103 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_accumulator_109 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_17),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_6 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_accumulator_60));

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_7),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5),
            select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_118 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_accumulator_59));

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_11),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_6), 0x3b283b28u);

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_103),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_32),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7),
            select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11),
            select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_100),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

        matrix_fragment_11 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_54), 0x44004400u);

        matrix_fragment_31 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_55), 0x44004400u);

        matrix_fragment_33 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                          select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_34 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

        matrix_fragment_32 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

        matrix_fragment_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_34),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_32),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_24), 0x37283728u);

        matrix_fragment_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34),
            select_half_operand<2, false, false>(matrix_fragment_35), 0x3b283b28u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_27 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                select_half_operand<2, false, false>(matrix_fragment_35));

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_25 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                          select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                          select_half_operand<2, false, false>(matrix_fragment_24));

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_74,
                                 matrix_fragment_75);

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_66,
                                 matrix_fragment_67);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_80,
                                 matrix_fragment_81);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_72,
                                 matrix_fragment_73);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_64,
                                 matrix_fragment_65);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_82,
                                 matrix_fragment_83);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_76,
                                 matrix_accumulator_77);

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_70,
                                 matrix_fragment_71);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_68,
                                 matrix_fragment_69);

        multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_22, matrix_fragment_23, packed_half_2,
                                 matrix_accumulator_3);

        split_address_words(matrix_fragment_94, matrix_fragment_95,
                            combine_words(matrix_fragment_94, matrix_fragment_95) + 0x400u);

        uniform_index_word_4 = uniform_parameter_word_7;
    } while (!uniform_condition_0);
    address_word_150 = parameters[parameter_trained_weights_low_4];
    address_word_151 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_4 = index_word_142 + 0x8u;

    index_word_141 = index_word_144 & 0x7ffffffcu;

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_6 = index_word_142 + 0x18u;

    matrix_fragment_10 = index_word_144 & 0xfffffffcu;

    matrix_fragment_5 = uint32_t(uint32_t(matrix_fragment_4) >> 0x1u);

    matrix_fragment_4 = index_word_142 + 0x10u;

    index_word_141 = 0u - index_word_141 + address_word_138;

    matrix_fragment_11 = uint32_t(uint32_t(matrix_fragment_6) >> 0x1u);

    matrix_fragment_5 = matrix_fragment_5 + address_word_138 + (0u - matrix_fragment_10);

    matrix_fragment_7 = uint32_t(uint32_t(matrix_fragment_4) >> 0x1u);

    matrix_fragment_6 = index_word_141 * 0x2u + index_word_142;

    matrix_fragment_7 = matrix_fragment_7 + address_word_138 + (0u - matrix_fragment_10);

    matrix_fragment_11 = matrix_fragment_11 + address_word_138 + (0u - matrix_fragment_10);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_6) >> 0x1u);

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_43 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x18010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x18010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_42 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_92,
                                         matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_88,
                                         matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        uint64_t(uint32_t(address_word_140)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_20);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_21);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        combine_words(matrix_fragment_16, matrix_fragment_17) + 0x14200u);

    uniform_index_word_4 = 0u;

    uniform_parameter_word_7 = uniform_index_word_6;

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) - 0x200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                            matrix_fragment_48, matrix_fragment_49,
                                            matrix_fragment_50, matrix_fragment_51, memory_ranges,
                                            error_counts);

        matrix_fragment_6 = (address_word_138 << 0x4u) + uniform_parameter_word_7;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_8 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x1000u);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_6, matrix_fragment_52,
                                            matrix_fragment_53, matrix_accumulator_54,
                                            matrix_accumulator_55, shared_bytes, memory_ranges,
                                            error_counts);

        uniform_parameter_word_7 = uniform_parameter_word_7 + 0x200u;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_6 + 0x800u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_8;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x1000u, matrix_accumulator_60, matrix_accumulator_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x1800u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_40, matrix_fragment_41,
                                 matrix_fragment_24, matrix_fragment_25);

        multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_42, matrix_fragment_43,
                                 packed_half_2, matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_48, matrix_fragment_49,
                                 matrix_fragment_44, matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_18,
                                 matrix_fragment_19);
    } while (!uniform_condition_0);
    __syncthreads();

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(address_word_148, address_word_149,
                        uint64_t(uint32_t(address_word_140)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_150, address_word_151));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(address_word_148, address_word_149,
                        combine_words(address_word_148, address_word_149) + 0x18b20u);

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_62 = 0;
    matrix_fragment_63 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(packed_half_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_20,
                                         matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                         shared_bytes, memory_ranges, error_counts);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(packed_half_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    packed_half_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_130 = 0;
    packed_half_131 = 0;

    matrix_accumulator_132 = 0;
    packed_half_133 = 0;

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    uniform_index_word_4 = 0u;

    uniform_parameter_word_7 = uniform_index_word_6;

    __syncthreads();
    do {
        split_address_words(address_word_152, packed_half_153,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_148, address_word_149));

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_152, packed_half_153) - 0xa00ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_152, packed_half_153) - 0x800ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_152, packed_half_153) - 0x600ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_152, packed_half_153) - 0x400ull, matrix_accumulator_76,
            matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_152, packed_half_153) - 0x200ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(address_word_152, packed_half_153),
                                            matrix_fragment_84, matrix_fragment_85,
                                            matrix_fragment_86, matrix_fragment_87, memory_ranges,
                                            error_counts);

        packed_half_154 = (address_word_138 << 0x4u) + uniform_parameter_word_7;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_8 = uniform_index_word_4 + 0x20u;

        split_address_words(address_word_148, address_word_149,
                            combine_words(address_word_148, address_word_149) + 0x3000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_154, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_7 = uniform_parameter_word_7 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_154 + 0x800u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_8;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_154 + 0x1000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_154 + 0x1800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, packed_half_133, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_132,
                                 packed_half_133);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_64, matrix_fragment_65,
                                 matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_130, packed_half_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_130,
                                 packed_half_131);

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

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_134,
                                 matrix_accumulator_135);

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

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_70, matrix_fragment_71,
                                 matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_128,
                                 matrix_accumulator_129);

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
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_76, matrix_accumulator_77,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_78, matrix_accumulator_79,
                                 matrix_accumulator_54, matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_78, matrix_accumulator_79,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_80, matrix_fragment_81,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_82, matrix_fragment_83,
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

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_62,
                                 matrix_fragment_63);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_84, matrix_fragment_85,
                                 matrix_accumulator_124, matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_126, matrix_accumulator_127);
    } while (!uniform_condition_0);
    split_address_words(matrix_fragment_68, matrix_fragment_69,
                        uint64_t(uint32_t(address_word_140)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_99 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_68, matrix_fragment_69) + 0x2c120ull, memory_ranges,
        error_counts);

    address_word_149 = address_word_140 * 0x800u;

    split_address_words(address_word_148, address_word_149,
                        uint64_t(uint32_t(address_word_149)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    split_address_words(address_word_148, address_word_149,
                        int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_148, address_word_149));

    __syncthreads();

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24120ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24320ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    address_word_152 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                       select_half_operand<2, false, false>(matrix_fragment_72));

    address_word_152 = add_half_pairs(select_half_operand<2, false, false>(address_word_152),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_100), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24520ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                     select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_100 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                                         select_half_operand<2, false, false>(matrix_fragment_95));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_81), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_100), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                     select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_89 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_fragment_86));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_154), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_101 = result_word;
    }

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                     select_half_operand<2, false, false>(matrix_fragment_86));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                         select_half_operand<2, false, false>(matrix_fragment_82));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_100 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                                         select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = permute_signed(matrix_fragment_102, matrix_fragment_102, 0x5432u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_152), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_153), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_90 = result_word;
    }

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(matrix_fragment_101));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_102 = result_word;
    }

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                         select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24720ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    packed_half_156 = permute_signed(matrix_fragment_101, matrix_fragment_101, 0x5432u);

    matrix_fragment_103 = permute_signed(matrix_fragment_94, matrix_fragment_94, 0x5432u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24920ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    address_word_152 = add_half_pairs(select_half_operand<2, false, false>(address_word_152),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                                         select_half_operand<2, false, false>(packed_half_156));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_103 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                         select_half_operand<2, false, false>(matrix_fragment_103));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24b20ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101), 0x4100410u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x4100410u);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24d20ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_93 = permute_signed(matrix_fragment_92, matrix_fragment_92, 0x5432u);

    float_bits_159 = half_to_float_bits((matrix_fragment_101 & 65535u) * 65537u);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_102));

    packed_half_158 = half_to_float_bits((matrix_fragment_101 >> 16) * 65537u);

    matrix_fragment_101 = half_to_float_bits((matrix_fragment_94 & 65535u) * 65537u);

    packed_half_157 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                     select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_95 & 65535u) * 65537u);

    packed_half_156 = half_to_float_bits((matrix_fragment_95 >> 16) * 65537u);

    matrix_fragment_102 = half_to_float_bits((matrix_fragment_94 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x24f20ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    matrix_fragment_101 = rsqrt_bits(matrix_fragment_101);

    matrix_fragment_102 = rsqrt_bits(matrix_fragment_102);

    matrix_fragment_103 = rsqrt_bits(matrix_fragment_103);

    packed_half_156 = rsqrt_bits(packed_half_156);

    matrix_fragment_101 = pack_half(matrix_fragment_102, matrix_fragment_101);

    float_bits_159 = rsqrt_bits(float_bits_159);

    packed_half_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                            select_half_operand<2, false, false>(matrix_fragment_101));

    packed_half_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                            select_half_operand<2, false, false>(matrix_fragment_101));

    packed_half_158 = rsqrt_bits(packed_half_158);

    matrix_fragment_103 = pack_half(packed_half_156, matrix_fragment_103);

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_fragment_101 = permute_signed(matrix_fragment_100, matrix_fragment_100, 0x5432u);

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_157 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_157), 0x4100410u);

    matrix_fragment_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                                       select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_fragment_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(0u));

    packed_half_156 = half_to_float_bits((packed_half_157 & 65535u) * 65537u);

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_103),
                                        select_half_operand<2, false, false>(0u));

    packed_half_157 = half_to_float_bits((packed_half_157 >> 16) * 65537u);

    matrix_fragment_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_fragment_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_158 = pack_half(packed_half_158, float_bits_159);

    matrix_fragment_103 = permute_signed(matrix_fragment_98, matrix_fragment_98, 0x5432u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    packed_half_156 = rsqrt_bits(packed_half_156);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_158));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    matrix_accumulator_128 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_fragment_103));

    packed_half_2 = pack_half(0u, matrix_fragment_99);

    matrix_fragment_102 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    matrix_fragment_99 = permute_signed(matrix_fragment_96, matrix_fragment_96, 0x5432u);

    packed_half_157 = rsqrt_bits(packed_half_157);

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_96 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_fragment_102 = rsqrt_bits(matrix_fragment_102);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_103 = rsqrt_bits(matrix_fragment_103);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    packed_half_156 = pack_half(packed_half_157, packed_half_156);

    packed_half_131 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x4100410u);

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_156 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                          select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_132 = half_to_float_bits((packed_half_131 & 65535u) * 65537u);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_3 = half_to_float_bits((packed_half_131 >> 16) * 65537u);

    packed_half_131 = pack_half(matrix_fragment_103, matrix_fragment_102);

    multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_100, matrix_fragment_101, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_accumulator_130 = rsqrt_bits(matrix_accumulator_132);

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x4100410u);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 = permute_signed(packed_half_155, packed_half_155, 0x5432u);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_131));

    matrix_accumulator_3 = rsqrt_bits(matrix_accumulator_3);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(matrix_accumulator_132));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_153), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_131));

    matrix_fragment_5 = half_to_float_bits((matrix_accumulator_128 & 65535u) * 65537u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_131));

    matrix_fragment_26 = half_to_float_bits((matrix_accumulator_128 >> 16) * 65537u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_131));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                             matrix_fragment_67);

    packed_half_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_152), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_131 = result_word;
    }

    packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_154), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_161 = result_word;
    }

    matrix_fragment_7 = half_to_float_bits((packed_half_155 & 65535u) * 65537u);

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_28 = half_to_float_bits((packed_half_155 >> 16) * 65537u);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_accumulator_3 = pack_half(matrix_accumulator_3, matrix_accumulator_130);

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_7);

    matrix_fragment_28 = rsqrt_bits(matrix_fragment_28);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    packed_half_131 = add_half_pairs(select_half_operand<2, false, false>(address_word_152),
                                     select_half_operand<2, false, false>(packed_half_131));

    matrix_fragment_6 = permute_signed(packed_half_153, packed_half_153, 0x5432u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 = pack_half(matrix_fragment_26, matrix_fragment_5);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_29 = pack_half(matrix_fragment_28, matrix_fragment_7);

    matrix_fragment_28 = permute_signed(packed_half_131, packed_half_131, 0x5432u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_133 = permute_signed(packed_half_154, packed_half_154, 0x5432u);

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    packed_half_131 = add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                     select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(0u));

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_6 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    packed_half_131 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_131), 0x4100410u);

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x4100410u);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_6);

    matrix_fragment_5 = half_to_float_bits((packed_half_154 & 65535u) * 65537u);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    packed_half_154 = half_to_float_bits((packed_half_154 >> 16) * 65537u);

    matrix_fragment_6 = half_to_float_bits((packed_half_131 & 65535u) * 65537u);

    packed_half_131 = half_to_float_bits((packed_half_131 >> 16) * 65537u);

    packed_half_153 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_6 = rsqrt_bits(matrix_fragment_6);

    packed_half_131 = rsqrt_bits(packed_half_131);

    matrix_accumulator_3 = rsqrt_bits(matrix_accumulator_3);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    packed_half_154 = rsqrt_bits(packed_half_154);

    matrix_fragment_30 = rsqrt_bits(packed_half_153);

    matrix_fragment_6 = pack_half(packed_half_131, matrix_fragment_6);

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_3 = pack_half(matrix_accumulator_3, matrix_fragment_4);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_5 = pack_half(packed_half_154, matrix_fragment_5);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_half(matrix_fragment_30, matrix_fragment_7);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                      select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_48 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(matrix_fragment_18));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_accumulator_3 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_74,
                             matrix_fragment_75);

    packed_half_2 = permute_signed(packed_half_2, matrix_accumulator_3, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_100, matrix_fragment_101, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_102),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_accumulator_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x3c203c20u);

    matrix_fragment_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_103),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3c203c20u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x3c203c20u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_fragment_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x3e473e47u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3c203c20u);

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_128 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3e473e47u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x3e473e47u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_129 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3e473e47u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_18 = (matrix_fragment_102 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = (matrix_accumulator_128 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                          select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_22 = (matrix_fragment_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_23 = (matrix_accumulator_129 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_98 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_50 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_20 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_21 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_73 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_72 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_81 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_80 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_85 = (matrix_fragment_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_78 = (matrix_fragment_82 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = (matrix_fragment_5 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 = (matrix_fragment_6 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_82 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_52 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_69 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_74 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_75 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_53 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_68 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_71 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_54 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_55 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = address_word_138 * 0x4u;

    condition_3 = (address_word_138 & 0x1u) != 0;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    condition_4 = (address_word_138 & 0x2u) != 0;

    matrix_fragment_95 = matrix_fragment_7 & 0x1cu;

    matrix_fragment_7 = condition_3 ? matrix_accumulator_3 : matrix_fragment_4;

    matrix_fragment_16 = condition_3 ? matrix_fragment_5 : matrix_fragment_6;

    matrix_accumulator_3 = condition_3 ? matrix_fragment_4 : matrix_accumulator_3;

    matrix_fragment_6 = condition_3 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_95 =
        uint32_t(int64_t(int32_t(address_word_138)) * (1ll << 0x1du) >> 32) + matrix_fragment_95;

    matrix_fragment_4 = !condition_4 ? matrix_fragment_7 : matrix_fragment_16;

    matrix_fragment_16 = !condition_4 ? matrix_fragment_16 : matrix_fragment_7;

    matrix_fragment_5 = !condition_4 ? matrix_accumulator_3 : matrix_fragment_6;

    matrix_accumulator_3 = !condition_4 ? matrix_fragment_6 : matrix_accumulator_3;

    matrix_fragment_94 = 0x1u ^ matrix_fragment_95;

    matrix_fragment_93 = 0x2u ^ matrix_fragment_95;

    matrix_fragment_92 = 0x3u ^ matrix_fragment_95;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    condition_5 = (address_word_138 & 0x8u) != 0;

    condition_6 = (address_word_138 & 0x10u) != 0;

    matrix_fragment_6 = !condition_5 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_fragment_7 = !condition_5 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_fragment_17 = !condition_5 ? matrix_fragment_16 : matrix_accumulator_3;

    matrix_fragment_84 = !condition_5 ? matrix_accumulator_3 : matrix_fragment_16;

    matrix_accumulator_77 = !condition_6 ? matrix_fragment_6 : matrix_fragment_17;

    matrix_fragment_4 = !condition_6 ? matrix_fragment_7 : matrix_fragment_84;

    matrix_fragment_6 = !condition_6 ? matrix_fragment_17 : matrix_fragment_6;

    matrix_fragment_7 = !condition_6 ? matrix_fragment_84 : matrix_fragment_7;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_4),
                                       select_half_operand<1, false, false>(matrix_fragment_4));

    matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5410u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x4100410u);

    matrix_fragment_5 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    matrix_fragment_90 = address_word_147 + 0x8u;

    matrix_fragment_5 = reciprocal_bits(matrix_fragment_5);

    matrix_accumulator_3 = uint32_t(int32_t(matrix_fragment_90) >> 0x1fu);

    matrix_fragment_4 = reciprocal_bits(matrix_fragment_4);

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x5u >> 32) + matrix_fragment_90;

    matrix_fragment_7 = matrix_fragment_7 & 0xffffffe0u;

    matrix_fragment_7 = matrix_fragment_90 + (0u - matrix_fragment_7);

    matrix_fragment_84 = pack_half(matrix_fragment_4, matrix_fragment_5);

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    matrix_fragment_6 = uint32_t(int32_t(index_word_144) >> 0x1fu);

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_10);

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_fragment_6, 0u) << 0x5u >> 32) + address_word_147;

    matrix_fragment_6 = matrix_fragment_6 & 0xffffffe0u;

    matrix_fragment_91 = address_word_147 + (0u - matrix_fragment_6);

    matrix_fragment_9 = transpose_half_tile(matrix_fragment_9);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_91), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_77 = permute_signed(matrix_fragment_7, matrix_fragment_7, 0x5410u);

    matrix_fragment_33 = transpose_half_tile(matrix_fragment_33);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_8 = transpose_half_tile(matrix_fragment_8);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_32 = transpose_half_tile(matrix_fragment_32);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = transpose_half_tile(matrix_fragment_34);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = uint32_t(int64_t(int32_t(index_word_144)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_22 = uint32_t(int64_t(int32_t(index_word_144)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_23 = uint32_t(int32_t(matrix_fragment_10) >> 0x1fu);

    matrix_fragment_89 = uint32_t(int32_t(matrix_fragment_22) >> 0x1fu);

    matrix_fragment_23 =
        uint32_t(combine_words(matrix_fragment_23, 0u) << 0x5u >> 32) + matrix_fragment_10;

    matrix_fragment_89 =
        uint32_t(combine_words(matrix_fragment_89, 0u) << 0x5u >> 32) + matrix_fragment_22;

    matrix_accumulator_79 = permute_signed(matrix_accumulator_79, matrix_accumulator_79, 0x5410u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = matrix_fragment_23 & 0xffffffe0u;

    matrix_fragment_89 = matrix_fragment_89 & 0xffffffe0u;

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_33 = matrix_fragment_10 + (0u - matrix_fragment_23);

    matrix_fragment_89 = matrix_fragment_22 + (0u - matrix_fragment_89);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_37);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = transpose_half_tile(matrix_fragment_13);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_39 = transpose_half_tile(matrix_fragment_39);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_33), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_89), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_12);

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_36);

    matrix_fragment_10 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_38 = transpose_half_tile(matrix_fragment_38);

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_96 = permute_signed(matrix_fragment_33, matrix_fragment_33, 0x5410u);

    matrix_fragment_97 = permute_signed(matrix_fragment_89, matrix_fragment_89, 0x5410u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_59 = transpose_half_tile(matrix_accumulator_59);

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_accumulator_121 = transpose_half_tile(matrix_accumulator_121);

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_58 = transpose_half_tile(matrix_accumulator_58);

    matrix_fragment_85 = transpose_half_tile(matrix_accumulator_120);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_accumulator_125 = transpose_half_tile(matrix_accumulator_125);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_57 = transpose_half_tile(matrix_accumulator_57);

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_61 = transpose_half_tile(matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_63 = transpose_half_tile(matrix_fragment_63);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 = transpose_half_tile(matrix_accumulator_124);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_122 = transpose_half_tile(matrix_accumulator_122);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_84 = transpose_half_tile(matrix_accumulator_56);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_86 = transpose_half_tile(matrix_accumulator_60);

    matrix_fragment_88 = transpose_half_tile(matrix_fragment_62);

    matrix_fragment_89 = transpose_half_tile(matrix_accumulator_126);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_77 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_22,
                             matrix_fragment_23);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_54));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_97 = uint32_t(uint32_t(index_word_142) >> 0x1u);

    matrix_fragment_52 = index_word_144 & 0xfffffffcu;

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_103 = matrix_fragment_97 + 0x4u;

    matrix_fragment_99 = matrix_fragment_97 + 0x8u;

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                             matrix_fragment_13);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_20,
                             matrix_fragment_21);

    index_word_141 = index_word_141 * 0x2u + index_word_142;

    matrix_fragment_97 = matrix_fragment_97 + 0xcu;

    matrix_fragment_5 = matrix_fragment_103 + address_word_138 + (0u - matrix_fragment_52);

    matrix_fragment_7 = matrix_fragment_99 + address_word_138 + (0u - matrix_fragment_52);

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_accumulator_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                          select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_53 = uint32_t(int32_t(index_word_141) >> 0x1u);

    matrix_fragment_21 = matrix_fragment_97 + address_word_138 + (0u - matrix_fragment_52);

    matrix_fragment_102 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_52, matrix_fragment_53,
                        int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        uint64_t(uint32_t(matrix_fragment_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_121 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_52, matrix_fragment_53) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_96 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(matrix_fragment_7));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                      select_half_operand<2, false, false>(matrix_fragment_6));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_72,
                                         matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_143 + 0x800u, matrix_accumulator_76, matrix_accumulator_77,
        matrix_accumulator_78, matrix_accumulator_79, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_113 = address_word_140 * 0x100u;

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        uint64_t(uint32_t(matrix_accumulator_113)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_150, address_word_151));

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_112, matrix_accumulator_113));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c130ull,
        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c330ull,
        matrix_fragment_32, matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d130ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d330ull,
        matrix_fragment_52, matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e130ull,
        matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e330ull,
        matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f130ull,
        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f330ull,
        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
        memory_ranges, error_counts);

    matrix_fragment_98 = (address_word_138 << 0x4u) + uniform_index_word_6;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98, matrix_accumulator_76, matrix_accumulator_77,
        matrix_accumulator_78, matrix_accumulator_79, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_81 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98 + 0x800u, matrix_fragment_80, matrix_fragment_81,
        matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_7 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_98 + 0x200u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_12,
                             matrix_fragment_13);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98 + 0xa00u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_12,
                             matrix_fragment_13);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_14,
                             matrix_fragment_15);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98 + 0xc00u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_78,
                             matrix_accumulator_79);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_98 + 0x400u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98 + 0xe00u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_98 + 0x600u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_52 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_68 = blockIdx.x;

    matrix_fragment_52 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 = parameters[parameter_optional_control_10];
    matrix_fragment_53 = parameters[parameter_optional_control_11];

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_118,
                             matrix_accumulator_119);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_120,
                             matrix_accumulator_121);

    matrix_fragment_70 = blockIdx.y;

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_54 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_55 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_59 = uint32_t(int32_t(packed_half_0) >> 0x1fu);

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_58 = uint32_t(int32_t(parameter_word_139) >> 0x1fu);

    matrix_accumulator_55 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_accumulator_59 =
        uint32_t(combine_words(matrix_accumulator_59, 0u) << 0x2u >> 32) + packed_half_0;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_accumulator_58 =
        uint32_t(combine_words(matrix_accumulator_58, 0u) << 0x2u >> 32) + parameter_word_139;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_54), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_57 = result_word;
    }

    matrix_fragment_71 = uint32_t(int32_t(matrix_accumulator_59) >> 0x2u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_55), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_56 = result_word;
    }

    matrix_fragment_69 = uint32_t(int32_t(matrix_accumulator_58) >> 0x2u);

    matrix_fragment_52 = matrix_fragment_68 * 0x8u + matrix_fragment_52;

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_38,
                             matrix_fragment_39);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    packed_half_0 = matrix_fragment_70 * 0x8u + matrix_fragment_53;

    condition_2 = int32_t(index_word_146) >= int32_t(matrix_fragment_69);

    condition_0 = int32_t(matrix_fragment_71) <= int32_t(uniform_index_word_5);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    matrix_fragment_20 = parameters[parameter_output_low_2];
    matrix_fragment_21 = parameters[parameter_output_high_3];

    condition_2 = int32_t(packed_half_0) > int32_t(0u - 0x4u) && !condition_2;

    condition_0 = int32_t(matrix_fragment_52) > int32_t(0u - 0x4u) && !condition_0;

    condition_1 = condition_2 & condition_0;

    matrix_fragment_70 = address_word_140 * 0x80u;

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_fragment_53 = index_word_146 * matrix_fragment_71 + uniform_index_word_5;

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_53 = matrix_fragment_53 * 0x200u + matrix_fragment_70;

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_62,
                             matrix_fragment_63);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_20, matrix_fragment_21));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }
    if (condition_1) {
        matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                          select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(matrix_fragment_22));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                          select_half_operand<2, false, false>(matrix_fragment_23));

        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        matrix_fragment_7 = matrix_fragment_32;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    uniform_index_word_4 = uniform_index_word_5 + 0x1u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    condition_1 = int32_t(matrix_fragment_71) <= int32_t(uniform_index_word_4);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                      select_half_operand<2, false, false>(0u));

    condition_1 = int32_t(matrix_fragment_52) > int32_t(0u - 0x8u) && !condition_1;

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                      select_half_operand<2, false, false>(0u));

    condition_2 = condition_2 & condition_1;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(0u));
    if (condition_2) {
        matrix_accumulator_61 = permute_signed(matrix_accumulator_61, matrix_fragment_6, 0x5410u);

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_4, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_accumulator_60, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_12, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = matrix_fragment_13;

        matrix_fragment_7 = matrix_accumulator_61;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x800ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25120ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25320ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25520ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25720ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25920ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25b20ull, matrix_fragment_52,
        matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25d20ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_148, address_word_149) + 0x25f20ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x5u >> 32) + matrix_fragment_90;

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_79 = permute_signed(matrix_accumulator_76, matrix_accumulator_76, 0x5432u);

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_80 = permute_signed(matrix_accumulator_77, matrix_accumulator_77, 0x5432u);

    matrix_accumulator_3 = matrix_accumulator_3 & 0xffffffe0u;

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_3 = matrix_fragment_90 + (0u - matrix_accumulator_3);

    matrix_accumulator_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76), 0x4100410u);

    matrix_accumulator_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77), 0x4100410u);

    matrix_accumulator_78 = half_to_float_bits((matrix_accumulator_76 & 65535u) * 65537u);

    matrix_fragment_80 = half_to_float_bits((matrix_accumulator_77 & 65535u) * 65537u);

    matrix_accumulator_77 = half_to_float_bits((matrix_accumulator_77 >> 16) * 65537u);

    matrix_accumulator_76 = half_to_float_bits((matrix_accumulator_76 >> 16) * 65537u);

    matrix_accumulator_78 = rsqrt_bits(matrix_accumulator_78);

    matrix_accumulator_79 = rsqrt_bits(matrix_accumulator_76);

    matrix_fragment_80 = rsqrt_bits(matrix_fragment_80);

    matrix_accumulator_77 = rsqrt_bits(matrix_accumulator_77);

    matrix_accumulator_79 = pack_half(matrix_accumulator_79, matrix_accumulator_78);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_81 = pack_half(matrix_accumulator_77, matrix_fragment_80);

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_40 = permute_signed(matrix_fragment_40, matrix_accumulator_77, 0x5410u);

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
                             matrix_fragment_100, matrix_fragment_101, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_4 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_5 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x4100410u);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_73 & 65535u) * 65537u);

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_73 >> 16) * 65537u);

    matrix_fragment_5 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_72 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_72 = rsqrt_bits(matrix_fragment_72);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_4 = pack_half(matrix_fragment_73, matrix_fragment_4);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3c203c20u);

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3e473e47u);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_6 = pack_half(matrix_fragment_72, matrix_fragment_5);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_5, 0x5410u);

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_13),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_5 = permute_signed(matrix_accumulator_105, matrix_fragment_6, 0x5410u);

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_6 = permute_signed(matrix_accumulator_108, matrix_fragment_7, 0x5410u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_21),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3c203c20u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x3c203c20u);

    matrix_fragment_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_22),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3c203c20u);

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_23),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3c203c20u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x3e473e47u);

    matrix_fragment_23 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x3c203c20u);

    matrix_fragment_22 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3e473e47u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_7 = permute_signed(matrix_accumulator_109, matrix_fragment_40, 0x5410u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_20 = (matrix_fragment_20 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x3c203c20u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_100, matrix_fragment_101, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x3e473e47u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3c203c20u);

    matrix_fragment_21 = (matrix_fragment_21 << 0x5u) + 0x7ff88000u;

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3e473e47u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3c203c20u);

    matrix_fragment_32 = (matrix_fragment_32 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3e473e47u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_fragment_24 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_39 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_27 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_47 = (matrix_fragment_34 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_54 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_accumulator_55 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3c203c20u);

    matrix_accumulator_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3e473e47u);

    matrix_accumulator_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3e473e47u);

    matrix_fragment_28 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_12 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_13 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_fragment_31 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 = (matrix_fragment_15 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_14 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_15 = (matrix_accumulator_54 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x3e473e47u);

    matrix_fragment_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_23 = (matrix_fragment_22 << 0x5u) + 0x7ff88000u;

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_22 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_35),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_40 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_38 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_41 = (matrix_accumulator_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_36 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_37 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3c203c20u);

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3e473e47u);

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3e473e47u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    packed_half_2 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    matrix_fragment_35 = (matrix_fragment_35 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_34 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_43 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_45 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_5 = condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_4 = condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_6 = condition_3 ? matrix_fragment_49 : matrix_fragment_46;

    matrix_fragment_49 = condition_3 ? matrix_fragment_46 : matrix_fragment_49;

    matrix_fragment_46 = !condition_4 ? matrix_fragment_5 : matrix_fragment_6;

    matrix_fragment_7 = !condition_4 ? matrix_fragment_4 : matrix_fragment_49;

    matrix_fragment_6 = !condition_4 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_49 = !condition_4 ? matrix_fragment_49 : matrix_fragment_4;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_4 = !condition_5 ? matrix_fragment_46 : matrix_fragment_7;

    matrix_fragment_5 = !condition_5 ? matrix_fragment_7 : matrix_fragment_46;

    matrix_fragment_7 = uint32_t(int64_t(int32_t(index_word_144)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_46 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    matrix_fragment_53 = !condition_5 ? matrix_fragment_6 : matrix_fragment_49;

    matrix_fragment_48 = !condition_5 ? matrix_fragment_49 : matrix_fragment_6;

    matrix_fragment_46 =
        uint32_t(combine_words(matrix_fragment_46, 0u) << 0x5u >> 32) + matrix_fragment_7;

    matrix_fragment_52 = !condition_6 ? matrix_fragment_4 : matrix_fragment_53;

    matrix_accumulator_55 = !condition_6 ? matrix_fragment_5 : matrix_fragment_48;

    matrix_fragment_4 = !condition_6 ? matrix_fragment_53 : matrix_fragment_4;

    matrix_fragment_5 = !condition_6 ? matrix_fragment_48 : matrix_fragment_5;

    matrix_fragment_46 = matrix_fragment_46 & 0xffffffe0u;

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_53 = matrix_fragment_7 + (0u - matrix_fragment_46);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_52 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_52),
                                        select_half_operand<1, false, false>(matrix_fragment_52));

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_52, 0x5410u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x4100410u);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_52 & 65535u) * 65537u);

    matrix_fragment_52 = half_to_float_bits((matrix_fragment_52 >> 16) * 65537u);

    matrix_fragment_4 = reciprocal_bits(matrix_fragment_4);

    matrix_fragment_5 = reciprocal_bits(matrix_fragment_52);

    matrix_fragment_6 = pack_half(matrix_fragment_5, matrix_fragment_4);

    matrix_fragment_5 = uint32_t(int64_t(int32_t(index_word_144)) * (1ll << 0x1eu) >> 32) + 0x10u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_91), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_4 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x5u >> 32) + matrix_fragment_5;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_53), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_fragment_4 = matrix_fragment_4 & 0xffffffe0u;

    matrix_fragment_49 = matrix_fragment_5 + (0u - matrix_fragment_4);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_91 = permute_signed(matrix_fragment_91, matrix_fragment_91, 0x5410u);

    matrix_accumulator_3 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5410u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_48 = permute_signed(matrix_fragment_48, matrix_fragment_48, 0x5410u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_46 = permute_signed(matrix_fragment_46, matrix_fragment_46, 0x5410u);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_accumulator_3 = index_word_144 & 0x7ffffffcu;

    index_word_144 = index_word_144 & 0xfffffffcu;

    matrix_fragment_103 = matrix_fragment_103 + address_word_138 + (0u - index_word_144);

    matrix_fragment_99 = matrix_fragment_99 + address_word_138 + (0u - index_word_144);

    matrix_fragment_97 = matrix_fragment_97 + address_word_138 + (0u - index_word_144);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_5 = 0u - matrix_accumulator_3 + address_word_138;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(0u));

    index_word_142 = matrix_fragment_5 * 0x2u + index_word_142;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_28,
                                         matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_11 = uint32_t(int32_t(index_word_142) >> 0x1u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_18,
                             matrix_fragment_19);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_99)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    matrix_fragment_52 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_54 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_103)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    split_address_words(packed_half_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_fragment_97)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_53 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_55 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, matrix_accumulator_3) + 0x30130ull, memory_ranges,
        error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c130ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c330ull,
        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d130ull,
        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d330ull,
        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e130ull,
        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e330ull,
        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f130ull,
        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f330ull,
        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
        error_counts);

    uniform_index_word_6 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_index_word_4 = (uniform_index_word_6 << 0x18u) + uniform_index_word_4;

    matrix_fragment_62 = (address_word_138 << 0x4u) + uniform_index_word_4;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_62, matrix_fragment_48,
                                        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_44);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_45);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0x800u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_44 >> 16);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_45 >> 16);

    matrix_fragment_71 = index_word_146 * matrix_fragment_71 + matrix_fragment_71;

    matrix_fragment_71 = matrix_fragment_71 + uniform_index_word_5;

    matrix_fragment_71 = matrix_fragment_71 * 0x200u + matrix_fragment_70;

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_36, matrix_fragment_37, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_46);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_47);

    matrix_fragment_46 = unpack_fp8_bits(matrix_fragment_46 >> 16);

    matrix_fragment_47 = unpack_fp8_bits(matrix_fragment_47 >> 16);

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_20, matrix_fragment_21, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_32);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_46,
                             matrix_fragment_47);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0x200u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0xa00u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, packed_half_2,
                             matrix_accumulator_3);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0x400u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0x600u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_48,
                             matrix_fragment_49);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0xc00u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_50,
                             matrix_fragment_51);

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_28, matrix_fragment_29, packed_half_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_62 + 0xe00u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_50,
                             matrix_fragment_51);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(packed_half_2, matrix_accumulator_3, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_12, matrix_fragment_13, packed_half_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_9 = threadIdx.x;

    matrix_fragment_8 = threadIdx.z;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_48,
                             matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_4, matrix_fragment_5, packed_half_2,
                             matrix_accumulator_3);

    packed_half_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    matrix_fragment_4 = index_word_146 + 0x1u;

    condition_2 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_69);

    condition_2 = int32_t(packed_half_0) > int32_t(0u - 0x8u) && !condition_2;

    condition_0 = condition_2 & condition_0;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_58,
                             matrix_accumulator_59);

    condition_3 = (address_word_140 | matrix_fragment_8 | matrix_fragment_9) != 0;

    split_address_words(packed_half_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    condition_1 = condition_2 & condition_1;

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_48,
                             matrix_fragment_49);
    if (condition_0) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, matrix_accumulator_3));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));
    if (condition_1) {
        matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_4, 0x5410u);

        split_address_words(packed_half_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_138)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, matrix_accumulator_3));

        packed_half_0 = permute_signed(matrix_fragment_51, packed_half_0, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_12, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_50, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = packed_half_0;

        matrix_fragment_7 = matrix_fragment_13;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, matrix_accumulator_3) + 0x800ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_3) {
        return;
    }

    packed_half_2 = parameters[parameter_publication_optional_low_16];
    matrix_accumulator_3 = parameters[parameter_publication_optional_high_17];

    address_word_145 = matrix_fragment_68 + address_word_145;

    split_address_words(packed_half_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, matrix_accumulator_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(156) void phoenix_window_attention_channels_128_input_view(KernelArguments parameters) {
    run_window_attention_channels_128_input_view<false>(parameters.words, nullptr, nullptr);
}
