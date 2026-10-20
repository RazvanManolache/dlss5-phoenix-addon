// DLSS5 Phoenix optimized kernel: window_attention/channels_256/input_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (__CUDA_ARCH__ >= 900)
#define DLSS5_TILE_SHARED_BYTES 17408
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
__device__ __forceinline__ void run_window_attention_channels_256_input_view(
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
    uint32_t packed_half_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_fragment_32 = 0;
    uint32_t matrix_fragment_33 = 0;
    uint32_t matrix_fragment_34 = 0;
    uint32_t matrix_fragment_35 = 0;
    uint32_t matrix_fragment_36 = 0;
    uint32_t matrix_fragment_37 = 0;
    uint32_t matrix_fragment_38 = 0;
    uint32_t matrix_fragment_39 = 0;
    uint32_t matrix_accumulator_40 = 0;
    uint32_t matrix_accumulator_41 = 0;
    uint32_t matrix_accumulator_42 = 0;
    uint32_t matrix_accumulator_43 = 0;
    uint32_t matrix_fragment_44 = 0;
    uint32_t matrix_fragment_45 = 0;
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
    uint32_t matrix_fragment_68 = 0;
    uint32_t matrix_fragment_69 = 0;
    uint32_t matrix_fragment_70 = 0;
    uint32_t matrix_fragment_71 = 0;
    uint32_t packed_half_72 = 0;
    uint32_t matrix_accumulator_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t packed_half_77 = 0;
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
    uint32_t packed_half_102 = 0;
    uint32_t packed_half_103 = 0;
    uint32_t matrix_accumulator_104 = 0;
    uint32_t matrix_accumulator_105 = 0;
    uint32_t matrix_accumulator_106 = 0;
    uint32_t matrix_accumulator_107 = 0;
    uint32_t matrix_accumulator_108 = 0;
    uint32_t matrix_accumulator_109 = 0;
    uint32_t matrix_accumulator_110 = 0;
    uint32_t matrix_accumulator_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t address_word_113 = 0;
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
    uint32_t packed_half_141 = 0;
    uint32_t packed_half_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t address_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t packed_half_150 = 0;
    uint32_t index_word_151 = 0;
    uint32_t tile_word_152 = 0;
    uint32_t address_word_153 = 0;
    uint32_t index_word_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_block_index_11 = 0;
    uint32_t uniform_reused_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
    uint32_t uniform_parameter_word_14 = 0;
    uint32_t uniform_parameter_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_block_index_17 = 0;
    uint32_t uniform_parameter_word_18 = 0;
    uint32_t uniform_parameter_word_19 = 0;
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
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    address_word_147 = threadIdx.x & 31u;

    uniform_block_index_11 = blockIdx.y;

    uniform_index_word_8 = parameters[parameter_optional_control_10];
    uniform_index_word_9 = parameters[parameter_optional_control_11];

    matrix_fragment_27 = matrix_fragment_27 & 0xffffffdfu;

    address_word_146 = threadIdx.y;

    uniform_parameter_word_15 = parameters[parameter_tensor_height_8];

    matrix_fragment_4 = parameters[parameter_optional_control_20];
    matrix_fragment_5 = parameters[parameter_optional_control_21];

    uniform_parameter_word_14 = parameters[parameter_tensor_width_9];

    uniform_reused_word_12 = 0u;
    uniform_reused_word_13 = 0u;

    uniform_block_index_17 = blockIdx.x;

    uniform_index_word_6 = (uniform_block_index_11 << 0x3u) + uniform_index_word_9;

    packed_half_0 = address_word_147;

    condition_0 = int32_t(matrix_fragment_4) > int32_t(0u);

    index_word_154 = address_word_147;

    matrix_fragment_7 = address_word_146 * 0x2u;

    matrix_accumulator_3 =
        uint32_t(int64_t(int32_t(packed_half_0)) * (1ll << 0x1cu) >> 32) + uniform_index_word_6;

    packed_half_0 = condition_0 ? matrix_fragment_4 : uniform_parameter_word_15;

    address_word_153 = uint32_t(int32_t(index_word_154) >> 0x2u);

    uniform_index_word_16 = (uniform_block_index_17 << 0x3u) + uniform_index_word_8;

    condition_1 = int32_t(matrix_fragment_5) > int32_t(0u);

    matrix_fragment_17 = matrix_accumulator_3 + 0x2u;

    condition_0 = int32_t(matrix_accumulator_3) >= int32_t(packed_half_0);

    matrix_accumulator_2 =
        uint32_t(combine_words(address_word_153, 0u) << 0x2u >> 32) + address_word_153;

    matrix_fragment_5 = condition_1 ? matrix_fragment_5 : uniform_parameter_word_14;

    condition_2 = int32_t(matrix_fragment_17) >= int32_t(packed_half_0);

    matrix_accumulator_2 = matrix_accumulator_2 & 0xfffffffcu;

    condition_0 = int32_t(matrix_accumulator_3) < int32_t(0u) || condition_0;

    matrix_fragment_21 = matrix_accumulator_3 + 0x4u;

    matrix_fragment_12 = address_word_153 + uniform_index_word_16 + (0u - matrix_accumulator_2);

    matrix_fragment_10 = matrix_fragment_5 * 0x4u;

    condition_2 = int32_t(matrix_fragment_17) < int32_t(0u) || condition_2;

    condition_3 = uint32_t(packed_half_0) != uint32_t(0x1u) && condition_0;

    condition_0 = uint32_t(packed_half_0) == uint32_t(0x1u) || !condition_0;

    condition_5 = int32_t(matrix_fragment_12) >= int32_t(matrix_fragment_5);

    matrix_fragment_14 = matrix_fragment_12 + 0x4u;

    matrix_fragment_23 = matrix_accumulator_3 + 0x6u;

    condition_1 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || condition_3;

    matrix_fragment_11 = matrix_fragment_7 + 0x1u;

    condition_5 = int32_t(matrix_fragment_12) > -1 && !condition_5;

    condition_6 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_5);

    if (condition_3) {
        matrix_fragment_27 = 0x20u | matrix_fragment_27;
    }

    condition_3 = condition_0 & (condition_1 | condition_5);

    matrix_fragment_27 = matrix_fragment_27 & 0xfffffffeu;

    condition_6 = int32_t(matrix_fragment_14) > -1 && !condition_6;

    condition_0 = condition_0 & (condition_1 | condition_6);

    if (condition_3) {
        matrix_fragment_27 = 0x1u | matrix_fragment_27;
    }

    condition_3 = uint32_t(packed_half_0) != uint32_t(0x1u) && condition_2;

    condition_2 = uint32_t(packed_half_0) == uint32_t(0x1u) || !condition_2;

    matrix_fragment_27 = matrix_fragment_27 & 0xffffffbfu;

    matrix_fragment_22 = uint32_t(condition_0);

    if (condition_3) {
        matrix_fragment_27 = 0x40u | matrix_fragment_27;
    }

    condition_3 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || condition_3;

    condition_1 = condition_2 & (condition_3 | condition_5);

    condition_2 = condition_2 & (condition_3 | condition_6);

    condition_3 = int32_t(matrix_fragment_21) >= int32_t(packed_half_0);

    matrix_fragment_24 = uint32_t(condition_1) << 1;

    matrix_fragment_20 = uint32_t(condition_2) << 2;

    condition_3 = int32_t(matrix_fragment_21) < int32_t(0u) || condition_3;

    condition_1 = (matrix_fragment_27 & 0x20u) != 0;

    condition_2 = (matrix_fragment_27 & 0x1u) != 0;

    condition_4 = uint32_t(packed_half_0) != uint32_t(0x1u) && condition_3;

    matrix_fragment_27 = matrix_fragment_27 & 0xffffffefu;

    condition_0 = uint32_t(packed_half_0) == uint32_t(0x1u) || !condition_3;

    condition_1 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) && !condition_1;

    if (condition_2) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];

        matrix_fragment_4 = index_word_154 & 0xfffffffcu;

        matrix_fragment_18 = matrix_fragment_12 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_27 = 0x10u | matrix_fragment_27;
    }

    condition_4 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || condition_4;

    if (condition_2) {
        matrix_fragment_16 = index_word_154 & 0xfffffffcu;
    }

    matrix_fragment_27 = matrix_fragment_27 & 0xfffffff7u;

    condition_3 = condition_0 & (condition_4 | condition_5);

    condition_4 = condition_0 & (condition_4 | condition_6);

    condition_0 = int32_t(matrix_fragment_23) >= int32_t(packed_half_0);

    matrix_fragment_25 = uint32_t(condition_3) << 3;

    if (condition_2) {
        matrix_fragment_15 = 0u - matrix_fragment_4 + address_word_147;

        matrix_fragment_4 = matrix_fragment_12 * 0x4u;
    }

    condition_0 = int32_t(matrix_fragment_23) < int32_t(0u) || condition_0;

    if (condition_2) {
        matrix_fragment_16 = 0u - matrix_fragment_16 + address_word_147;

        matrix_fragment_18 = !condition_1 ? matrix_fragment_18 : 0u;

        matrix_fragment_4 = !condition_1 ? matrix_fragment_4 : 0u;
    }

    condition_3 = uint32_t(packed_half_0) != uint32_t(0x1u) && condition_0;

    condition_0 = uint32_t(packed_half_0) == uint32_t(0x1u) || !condition_0;

    if (condition_4) {
        packed_half_30 = parameters[parameter_input_low_0];
        matrix_fragment_31 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_27 = 0x8u | matrix_fragment_27;
    }

    condition_3 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || condition_3;

    condition_5 = condition_0 & (condition_3 | condition_5);

    condition_6 = condition_0 & (condition_3 | condition_6);

    condition_0 = uint32_t(packed_half_0) != uint32_t(0x1u);

    condition_3 = (matrix_fragment_27 & 0x40u) != 0;

    matrix_fragment_26 = uint32_t(condition_5) << 5;

    matrix_fragment_13 = condition_0 ? matrix_accumulator_3 : 0u;

    matrix_fragment_17 = condition_0 ? matrix_fragment_17 : 0u;

    if (condition_2) {
        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    matrix_fragment_21 = condition_0 ? matrix_fragment_21 : 0u;

    matrix_fragment_23 = condition_0 ? matrix_fragment_23 : 0u;

    if (condition_2) {
        matrix_fragment_6 = packed_half_0 * matrix_fragment_7 + matrix_fragment_13;
    }

    condition_0 = int32_t(matrix_fragment_22) != int32_t(0u);

    if (condition_2) {
        matrix_fragment_19 = packed_half_0 * matrix_fragment_11 + matrix_fragment_13;
    }

    condition_3 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) && !condition_3;

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_10 * matrix_fragment_6 + matrix_fragment_15;
    }

    matrix_fragment_6 = 0u;

    if (condition_2) {
        matrix_fragment_19 = matrix_fragment_10 * matrix_fragment_19 + matrix_fragment_16;
    }

    condition_5 = (matrix_fragment_27 & 0x10u) != 0;

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_4 + matrix_fragment_15;
    }

    matrix_fragment_4 = 0u;

    if (condition_2) {
        matrix_fragment_19 = matrix_fragment_18 + matrix_fragment_19;
    }

    if (condition_0) {
        matrix_fragment_16 = matrix_fragment_14 * 0x4u;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_6 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges, error_counts);
    }

    if (condition_0) {
        matrix_fragment_15 = matrix_fragment_14 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_4 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges, error_counts);
    }

    if (condition_0) {
        matrix_fragment_39 = !condition_1 ? matrix_fragment_16 : 0u;
    }

    condition_2 = int32_t(matrix_fragment_20) != int32_t(0u);

    if (condition_0) {
        matrix_fragment_32 = !condition_1 ? matrix_fragment_15 : 0u;
    }

    condition_1 = int32_t(matrix_fragment_24) != int32_t(0u);

    matrix_fragment_16 = uint32_t(condition_0);

    condition_0 = (matrix_fragment_27 & 0x8u) != 0;

    if (condition_2) {
        matrix_fragment_34 = matrix_fragment_14 * 0x4u;

        matrix_fragment_8 = matrix_fragment_14 * 0x4u;
    }

    if (condition_1) {
        matrix_fragment_15 = matrix_fragment_12 * 0x4u;

        matrix_accumulator_3 = matrix_fragment_12 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_34 = !condition_3 ? matrix_fragment_34 : 0u;

        matrix_fragment_44 = !condition_3 ? matrix_fragment_8 : 0u;
    }

    if (condition_1) {
        matrix_accumulator_2 = !condition_3 ? matrix_fragment_15 : 0u;

        matrix_fragment_27 = !condition_3 ? matrix_accumulator_3 : 0u;
    }

    condition_3 = int32_t(matrix_fragment_25) != int32_t(0u);

    condition_5 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) && !condition_5;

    if (condition_4) {
        matrix_fragment_38 = matrix_fragment_14 * 0x4u;

        matrix_fragment_9 = matrix_fragment_14 * 0x4u;

        matrix_fragment_38 = !condition_5 ? matrix_fragment_38 : 0u;
    }

    if (condition_3) {
        matrix_fragment_8 = matrix_fragment_12 * 0x4u;

        matrix_accumulator_3 = matrix_fragment_12 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_55 = !condition_5 ? matrix_fragment_9 : 0u;
    }

    if (condition_3) {
        matrix_fragment_8 = !condition_5 ? matrix_fragment_8 : 0u;

        matrix_fragment_24 = !condition_5 ? matrix_accumulator_3 : 0u;
    }

    condition_5 = int32_t(matrix_fragment_26) != int32_t(0u);

    if (condition_6) {
        matrix_accumulator_40 = matrix_fragment_14 * 0x4u;
    }

    condition_0 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) && !condition_0;

    if (condition_6) {
        matrix_fragment_14 = matrix_fragment_14 * 0x4u;

        matrix_accumulator_40 = !condition_0 ? matrix_accumulator_40 : 0u;
    }

    if (condition_5) {
        matrix_fragment_33 = matrix_fragment_12 * 0x4u;

        matrix_fragment_12 = matrix_fragment_12 * 0x4u;
    }

    if (condition_6) {
        matrix_fragment_50 = !condition_0 ? matrix_fragment_14 : 0u;
    }

    if (condition_5) {
        matrix_fragment_33 = !condition_0 ? matrix_fragment_33 : 0u;

        packed_half_46 = !condition_0 ? matrix_fragment_12 : 0u;
    }

    condition_0 = int32_t(matrix_fragment_16) != int32_t(0u);

    if (condition_1) {
        matrix_fragment_12 = index_word_154 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_18 = index_word_154 & 0xfffffffcu;
    }

    if (condition_1) {
        matrix_fragment_14 = index_word_154 & 0xfffffffcu;

        matrix_accumulator_3 = packed_half_0 * matrix_fragment_7 + matrix_fragment_17;
    }

    if (condition_3) {
        matrix_fragment_22 = index_word_154 & 0xfffffffcu;

        matrix_fragment_9 = packed_half_0 * matrix_fragment_7 + matrix_fragment_21;
    }

    if (condition_1) {
        matrix_fragment_12 = 0u - matrix_fragment_12 + address_word_147;
    }

    if (condition_3) {
        matrix_fragment_18 = 0u - matrix_fragment_18 + address_word_147;
    }

    if (condition_1) {
        matrix_fragment_14 = 0u - matrix_fragment_14 + address_word_147;

        matrix_fragment_5 = packed_half_0 * matrix_fragment_11 + matrix_fragment_17;
    }

    if (condition_3) {
        matrix_fragment_22 = 0u - matrix_fragment_22 + address_word_147;
    }

    if (condition_0) {
        matrix_fragment_16 = packed_half_0 * matrix_fragment_7 + matrix_fragment_13;

        matrix_fragment_20 = packed_half_0 * matrix_fragment_11 + matrix_fragment_13;
    }

    if (condition_1) {
        matrix_accumulator_3 = matrix_fragment_10 * matrix_accumulator_3 + matrix_fragment_12;
    }

    if (condition_0) {
        matrix_fragment_12 = index_word_154 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_9 = matrix_fragment_10 * matrix_fragment_9 + matrix_fragment_18;
    }

    if (condition_0) {
        matrix_fragment_18 = index_word_154 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_19 = packed_half_0 * matrix_fragment_11 + matrix_fragment_21;
    }

    if (condition_5) {
        matrix_fragment_15 = packed_half_0 * matrix_fragment_7 + matrix_fragment_23;

        matrix_fragment_25 = packed_half_0 * matrix_fragment_11 + matrix_fragment_23;
    }

    if (condition_2) {
        matrix_fragment_13 = packed_half_0 * matrix_fragment_7 + matrix_fragment_17;
    }

    if (condition_4) {
        matrix_fragment_28 = packed_half_0 * matrix_fragment_7 + matrix_fragment_21;
    }

    if (condition_6) {
        matrix_fragment_29 = packed_half_0 * matrix_fragment_7 + matrix_fragment_23;
    }

    if (condition_2) {
        matrix_fragment_17 = packed_half_0 * matrix_fragment_11 + matrix_fragment_17;
    }

    if (condition_4) {
        matrix_fragment_21 = packed_half_0 * matrix_fragment_11 + matrix_fragment_21;
    }

    if (condition_6) {
        matrix_fragment_23 = packed_half_0 * matrix_fragment_11 + matrix_fragment_23;
    }

    if (condition_1) {
        packed_half_0 = matrix_fragment_10 * matrix_fragment_5 + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_14 = index_word_154 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_fragment_19 = matrix_fragment_10 * matrix_fragment_19 + matrix_fragment_22;
    }

    if (condition_5) {
        matrix_fragment_22 = index_word_154 & 0xfffffffcu;
    }

    if (condition_0) {
        matrix_fragment_5 = 0u - matrix_fragment_12 + address_word_147;

        matrix_fragment_11 = 0u - matrix_fragment_18 + address_word_147;
    }

    if (condition_2) {
        matrix_fragment_7 = 0u - matrix_fragment_14 + address_word_147;
    }

    if (condition_4) {
        matrix_fragment_18 = index_word_154 & 0xfffffffcu;
    }

    if (condition_5) {
        matrix_fragment_22 = 0u - matrix_fragment_22 + address_word_147;
    }

    if (condition_0) {
        matrix_fragment_37 = matrix_fragment_10 * matrix_fragment_16 + matrix_fragment_5;
    }

    if (condition_5) {
        matrix_fragment_26 = index_word_154 & 0xfffffffcu;
    }

    if (condition_0) {
        matrix_accumulator_42 = matrix_fragment_10 * matrix_fragment_20 + matrix_fragment_11;
    }

    if (condition_2) {
        matrix_fragment_12 = index_word_154 & 0xfffffffcu;
    }

    if (condition_4) {
        matrix_fragment_14 = index_word_154 & 0xfffffffcu;
    }

    if (condition_6) {
        matrix_fragment_16 = index_word_154 & 0xfffffffcu;

        matrix_fragment_20 = index_word_154 & 0xfffffffcu;
    }

    if (condition_2) {
        matrix_accumulator_41 = matrix_fragment_10 * matrix_fragment_13 + matrix_fragment_7;
    }

    if (condition_4) {
        matrix_fragment_13 = 0u - matrix_fragment_18 + address_word_147;
    }

    if (condition_5) {
        matrix_fragment_36 = matrix_fragment_10 * matrix_fragment_15 + matrix_fragment_22;

        matrix_fragment_26 = 0u - matrix_fragment_26 + address_word_147;
    }

    if (condition_2) {
        matrix_fragment_5 = 0u - matrix_fragment_12 + address_word_147;
    }

    if (condition_4) {
        matrix_fragment_7 = 0u - matrix_fragment_14 + address_word_147;
    }

    if (condition_6) {
        matrix_fragment_11 = 0u - matrix_fragment_16 + address_word_147;

        matrix_fragment_15 = 0u - matrix_fragment_20 + address_word_147;
    }

    if (condition_4) {
        matrix_fragment_48 = matrix_fragment_10 * matrix_fragment_21 + matrix_fragment_13;
    }

    if (condition_3) {
        matrix_accumulator_43 = matrix_fragment_8 + matrix_fragment_9;
    }

    if (condition_2) {
        matrix_fragment_12 = parameters[parameter_input_low_0];
        matrix_fragment_13 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_35 = matrix_fragment_10 * matrix_fragment_25 + matrix_fragment_26;
    }

    if (condition_2) {
        matrix_fragment_51 = matrix_fragment_10 * matrix_fragment_17 + matrix_fragment_5;
    }

    if (condition_4) {
        matrix_fragment_53 = matrix_fragment_10 * matrix_fragment_28 + matrix_fragment_7;
    }

    if (condition_6) {
        matrix_fragment_57 = matrix_fragment_10 * matrix_fragment_29 + matrix_fragment_11;
    }

    if (condition_1) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_fragment_59 = matrix_fragment_10 * matrix_fragment_23 + matrix_fragment_15;
    }

    if (condition_1) {
        matrix_fragment_5 = matrix_accumulator_2 + matrix_accumulator_3;
    }

    if (condition_3) {
        matrix_fragment_47 = matrix_fragment_24 + matrix_fragment_19;
    }

    if (condition_0) {
        matrix_fragment_10 = parameters[parameter_input_low_0];
        matrix_fragment_11 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_49 = packed_half_46 + matrix_fragment_35;
    }

    if (condition_1) {
        matrix_fragment_7 = matrix_fragment_27 + packed_half_0;
    }

    if (condition_5) {
        matrix_fragment_45 = matrix_fragment_33 + matrix_fragment_36;
    }

    if (condition_0) {
        matrix_fragment_35 = matrix_fragment_37 + matrix_fragment_32;

        matrix_fragment_14 = parameters[parameter_input_low_0];
        matrix_fragment_15 = parameters[parameter_input_high_1];
    }

    if (condition_1) {
        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_16 = parameters[parameter_input_low_0];
        matrix_fragment_17 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_18 = parameters[parameter_input_low_0];
        matrix_fragment_19 = parameters[parameter_input_high_1];

        matrix_fragment_22 = parameters[parameter_input_low_0];
        matrix_fragment_23 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_20 = parameters[parameter_input_low_0];
        matrix_fragment_21 = parameters[parameter_input_high_1];

        matrix_fragment_24 = parameters[parameter_input_low_0];
        matrix_fragment_25 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_26 = parameters[parameter_input_low_0];
        matrix_fragment_27 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_fragment_28 = parameters[parameter_input_low_0];
        matrix_fragment_29 = parameters[parameter_input_high_1];

        matrix_fragment_32 = parameters[parameter_input_low_0];
        matrix_fragment_33 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_37 = matrix_accumulator_41 + matrix_fragment_34;
    }

    if (condition_1) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_2) {
        matrix_accumulator_41 = matrix_fragment_51 + matrix_fragment_44;
    }

    matrix_fragment_7 = 0u;

    if (condition_2) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));
    }

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    if (condition_0) {
        matrix_fragment_39 = matrix_accumulator_42 + matrix_fragment_39;
    }

    if (condition_4) {
        matrix_fragment_51 = matrix_fragment_53 + matrix_fragment_38;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (condition_6) {
        matrix_fragment_53 = matrix_fragment_57 + matrix_accumulator_40;
    }

    if (condition_4) {
        matrix_fragment_55 = matrix_fragment_48 + matrix_fragment_55;
    }

    if (condition_6) {
        matrix_fragment_57 = matrix_fragment_59 + matrix_fragment_50;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));

        matrix_fragment_36 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), memory_ranges, error_counts);
    }

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    if (condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_7 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges, error_counts);
    }

    matrix_fragment_5 = 0u;

    if (condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_accumulator_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    if (condition_0) {
        matrix_fragment_38 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
    }

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    if (condition_3) {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(matrix_accumulator_43)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    if (condition_1) {
        matrix_fragment_5 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges, error_counts);
    }

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    if (condition_3) {
        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_fragment_47)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    if (condition_2) {
        matrix_fragment_37 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (condition_2) {
        matrix_fragment_39 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));
    }

    if (condition_3) {
        matrix_accumulator_40 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19), memory_ranges, error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_26, matrix_fragment_27,
                            int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_26, matrix_fragment_27));
    }

    if (condition_3) {
        matrix_accumulator_42 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23), memory_ranges, error_counts);
    }

    if (condition_4) {
        split_address_words(packed_half_30, matrix_fragment_31,
                            int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_30, matrix_fragment_31));
    }

    if (condition_5) {
        matrix_accumulator_41 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_28, matrix_fragment_29));
    }

    if (condition_5) {
        matrix_accumulator_43 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_32, matrix_fragment_33,
                            int64_t(int32_t(matrix_fragment_57)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));
    }

    if (condition_4) {
        matrix_fragment_8 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_26, matrix_fragment_27), memory_ranges, error_counts);

        matrix_fragment_10 = load_aligned_word<CheckBounds>(
            combine_words(packed_half_30, matrix_fragment_31), memory_ranges, error_counts);
    }

    if (condition_6) {
        matrix_fragment_9 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29), memory_ranges, error_counts);

        matrix_fragment_11 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_32, matrix_fragment_33), memory_ranges, error_counts);
    }

    uniform_index_word_10 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_8));

    uniform_index_word_5 = 0u - uniform_index_word_5 + uniform_parameter_word_14;

    uniform_condition_0 = uint32_t(uniform_index_word_5) != 0u;

    uniform_index_word_10 = (uniform_index_word_10 << 0x18u) + uniform_index_word_4;

    tile_word_152 = (address_word_146 << 0x9u) + uniform_index_word_10;

    tile_word_152 = address_word_147 * 0x10u + tile_word_152;

    uniform_parameter_word_18 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_19 = parameters[parameter_trained_weights_high_5];

    if (uniform_condition_0) {
        uniform_index_word_8 = uniform_index_word_5 + 0xffffffffu;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);
    }

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_16) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_6;

    if (uniform_condition_0) {
        uniform_index_word_9 =
            uint32_t(combine_words(uniform_index_word_9, 0u) << 0x3u >> 32) + uniform_index_word_8;
    }

    uniform_index_word_8 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x2u >> 32) + uniform_index_word_16;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    if (uniform_condition_0) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_9) >> 0x3u);
    }

    matrix_fragment_92 = 0;
    matrix_fragment_93 = 0;

    packed_half_72 = 0;
    matrix_accumulator_73 = 0;

    matrix_fragment_98 = 0;
    matrix_fragment_99 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_accumulator_74 = 0;
    matrix_accumulator_75 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_76 = 0;
    packed_half_77 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_fragment_86 = 0;
    matrix_fragment_87 = 0;

    matrix_fragment_96 = 0;
    matrix_fragment_97 = 0;

    matrix_fragment_84 = 0;
    matrix_fragment_85 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_fragment_94 = 0;
    matrix_fragment_95 = 0;

    split_address_words(packed_half_102, packed_half_103,
                        combine_words(uniform_parameter_word_18, uniform_parameter_word_19));

    uniform_index_word_4 = 0u;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_8) >> 0x2u);

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uniform_index_word_9 * uniform_block_index_11 + uniform_block_index_11;
    }

    uniform_index_word_5 = 0u;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_152 + 0x2000u, matrix_accumulator_40, matrix_accumulator_41,
        matrix_accumulator_42, matrix_accumulator_43, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x3000u, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    __syncthreads();
    do {
        split_address_words(packed_half_112, address_word_113,
                            uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0x8000u)) +
                                combine_words(packed_half_102, packed_half_103));

        split_address_words(packed_half_112, address_word_113,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_112, address_word_113));

        load_four_words_cached<CheckBounds>(combine_words(packed_half_112, address_word_113),
                                            matrix_fragment_60, matrix_fragment_61,
                                            matrix_fragment_62, matrix_fragment_63, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x200ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x1000ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x1200ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x2200ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x2000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x3000ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x3200ull, matrix_accumulator_40,
            matrix_accumulator_41, matrix_accumulator_42, matrix_accumulator_43, memory_ranges,
            error_counts);

        packed_half_0 = (address_word_147 << 0x4u) + uniform_index_word_10;

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x4000ull, matrix_fragment_28,
            matrix_fragment_29, packed_half_30, matrix_fragment_31, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x4200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0, matrix_fragment_20, matrix_fragment_21, matrix_fragment_22,
            matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1000u, matrix_fragment_4,
                                            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2000u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3000u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_44,
                                            matrix_fragment_45, packed_half_46, matrix_fragment_47,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1200u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3200u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x5000ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x5200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2200u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_44,
                                 matrix_fragment_45, packed_half_46, matrix_fragment_47,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_44,
                                 matrix_fragment_45, packed_half_46, matrix_fragment_47,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x400u,
                                            matrix_accumulator_104, matrix_accumulator_105,
                                            matrix_accumulator_106, matrix_accumulator_107,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x6000ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1400u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x6200ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_44,
                                 matrix_fragment_45, packed_half_46, matrix_fragment_47,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_44,
                                 matrix_fragment_45, packed_half_46, matrix_fragment_47,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_8, matrix_fragment_9);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x7000ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3400u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_12, matrix_fragment_13, matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(
            matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_14, matrix_fragment_15, matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_26,
                                 matrix_fragment_27, packed_half_46, matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                                 matrix_fragment_45);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_112, address_word_113) + 0x7200ull, matrix_fragment_44,
            matrix_fragment_45, packed_half_46, matrix_fragment_47, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x600u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2400u, matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1600u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2600u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1800u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x800u, matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_40, matrix_accumulator_41,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_42, matrix_accumulator_43,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3600u, matrix_fragment_12, matrix_fragment_13,
            matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2800u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_40, matrix_accumulator_41,
                                 matrix_accumulator_126, matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_42, matrix_accumulator_43,
                                 matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 packed_half_30, matrix_fragment_31, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0xa00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 packed_half_30, matrix_fragment_31, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3800u, matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_accumulator_40, matrix_accumulator_41,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_accumulator_42, matrix_accumulator_43,
                                 matrix_accumulator_124, matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_40, matrix_accumulator_41,
                                 matrix_accumulator_128, matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_accumulator_42, matrix_accumulator_43,
                                 matrix_accumulator_130, matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1a00u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2a00u, matrix_fragment_12, matrix_fragment_13,
            matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3a00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1c00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 packed_half_30, matrix_fragment_31, matrix_accumulator_122,
                                 matrix_accumulator_123);

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(packed_half_102, packed_half_103));

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_116,
                                 matrix_accumulator_117);

        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x40000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x40200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_30, matrix_fragment_31, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xc00u, matrix_fragment_28,
                                            matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2c00u, matrix_fragment_12, matrix_fragment_13,
            matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0xe00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3c00u, matrix_fragment_20, matrix_fragment_21,
            matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x1e00u, matrix_fragment_28, matrix_fragment_29,
            packed_half_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x2e00u, matrix_fragment_12, matrix_fragment_13,
            matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_42,
                                 matrix_accumulator_43);

        matrix_fragment_48 = 0x2b28u;

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_22 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_40), 0x44004400u);

        matrix_fragment_20 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_42), 0x44004400u);

        matrix_fragment_21 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_43), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 packed_half_46, matrix_fragment_47, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_fragment_23 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_41), 0x44004400u);

        matrix_fragment_20 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_28,
                                 matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                 packed_half_46, matrix_fragment_47, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(packed_half_30, matrix_fragment_31, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_50,
                                 matrix_fragment_51);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

        matrix_fragment_22 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

        matrix_fragment_21 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_20),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_23 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_22),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20),
            select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22),
            select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0 + 0x3e00u, matrix_fragment_20, matrix_fragment_21,
            matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

        matrix_accumulator_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                                select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_108,
                                 matrix_accumulator_109);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

        matrix_fragment_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                select_half_operand<2, false, false>(packed_half_0));

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_120,
                                 matrix_accumulator_121);

        packed_half_0 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                           0x44004400u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 packed_half_46, matrix_fragment_47, matrix_accumulator_124,
                                 matrix_accumulator_125);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_15 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_13 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_accumulator_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

        packed_half_0 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                           0x44004400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_13),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_110,
                                 matrix_accumulator_111);

        packed_half_0 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_accumulator_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_15),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_9 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_0),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_14 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_15 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_0),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_9),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_14),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9),
            select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

        matrix_fragment_9 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x44004400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14),
            select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_62,
                                 matrix_fragment_63);

        matrix_fragment_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

        matrix_fragment_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                select_half_operand<2, false, false>(packed_half_0));

        packed_half_0 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                           0x44004400u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_8 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_11 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_11), 0x3b283b28u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_accumulator_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 packed_half_46, matrix_fragment_47, matrix_fragment_66,
                                 matrix_fragment_67);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x44004400u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_27 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                select_half_operand<2, false, false>(matrix_fragment_9));

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_9 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x44004400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_20 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

        matrix_fragment_51 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_9 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

        packed_half_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_11 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_fragment_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_21 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_9),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_20 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_11),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        matrix_fragment_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_21),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9),
            select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_12 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_32 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_13 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                           select_half_operand<2, false, false>(0u));

        packed_half_0 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11),
            select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_21),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                          select_half_operand<2, false, false>(matrix_fragment_22));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                          select_half_operand<2, false, false>(matrix_fragment_20));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        matrix_fragment_22 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0x44004400u);

        matrix_fragment_20 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_21 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_23 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0x44004400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

        matrix_fragment_20 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

        matrix_fragment_22 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

        matrix_fragment_35 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_21 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_20),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_23 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_22),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_35),
            select_half_operand<0, false, true>(matrix_fragment_48), 0x37283728u);

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20),
            select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

        matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22),
            select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

        matrix_fragment_25 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_20 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        matrix_fragment_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                          select_half_operand<2, false, false>(matrix_fragment_22));

        matrix_fragment_22 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(matrix_fragment_23));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                           select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_82,
                                 matrix_accumulator_83);

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_76, packed_half_77, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_76,
                                 packed_half_77);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(packed_half_72, matrix_accumulator_73, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_6, matrix_fragment_7, packed_half_72,
                                 matrix_accumulator_73);

        multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_74,
                                 matrix_accumulator_75);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_94,
                                 matrix_fragment_95);

        split_address_words(packed_half_102, packed_half_103,
                            combine_words(packed_half_102, packed_half_103) + 0x400u);

        uniform_index_word_5 = uniform_index_word_9;
    } while (!uniform_condition_0);
    packed_half_0 = index_word_154 & 0x7ffffffcu;

    address_word_158 = parameters[parameter_trained_weights_low_4];
    address_word_159 = parameters[parameter_trained_weights_high_5];

    index_word_151 = address_word_146 * 0x20u;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_0 = 0u - packed_half_0 + address_word_147;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x3000u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_0 = packed_half_0 * 0x2u + index_word_151;

    index_word_151 = uint32_t(uint32_t(index_word_151) >> 0x1u);

    address_word_161 = uint32_t(int32_t(packed_half_0) >> 0x1u);

    matrix_fragment_4 = index_word_151 + 0x4u;

    packed_half_0 = index_word_154 & 0xfffffffcu;

    matrix_fragment_7 = matrix_fragment_4 + address_word_147 + (0u - packed_half_0);

    matrix_fragment_4 = index_word_151 + 0x8u;

    split_address_words(packed_half_160, address_word_161,
                        int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_9 = matrix_fragment_4 + address_word_147 + (0u - packed_half_0);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_4 = index_word_151 + 0xcu;

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0x58010ull, memory_ranges, error_counts);

    matrix_fragment_13 = matrix_fragment_4 + address_word_147 + (0u - packed_half_0);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x58010ull, memory_ranges,
        error_counts);

    matrix_accumulator_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x58010ull, memory_ranges,
        error_counts);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    uniform_index_word_5 = 0u;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    uniform_index_word_9 = uniform_index_word_10;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x2000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_77),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    __syncthreads();

    packed_half_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                    select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_72),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                   select_half_operand<2, false, false>(0u));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                    select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_152 + 0x2000u, matrix_accumulator_100, matrix_accumulator_101,
        packed_half_102, packed_half_103, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_30 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x3000u, matrix_fragment_96,
                                         matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_21);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_20);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_accumulator_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x48200u);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_accumulator_40));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    packed_half_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) - 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                            matrix_fragment_48, matrix_fragment_49,
                                            matrix_fragment_50, matrix_fragment_51, memory_ranges,
                                            error_counts);

        matrix_fragment_6 = (address_word_147 << 0x4u) + uniform_index_word_9;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_block_index_11 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_9 = uniform_index_word_9 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x1000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_block_index_11;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x2000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x3000u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(packed_half_30, matrix_fragment_31, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_50, matrix_fragment_51, packed_half_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_18,
                                 matrix_fragment_19);
    } while (!uniform_condition_0);
    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_60, matrix_fragment_61,
                        uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    packed_half_112 = 0;
    address_word_113 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        combine_words(matrix_fragment_60, matrix_fragment_61) + 0x58c20u);

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_32,
                                         matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x2000u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    packed_half_46 = 0;
    matrix_fragment_47 = 0;

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x3000u, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    matrix_fragment_56 = 0;
    matrix_fragment_57 = 0;

    matrix_fragment_58 = 0;
    matrix_fragment_59 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    matrix_accumulator_140 = 0;
    packed_half_141 = 0;

    packed_half_142 = 0;
    matrix_accumulator_143 = 0;

    matrix_accumulator_144 = 0;
    matrix_accumulator_145 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_accumulator_132 = 0;
    matrix_accumulator_133 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    uniform_index_word_5 = 0u;

    uniform_index_word_9 = uniform_index_word_10;

    __syncthreads();
    do {
        split_address_words(packed_half_102, packed_half_103,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_102, packed_half_103) - 0xa00ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_102, packed_half_103) - 0x800ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_102, packed_half_103) - 0x600ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_102, packed_half_103) - 0x400ull, packed_half_72,
            matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_102, packed_half_103) - 0x200ull, matrix_accumulator_76,
            packed_half_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(packed_half_102, packed_half_103),
                                            matrix_accumulator_80, matrix_accumulator_81,
                                            matrix_accumulator_82, matrix_accumulator_83,
                                            memory_ranges, error_counts);

        matrix_accumulator_104 = (address_word_147 << 0x4u) + uniform_index_word_9;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_block_index_11 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            combine_words(matrix_accumulator_100, matrix_accumulator_101) +
                                0x6000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_104, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_9 = uniform_index_word_9 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_104 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_block_index_11;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_104 + 0x2000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_104 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, packed_half_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(packed_half_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, packed_half_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_144,
                                 matrix_accumulator_145);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(packed_half_142, matrix_accumulator_143, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, packed_half_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(packed_half_112, address_word_113, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, packed_half_112,
                                 address_word_113);

        multiply_accumulate_half(matrix_accumulator_140, packed_half_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 packed_half_72, matrix_accumulator_73, matrix_accumulator_140,
                                 packed_half_141);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 packed_half_72, matrix_accumulator_73, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 packed_half_72, matrix_accumulator_73, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 packed_half_72, matrix_accumulator_73, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_74, matrix_accumulator_75,
                                 matrix_accumulator_138, matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_74, matrix_accumulator_75,
                                 matrix_accumulator_128, matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_74, matrix_accumulator_75,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_76, packed_half_77, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_76, packed_half_77, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_76, packed_half_77, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_76, packed_half_77, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_78, matrix_accumulator_79,
                                 matrix_accumulator_42, matrix_accumulator_43);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_40, matrix_accumulator_41);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_28,
                                 matrix_fragment_29);
    } while (!uniform_condition_0);
    split_address_words(matrix_fragment_60, matrix_fragment_61,
                        uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_65 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_60, matrix_fragment_61) + 0x98220ull, memory_ranges,
        error_counts);

    address_word_163 = address_word_146 * 0x800u;

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_131));

    __syncthreads();

    split_address_words(address_word_162, address_word_163,
                        uint64_t(uint32_t(address_word_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_64 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                             select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    split_address_words(address_word_162, address_word_163,
                        int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88220ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_61 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88420ull, matrix_accumulator_104,
        matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107, memory_ranges,
        error_counts);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88820ull, matrix_accumulator_100,
        matrix_accumulator_101, packed_half_102, packed_half_103, memory_ranges, error_counts);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                             select_half_operand<2, false, false>(packed_half_30));

    packed_half_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    {
        const uint32_t bits = address_word_147;
        condition_0 = bool(bits & 1u);
        condition_1 = bool(bits & 2u);
        condition_3 = bool(bits & 8u);
        condition_4 = bool(bits & 16u);
    }

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_accumulator_109 = transpose_half_tile(matrix_accumulator_109);

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_61), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                     select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_59 = transpose_half_tile(matrix_fragment_59);

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                     select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                             select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_43 = transpose_half_tile(matrix_accumulator_43);

    packed_half_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                                      select_half_operand<2, false, false>(matrix_accumulator_144),
                                      select_half_operand<2, false, false>(packed_half_72));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                      select_half_operand<2, false, false>(packed_half_142),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_58);

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_60), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    matrix_accumulator_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                                      select_half_operand<2, false, false>(matrix_accumulator_145),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(packed_half_72));

    matrix_accumulator_42 = transpose_half_tile(matrix_accumulator_42);

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                     select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_57 = transpose_half_tile(matrix_fragment_57);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_62), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_77 = result_word;
    }

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                                      select_half_operand<2, false, false>(matrix_accumulator_143),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_55 = transpose_half_tile(matrix_fragment_55);

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_accumulator_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                       select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_fragment_54 = transpose_half_tile(matrix_fragment_54);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_160 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0xa8240ull, memory_ranges, error_counts);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    packed_half_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    packed_half_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(packed_half_72));

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(packed_half_72),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    packed_half_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                      select_half_operand<2, false, false>(packed_half_112),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(packed_half_77));

    packed_half_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                    select_half_operand<2, false, false>(packed_half_72));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    packed_half_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(address_word_113),
                                      select_half_operand<2, false, false>(address_word_113),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88620ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_75), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_80 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_150), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_83 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_74), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_83 = permute_signed(matrix_accumulator_80, matrix_accumulator_80, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_75 = result_word;
    }

    matrix_accumulator_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                       select_half_operand<2, false, false>(matrix_fragment_88));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(packed_half_77),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_81 = result_word;
    }

    matrix_accumulator_82 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_77 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_80 = result_word;
    }

    matrix_fragment_71 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_155 = result_word;
    }

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_74 = permute_signed(matrix_fragment_71, matrix_fragment_71, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_74 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    packed_half_72 = add_half_pairs(select_half_operand<2, false, false>(packed_half_72),
                                    select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_67 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_accumulator_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_75 = result_word;
    }

    matrix_fragment_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(packed_half_77));

    matrix_fragment_68 = permute_signed(matrix_accumulator_79, matrix_accumulator_79, 0x5432u);

    matrix_accumulator_78 = half_to_float_bits((matrix_accumulator_74 & 65535u) * 65537u);

    matrix_accumulator_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_81 = half_to_float_bits((matrix_accumulator_74 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_74 = result_word;
    }

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                     select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_70 = permute_signed(matrix_accumulator_73, matrix_accumulator_73, 0x5432u);

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x4100410u);

    matrix_accumulator_73 = half_to_float_bits((matrix_accumulator_79 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_fragment_92 = half_to_float_bits((matrix_accumulator_79 >> 16) * 65537u);

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_69 = half_to_float_bits((matrix_fragment_90 & 65535u) * 65537u);

    matrix_accumulator_73 = rsqrt_bits(matrix_accumulator_73);

    matrix_fragment_96 = half_to_float_bits((matrix_fragment_90 >> 16) * 65537u);

    matrix_fragment_66 = rsqrt_bits(matrix_fragment_92);

    matrix_fragment_67 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    matrix_fragment_69 = rsqrt_bits(matrix_fragment_69);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_150), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_77 = result_word;
    }

    matrix_accumulator_74 = rsqrt_bits(matrix_fragment_96);

    matrix_fragment_92 = permute_signed(matrix_accumulator_75, matrix_accumulator_75, 0x5432u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(packed_half_72),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_89 = pack_half(matrix_fragment_66, matrix_accumulator_73);

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_fragment_92));

    packed_half_72 = permute_signed(matrix_fragment_91, matrix_fragment_91, 0x5432u);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_66 = permute_signed(matrix_fragment_67, matrix_fragment_67, 0x5432u);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75), 0x4100410u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                        select_half_operand<2, false, false>(packed_half_72));

    matrix_accumulator_80 = rsqrt_bits(matrix_accumulator_78);

    packed_half_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                    select_half_operand<2, false, false>(matrix_fragment_66));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88c20ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    packed_half_157 = pack_half(matrix_accumulator_74, matrix_fragment_69);

    matrix_fragment_69 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5432u);

    matrix_fragment_90 = half_to_float_bits((matrix_accumulator_75 & 65535u) * 65537u);

    matrix_accumulator_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x4100410u);

    matrix_accumulator_75 = half_to_float_bits((matrix_accumulator_75 >> 16) * 65537u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_72), 0x4100410u);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(packed_half_157));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                             select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_91 = rsqrt_bits(matrix_accumulator_75);

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_71 = half_to_float_bits((matrix_accumulator_73 & 65535u) * 65537u);

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_140 = half_to_float_bits((matrix_accumulator_73 >> 16) * 65537u);

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(packed_half_77));

    matrix_accumulator_138 = half_to_float_bits((matrix_fragment_97 & 65535u) * 65537u);

    packed_half_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x4100410u);

    matrix_accumulator_139 = half_to_float_bits((matrix_fragment_97 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88a20ull, matrix_accumulator_76,
        packed_half_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88e20ull, packed_half_72,
        matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89020ull, matrix_fragment_96,
        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_accumulator_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82), 0x4100410u);

    matrix_fragment_90 = rsqrt_bits(matrix_fragment_90);

    matrix_accumulator_83 = half_to_float_bits((matrix_accumulator_82 & 65535u) * 65537u);

    matrix_accumulator_82 = half_to_float_bits((matrix_accumulator_82 >> 16) * 65537u);

    matrix_fragment_71 = rsqrt_bits(matrix_fragment_71);

    matrix_accumulator_140 = rsqrt_bits(matrix_accumulator_140);

    matrix_accumulator_138 = rsqrt_bits(matrix_accumulator_138);

    matrix_accumulator_139 = rsqrt_bits(matrix_accumulator_139);

    matrix_accumulator_83 = rsqrt_bits(matrix_accumulator_83);

    matrix_accumulator_82 = rsqrt_bits(matrix_accumulator_82);

    matrix_accumulator_81 = rsqrt_bits(matrix_accumulator_81);

    packed_half_141 = pack_half(matrix_fragment_91, matrix_fragment_90);

    matrix_fragment_90 = half_to_float_bits((packed_half_142 & 65535u) * 65537u);

    matrix_accumulator_140 = pack_half(matrix_accumulator_140, matrix_fragment_71);

    matrix_fragment_91 = half_to_float_bits((packed_half_142 >> 16) * 65537u);

    packed_half_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_141));

    packed_half_141 = pack_half(matrix_accumulator_139, matrix_accumulator_138);

    matrix_accumulator_82 = pack_half(matrix_accumulator_82, matrix_accumulator_83);

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 = pack_half(matrix_accumulator_81, matrix_accumulator_80);

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_114 = pack_half(0u, matrix_fragment_65);

    matrix_accumulator_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                      select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_89 = permute_signed(matrix_fragment_64, matrix_fragment_64, 0x5432u);

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    packed_half_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(packed_half_141));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_118 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_90 = rsqrt_bits(matrix_fragment_90);

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_91 = rsqrt_bits(matrix_fragment_91);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89220ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x4100410u);

    matrix_accumulator_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_accumulator_131 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5432u);

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_67 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_129 = half_to_float_bits((matrix_accumulator_124 & 65535u) * 65537u);

    matrix_accumulator_125 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                       select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_131 = half_to_float_bits((matrix_accumulator_124 >> 16) * 65537u);

    matrix_accumulator_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                      select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_129 = rsqrt_bits(matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_accumulator_116 = pack_half(matrix_fragment_91, matrix_fragment_90);

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89420ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_84 = rsqrt_bits(matrix_accumulator_131);

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_116 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_85 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x4100410u);

    matrix_accumulator_118 = permute_signed(packed_half_155, packed_half_155, 0x5432u);

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_136 = half_to_float_bits((matrix_fragment_85 & 65535u) * 65537u);

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                       select_half_operand<2, false, false>(matrix_accumulator_118));

    packed_half_142 = pack_half(matrix_fragment_84, matrix_accumulator_129);

    matrix_accumulator_137 = half_to_float_bits((matrix_fragment_85 >> 16) * 65537u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_104,
                             matrix_accumulator_105);

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x4100410u);

    matrix_accumulator_136 = rsqrt_bits(matrix_accumulator_136);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 = permute_signed(packed_half_150, packed_half_150, 0x5432u);

    matrix_accumulator_137 = rsqrt_bits(matrix_accumulator_137);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_105 = half_to_float_bits((matrix_accumulator_130 >> 16) * 65537u);

    matrix_accumulator_143 = half_to_float_bits((matrix_accumulator_130 & 65535u) * 65537u);

    matrix_accumulator_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_104 = rsqrt_bits(matrix_accumulator_143);

    matrix_accumulator_138 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_150), 0x4100410u);

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                      select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_105 = rsqrt_bits(matrix_accumulator_105);

    matrix_accumulator_140 = half_to_float_bits((matrix_accumulator_138 >> 16) * 65537u);

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                            select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_143 = half_to_float_bits((matrix_accumulator_138 & 65535u) * 65537u);

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                          select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_18 = pack_half(matrix_accumulator_137, matrix_accumulator_136);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_accumulator_140 = rsqrt_bits(matrix_accumulator_140);

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 = rsqrt_bits(matrix_accumulator_143);

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                          select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_20 = pack_half(matrix_accumulator_105, matrix_accumulator_104);

    matrix_accumulator_134 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(0u));

    packed_half_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    address_word_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_113),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_110 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_144 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_144),
                      select_half_operand<2, false, false>(0u));

    packed_half_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                    select_half_operand<2, false, false>(0u));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_115 = pack_half(matrix_accumulator_140, matrix_accumulator_115);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_121 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(matrix_accumulator_144));

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                         select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                       select_half_operand<2, false, false>(packed_half_112));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_86,
                             matrix_fragment_87);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89620ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_62,
                             matrix_fragment_63);

    multiply_accumulate_half(packed_half_102, packed_half_103, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_18, matrix_fragment_19, packed_half_102,
                             packed_half_103);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89820ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89a20ull, matrix_accumulator_104,
        matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107, memory_ranges,
        error_counts);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_148), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89c20ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_46 = result_word;
    }

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89e20ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                    select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_20 = permute_signed(matrix_fragment_21, matrix_fragment_21, 0x5432u);

    matrix_fragment_47 = permute_signed(packed_half_46, packed_half_46, 0x5432u);

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    packed_half_46 = add_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                    select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x4100410u);

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x4100410u);

    matrix_fragment_20 = half_to_float_bits((matrix_fragment_21 & 65535u) * 65537u);

    matrix_fragment_53 = half_to_float_bits((matrix_fragment_21 >> 16) * 65537u);

    packed_half_112 = half_to_float_bits((packed_half_46 & 65535u) * 65537u);

    address_word_113 = half_to_float_bits((packed_half_46 >> 16) * 65537u);

    matrix_fragment_20 = rsqrt_bits(matrix_fragment_20);

    matrix_fragment_53 = rsqrt_bits(matrix_fragment_53);

    packed_half_112 = rsqrt_bits(packed_half_112);

    address_word_113 = rsqrt_bits(address_word_113);

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                         select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_53 = pack_half(matrix_fragment_53, matrix_fragment_20);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_52 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_21 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    packed_half_112 = pack_half(address_word_113, packed_half_112);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                         select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_112));

    multiply_accumulate_half(matrix_accumulator_76, packed_half_77, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_76,
                             packed_half_77);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(packed_half_72, matrix_accumulator_73, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_132, matrix_accumulator_133, packed_half_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x8a020ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_30 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_30 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                    select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_21 = permute_signed(packed_half_30, packed_half_30, 0x5432u);

    matrix_fragment_20 = permute_signed(matrix_fragment_31, matrix_fragment_31, 0x5432u);

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                    select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    packed_half_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x4100410u);

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x4100410u);

    matrix_fragment_21 = half_to_float_bits((packed_half_30 & 65535u) * 65537u);

    packed_half_30 = half_to_float_bits((packed_half_30 >> 16) * 65537u);

    matrix_fragment_20 = half_to_float_bits((matrix_fragment_31 & 65535u) * 65537u);

    matrix_fragment_21 = rsqrt_bits(matrix_fragment_21);

    matrix_fragment_31 = half_to_float_bits((matrix_fragment_31 >> 16) * 65537u);

    packed_half_30 = rsqrt_bits(packed_half_30);

    matrix_fragment_20 = rsqrt_bits(matrix_fragment_20);

    matrix_fragment_31 = rsqrt_bits(matrix_fragment_31);

    matrix_fragment_21 = pack_half(packed_half_30, matrix_fragment_21);

    packed_half_30 = 0x29c0u;

    matrix_accumulator_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_36 = pack_half(matrix_fragment_31, matrix_fragment_20);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3c203c20u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_118 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3c203c20u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3c203c20u);

    matrix_accumulator_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3c203c20u);

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_136 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_138 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_124 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3e473e47u);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_118 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3e473e47u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_128 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3e473e47u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_130 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3e473e47u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_125 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3c203c20u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_119 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3c203c20u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3c203c20u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_131 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3c203c20u);

    matrix_accumulator_137 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_136 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3c203c20u);

    matrix_accumulator_139 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_138 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_accumulator_125 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3e473e47u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_119 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3e473e47u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_129 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3e473e47u);

    matrix_accumulator_131 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_82,
                             matrix_accumulator_83);

    address_word_113 = (matrix_accumulator_124 << 0x5u) + 0x7ff88000u;

    packed_half_112 = (matrix_accumulator_118 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 = (matrix_accumulator_128 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_136 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3e473e47u);

    matrix_accumulator_138 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_137 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3c203c20u);

    matrix_accumulator_139 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3c203c20u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<0, false, false>(matrix_accumulator_114));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(address_word_113),
                                        select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_90,
                             matrix_fragment_91);

    packed_half_46 = (matrix_accumulator_119 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 = (matrix_accumulator_129 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0x3c203c20u);

    matrix_fragment_16 = (matrix_accumulator_136 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_21 = (matrix_accumulator_138 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_137 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3e473e47u);

    matrix_accumulator_139 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3e473e47u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_70,
                             matrix_fragment_71);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0x3c203c20u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_84,
                             matrix_fragment_85);

    packed_half_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0x3e473e47u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_34 = (matrix_accumulator_130 << 0x5u) + 0x7ff88000u;

    matrix_fragment_35 = (matrix_accumulator_125 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_accumulator_131 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                        select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0x3e473e47u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_17 = (matrix_accumulator_137 << 0x5u) + 0x7ff88000u;

    matrix_fragment_32 = (matrix_accumulator_139 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_10 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_11 = (packed_half_102 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_20 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_31 = (packed_half_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_77),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    packed_half_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                     select_half_operand<2, false, false>(matrix_fragment_31));

    packed_half_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_77), 0x3c203c20u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    packed_half_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                     select_half_operand<2, false, false>(packed_half_102));

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    packed_half_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_72),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_77), 0x3e473e47u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_accumulator_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_100 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_78 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_95 = (packed_half_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_79 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_93 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_77 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_72), 0x3c203c20u);

    matrix_accumulator_73 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3c203c20u);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    packed_half_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_72), 0x3e473e47u);

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_73 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3e473e47u);

    matrix_fragment_47 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                                        select_half_operand<2, false, false>(packed_half_77));

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    packed_half_103 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                     select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 = (packed_half_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    packed_half_72 = (matrix_accumulator_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    matrix_accumulator_74 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(packed_half_72),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_73 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_accumulator_76 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_75 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_fragment_97));

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_104,
                             matrix_accumulator_105);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_fragment_98));

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_97 = condition_0 ? matrix_fragment_98 : matrix_accumulator_115;

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_115 = condition_0 ? matrix_accumulator_115 : matrix_fragment_98;

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_96 = condition_0 ? packed_half_102 : matrix_accumulator_101;

    matrix_accumulator_114 = condition_0 ? matrix_accumulator_101 : packed_half_102;

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_accumulator_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_90 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_accumulator_82 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_89 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_83 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_accumulator_81 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    packed_half_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                                     select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_69 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_110 = !condition_1 ? matrix_fragment_96 : matrix_fragment_97;

    matrix_accumulator_111 = !condition_1 ? matrix_fragment_97 : matrix_fragment_96;

    matrix_fragment_70 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_71 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_99 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                       select_half_operand<2, false, false>(packed_half_102));

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_104 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x3c203c20u);

    packed_half_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                     select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_106 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x3c203c20u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_84 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                       select_half_operand<2, false, false>(packed_half_102));

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_accumulator_104 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x3e473e47u);

    matrix_accumulator_106 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x3e473e47u);

    matrix_accumulator_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    packed_half_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_85 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_accumulator_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_accumulator_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_61 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_60 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_97 = (matrix_accumulator_104 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = (matrix_accumulator_106 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    packed_half_102 = (packed_half_102 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x3c203c20u);

    matrix_accumulator_107 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107), 0x3c203c20u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                       select_half_operand<2, false, false>(packed_half_102));

    matrix_fragment_62 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_105 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x3e473e47u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_107 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107), 0x3e473e47u);

    packed_half_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_63 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_99 = (matrix_accumulator_105 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = (matrix_accumulator_107 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    packed_half_103 = (packed_half_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    packed_half_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3c203c20u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_87 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3e473e47u);

    matrix_fragment_67 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    packed_half_30 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_85 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_65 = (matrix_fragment_49 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_64 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = address_word_147 * 0x4u;

    matrix_fragment_84 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = matrix_fragment_14 & 0x1cu;

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_15 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1du) >> 32) + matrix_fragment_18;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_14 = !condition_1 ? matrix_accumulator_114 : matrix_accumulator_115;

    matrix_accumulator_114 = !condition_1 ? matrix_accumulator_115 : matrix_accumulator_114;

    matrix_fragment_18 = condition_0 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_50 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_fragment_13 = 0x1u ^ matrix_fragment_15;

    matrix_fragment_12 = 0x2u ^ matrix_fragment_15;

    matrix_fragment_49 = 0x3u ^ matrix_fragment_15;

    matrix_accumulator_105 = condition_0 ? matrix_fragment_19 : matrix_fragment_48;

    matrix_fragment_51 = condition_0 ? matrix_fragment_48 : matrix_fragment_19;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_110), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_110 = result_word;
    }

    matrix_accumulator_104 = !condition_1 ? matrix_fragment_50 : matrix_accumulator_105;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    matrix_fragment_48 = !condition_1 ? matrix_fragment_18 : matrix_fragment_51;

    matrix_accumulator_107 = !condition_1 ? matrix_fragment_51 : matrix_fragment_18;

    matrix_fragment_50 = !condition_1 ? matrix_accumulator_105 : matrix_fragment_50;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_111), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_111 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_49), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_48), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_104), "r"(matrix_fragment_13), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_105 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_107), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    matrix_fragment_14 = !condition_3 ? matrix_accumulator_110 : matrix_fragment_19;

    matrix_fragment_51 = !condition_3 ? matrix_accumulator_111 : matrix_accumulator_114;

    matrix_fragment_19 = !condition_3 ? matrix_fragment_19 : matrix_accumulator_110;

    matrix_accumulator_114 = !condition_3 ? matrix_accumulator_114 : matrix_accumulator_111;

    matrix_fragment_18 = !condition_4 ? matrix_fragment_14 : matrix_fragment_51;

    matrix_fragment_14 = !condition_4 ? matrix_fragment_51 : matrix_fragment_14;

    matrix_fragment_51 = !condition_3 ? matrix_fragment_48 : matrix_accumulator_105;

    matrix_fragment_48 = !condition_3 ? matrix_accumulator_105 : matrix_fragment_48;

    matrix_fragment_49 = address_word_153 + 0x8u;

    matrix_accumulator_104 = !condition_3 ? matrix_accumulator_107 : matrix_fragment_50;

    matrix_accumulator_107 = !condition_3 ? matrix_fragment_50 : matrix_accumulator_107;

    matrix_fragment_13 = !condition_4 ? matrix_fragment_19 : matrix_accumulator_114;

    matrix_fragment_12 = !condition_4 ? matrix_fragment_51 : matrix_accumulator_104;

    matrix_fragment_15 = !condition_4 ? matrix_fragment_48 : matrix_accumulator_107;

    matrix_fragment_51 = !condition_4 ? matrix_accumulator_104 : matrix_fragment_51;

    matrix_accumulator_104 = uint32_t(int32_t(matrix_fragment_49) >> 0x1fu);

    matrix_fragment_50 = uint32_t(int32_t(index_word_154) >> 0x1fu);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_104 =
        uint32_t(combine_words(matrix_accumulator_104, 0u) << 0x6u >> 32) + matrix_fragment_49;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_50 =
        uint32_t(combine_words(matrix_fragment_50, 0u) << 0x6u >> 32) + address_word_153;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_19 = !condition_4 ? matrix_accumulator_114 : matrix_fragment_19;

    matrix_fragment_48 = !condition_4 ? matrix_accumulator_107 : matrix_fragment_48;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_104 = matrix_accumulator_104 & 0xffc0u;

    matrix_fragment_50 = matrix_fragment_50 & 0xffc0u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_104 = matrix_fragment_49 + (0u - matrix_accumulator_104);

    matrix_fragment_50 = address_word_153 + (0u - matrix_fragment_50);

    matrix_fragment_13 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_13),
                                        select_half_operand<1, false, false>(matrix_fragment_13));

    matrix_fragment_12 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_12),
                                        select_half_operand<1, false, false>(matrix_fragment_12));

    matrix_fragment_15 = permute_signed(matrix_accumulator_104, 0u, 0x8880u);

    matrix_fragment_14 = permute_signed(matrix_fragment_50, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x5410u);

    matrix_fragment_13 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_13) >> 0xau);

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_14) >> 0xau);

    matrix_fragment_19 = matrix_fragment_18 & 0x1fu;

    matrix_fragment_15 = matrix_fragment_14 & 0x1fu;

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_19 = matrix_accumulator_104 + matrix_fragment_19;

    matrix_fragment_15 = matrix_fragment_50 + matrix_fragment_15;

    matrix_fragment_13 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    matrix_fragment_51 = uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_18 = matrix_fragment_19 & 0xe0u;

    matrix_fragment_14 = matrix_fragment_15 & 0xe0u;

    matrix_fragment_48 = uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_13 = reciprocal_bits(matrix_fragment_13);

    matrix_accumulator_106 = uint32_t(int32_t(matrix_fragment_51) >> 0x1fu);

    matrix_fragment_18 = 0u - matrix_fragment_18;

    matrix_fragment_49 = uint32_t(int32_t(matrix_fragment_48) >> 0x1fu);

    matrix_fragment_14 = 0u - matrix_fragment_14;

    matrix_fragment_12 = reciprocal_bits(matrix_fragment_12);

    matrix_accumulator_106 =
        uint32_t(combine_words(matrix_accumulator_106, 0u) << 0x6u >> 32) + matrix_fragment_51;

    matrix_fragment_49 =
        uint32_t(combine_words(matrix_fragment_49, 0u) << 0x6u >> 32) + matrix_fragment_48;

    matrix_accumulator_111 = permute_signed(matrix_fragment_18, 0u, 0x7710u);

    matrix_accumulator_105 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_accumulator_106 = matrix_accumulator_106 & 0xffc0u;

    matrix_fragment_49 = matrix_fragment_49 & 0xffc0u;

    matrix_accumulator_104 = matrix_accumulator_104 + matrix_accumulator_111;

    matrix_fragment_50 = matrix_fragment_50 + matrix_accumulator_105;

    matrix_accumulator_105 = matrix_fragment_51 + (0u - matrix_accumulator_106);

    matrix_fragment_14 = matrix_fragment_48 + (0u - matrix_fragment_49);

    matrix_accumulator_104 = matrix_accumulator_104 & 0xffffu;

    matrix_fragment_50 = matrix_fragment_50 & 0xffffu;

    matrix_fragment_49 = permute_signed(matrix_accumulator_105, 0u, 0x8880u);

    matrix_accumulator_107 =
        uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x20u;

    matrix_accumulator_106 = pack_half(matrix_fragment_12, matrix_fragment_13);

    matrix_fragment_12 = permute_signed(matrix_fragment_14, 0u, 0x8880u);

    matrix_accumulator_104 = permute_signed(matrix_accumulator_104, 0u, 0x8880u);

    matrix_fragment_13 = permute_signed(matrix_fragment_50, 0u, 0x8880u);

    matrix_fragment_49 = permute_signed(matrix_fragment_49, 0u, 0x7710u);

    matrix_accumulator_110 = uint32_t(int32_t(matrix_accumulator_107) >> 0x1fu);

    matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    matrix_fragment_50 = matrix_fragment_15 & 0xffffu;

    matrix_fragment_15 = matrix_accumulator_104;

    matrix_fragment_49 = uint32_t(uint32_t(matrix_fragment_49) >> 0xau);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_fragment_13), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_accumulator_110 =
        uint32_t(combine_words(matrix_accumulator_110, 0u) << 0x6u >> 32) + matrix_accumulator_107;

    matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_12) >> 0xau);

    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_50) >> 0x5u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_50 = matrix_fragment_49 & 0x1fu;

    matrix_accumulator_110 = matrix_accumulator_110 & 0xffc0u;

    matrix_fragment_49 = matrix_accumulator_105 + matrix_fragment_50;

    matrix_fragment_48 = matrix_accumulator_107 + (0u - matrix_accumulator_110);

    matrix_accumulator_107 =
        uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x28u;

    matrix_fragment_51 = matrix_fragment_18 & 0x1fu;

    condition_0 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_fragment_50 = matrix_fragment_49 & 0xe0u;

    matrix_fragment_12 = uint32_t(int32_t(matrix_accumulator_107) >> 0x1fu);

    matrix_accumulator_110 = matrix_fragment_19 & 0xffffu;

    matrix_fragment_18 = transpose_half_tile(matrix_accumulator_108);

    matrix_fragment_51 = matrix_fragment_14 + matrix_fragment_51;

    matrix_accumulator_104 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x6u >> 32) + matrix_accumulator_107;

    matrix_fragment_12 = uint32_t(uint32_t(matrix_accumulator_110) >> 0x5u);

    matrix_accumulator_108 = 0u - matrix_fragment_50;

    matrix_fragment_19 = matrix_fragment_51 & 0xe0u;

    condition_1 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_accumulator_108 = permute_signed(matrix_accumulator_108, 0u, 0x7710u);

    matrix_fragment_12 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x7632u);

    matrix_accumulator_111 = 0u - matrix_fragment_19;

    matrix_fragment_50 = matrix_accumulator_104 & 0xffc0u;

    matrix_accumulator_104 = permute_signed(matrix_fragment_15, matrix_accumulator_104, 0x7632u);

    matrix_accumulator_108 = matrix_accumulator_105 + matrix_accumulator_108;

    matrix_accumulator_105 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_accumulator_111 = permute_signed(matrix_accumulator_111, 0u, 0x7710u);

    matrix_accumulator_104 = condition_1 ? matrix_accumulator_104 : matrix_fragment_15;

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_fragment_50 = matrix_accumulator_107 + (0u - matrix_fragment_50);

    matrix_accumulator_107 = matrix_fragment_14 + matrix_accumulator_111;

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_35 = permute_signed(matrix_fragment_48, 0u, 0x8880u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = permute_signed(matrix_fragment_35, 0u, 0x7710u);

    matrix_fragment_45 = transpose_half_tile(matrix_accumulator_41);

    matrix_accumulator_108 = matrix_accumulator_108 & 0xffffu;

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_44);

    matrix_accumulator_108 = permute_signed(matrix_accumulator_108, 0u, 0x8880u);

    matrix_fragment_36 = uint32_t(uint32_t(matrix_fragment_34) >> 0xau);

    matrix_fragment_44 = transpose_half_tile(matrix_fragment_56);

    matrix_fragment_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                      select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_33 = transpose_half_tile(matrix_accumulator_40);

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 = matrix_fragment_36 & 0x1fu;

    matrix_accumulator_43 = matrix_accumulator_108;

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_56 = matrix_fragment_48 + matrix_accumulator_109;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_accumulator_43), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_59 = result_word;
    }

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_39);

    matrix_accumulator_42 = matrix_fragment_51 & 0xffffu;

    packed_half_46 = matrix_fragment_56 & 0xe0u;

    matrix_fragment_51 = transpose_half_tile(matrix_fragment_38);

    matrix_accumulator_107 = matrix_accumulator_107 & 0xffffu;

    matrix_accumulator_42 = uint32_t(uint32_t(matrix_accumulator_42) >> 0x5u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_43 = 0u - packed_half_46;

    matrix_accumulator_107 = permute_signed(matrix_accumulator_107, 0u, 0x8880u);

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                   select_half_operand<2, false, false>(0u));

    condition_0 = (matrix_accumulator_42 & 0x1u) != 0;

    matrix_accumulator_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                          select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_57 = permute_signed(matrix_accumulator_43, 0u, 0x7710u);

    matrix_accumulator_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(packed_half_46));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_accumulator_107), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    matrix_fragment_49 = matrix_fragment_49 & 0xffffu;

    matrix_fragment_33 = permute_signed(matrix_fragment_50, 0u, 0x8880u);

    packed_half_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_fragment_39 = uint32_t(uint32_t(matrix_fragment_49) >> 0x5u);

    matrix_fragment_33 = permute_signed(matrix_fragment_33, 0u, 0x7710u);

    address_word_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_113),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    condition_1 = (matrix_fragment_39 & 0x1u) != 0;

    matrix_fragment_44 = permute_signed(matrix_fragment_59, matrix_fragment_44, 0x7632u);

    matrix_fragment_45 = uint32_t(uint32_t(matrix_fragment_33) >> 0xau);

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = matrix_fragment_45 & 0x1fu;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_33 = condition_1 ? matrix_fragment_44 : matrix_fragment_59;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_113),
                                       select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_59 = matrix_fragment_50 + matrix_fragment_49;

    matrix_fragment_57 = matrix_fragment_48 + matrix_fragment_57;

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    packed_half_46 = permute_signed(matrix_accumulator_107, packed_half_46, 0x7632u);

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    packed_half_46 = condition_0 ? packed_half_46 : matrix_accumulator_107;

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(0u));

    packed_half_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_77),
                                         select_half_operand<0, false, false>(packed_half_46));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_42, matrix_accumulator_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_14 = matrix_fragment_59 & 0xe0u;

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_51 = 0u - matrix_fragment_14;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_51 = permute_signed(matrix_fragment_51, 0u, 0x7710u);

    matrix_fragment_57 = matrix_fragment_57 & 0xffffu;

    matrix_fragment_55 = uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x30u;

    matrix_fragment_54 = matrix_fragment_50 + matrix_fragment_51;

    index_word_154 = uint32_t(int64_t(int32_t(index_word_154)) * (1ll << 0x1eu) >> 32) + 0x38u;

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(packed_half_46));

    packed_half_77 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_77),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<0, false, false>(packed_half_46));

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x8880u);

    matrix_fragment_58 = uint32_t(int32_t(matrix_fragment_55) >> 0x1fu);

    matrix_accumulator_79 = uint32_t(int32_t(index_word_154) >> 0x1fu);

    matrix_fragment_54 = matrix_fragment_54 & 0xffffu;

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(packed_half_46));

    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(packed_half_77));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_fragment_57), "r"(0x1fu),
              "r"(__activemask()));
        packed_half_77 = result_word;
    }

    matrix_fragment_58 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x6u >> 32) + matrix_fragment_55;

    matrix_accumulator_79 =
        uint32_t(combine_words(matrix_accumulator_79, 0u) << 0x6u >> 32) + index_word_154;

    matrix_fragment_54 = permute_signed(matrix_fragment_54, 0u, 0x8880u);

    matrix_fragment_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                      select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_58 = matrix_fragment_58 & 0xffc0u;

    matrix_fragment_93 = matrix_accumulator_79 & 0xffc0u;

    matrix_accumulator_79 = matrix_fragment_54;

    matrix_accumulator_78 = matrix_fragment_55 + (0u - matrix_fragment_58);

    index_word_154 = index_word_154 + (0u - matrix_fragment_93);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_accumulator_79), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    matrix_fragment_57 = permute_signed(matrix_accumulator_78, 0u, 0x8880u);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_56 = matrix_fragment_56 & 0xffffu;

    matrix_fragment_93 = permute_signed(index_word_154, 0u, 0x8880u);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = uint32_t(uint32_t(matrix_fragment_56) >> 0x5u);

    matrix_fragment_93 = permute_signed(matrix_fragment_93, 0u, 0x7710u);

    matrix_fragment_92 = uint32_t(uint32_t(matrix_fragment_57) >> 0xau);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    condition_0 = (matrix_fragment_56 & 0x1u) != 0;

    matrix_fragment_58 = permute_signed(packed_half_77, matrix_fragment_58, 0x7632u);

    matrix_fragment_94 = uint32_t(uint32_t(matrix_fragment_93) >> 0xau);

    matrix_fragment_59 = matrix_fragment_59 & 0xffffu;

    matrix_fragment_93 = matrix_fragment_92 & 0x1fu;

    packed_half_77 = condition_0 ? matrix_fragment_58 : packed_half_77;

    matrix_fragment_58 = uint32_t(uint32_t(matrix_fragment_59) >> 0x5u);

    matrix_fragment_93 = matrix_accumulator_78 + matrix_fragment_93;

    matrix_fragment_59 = matrix_fragment_94 & 0x1fu;

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(packed_half_77));

    condition_0 = (matrix_fragment_58 & 0x1u) != 0;

    matrix_fragment_58 = permute_signed(matrix_accumulator_79, matrix_fragment_58, 0x7632u);

    matrix_fragment_91 = matrix_fragment_93 & 0xe0u;

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_fragment_90 = condition_0 ? matrix_fragment_58 : matrix_accumulator_79;

    matrix_accumulator_79 = 0u - matrix_fragment_91;

    matrix_fragment_92 = index_word_154 + matrix_fragment_59;

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_accumulator_79 = permute_signed(matrix_accumulator_79, 0u, 0x7710u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                           select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_42, matrix_accumulator_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_accumulator_80 = matrix_accumulator_78 + matrix_accumulator_79;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_82 = index_word_151 + 0x4u;

    matrix_accumulator_80 = matrix_accumulator_80 & 0xffffu;

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_fragment_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                      select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_83 = matrix_accumulator_82 + address_word_147 + (0u - packed_half_0);

    matrix_fragment_95 = permute_signed(matrix_accumulator_80, 0u, 0x8880u);

    matrix_accumulator_80 = index_word_151 + 0x8u;

    index_word_151 = index_word_151 + 0xcu;

    split_address_words(matrix_accumulator_82, matrix_accumulator_83,
                        uint64_t(uint32_t(matrix_accumulator_83)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_89 = matrix_accumulator_80 + address_word_147 + (0u - packed_half_0);

    matrix_accumulator_109 = index_word_151 + address_word_147 + (0u - packed_half_0);

    matrix_fragment_91 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_82, matrix_accumulator_83) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_88));

    split_address_words(matrix_fragment_88, matrix_fragment_89,
                        uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_accumulator_108, matrix_accumulator_109,
                        uint64_t(uint32_t(matrix_accumulator_109)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_94 = matrix_fragment_92 & 0xe0u;

    packed_half_0 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_88, matrix_fragment_89) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_accumulator_100 = matrix_fragment_92 & 0xffffu;

    matrix_fragment_92 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_108, matrix_accumulator_109) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_fragment_94 = 0u - matrix_fragment_94;

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_81 = permute_signed(matrix_fragment_94, 0u, 0x7710u);

    index_word_154 = index_word_154 + matrix_accumulator_81;

    index_word_154 = index_word_154 & 0xffffu;

    matrix_accumulator_107 = permute_signed(index_word_154, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_fragment_95), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_accumulator_107), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    matrix_fragment_93 = matrix_fragment_93 & 0xffffu;

    matrix_accumulator_100 = uint32_t(uint32_t(matrix_accumulator_100) >> 0x5u);

    matrix_fragment_93 = uint32_t(uint32_t(matrix_fragment_93) >> 0x5u);

    condition_1 = (matrix_accumulator_100 & 0x1u) != 0;

    condition_0 = (matrix_fragment_93 & 0x1u) != 0;

    matrix_fragment_94 = permute_signed(matrix_fragment_95, matrix_fragment_94, 0x7632u);

    matrix_fragment_88 = permute_signed(matrix_accumulator_107, matrix_fragment_88, 0x7632u);

    matrix_fragment_94 = condition_0 ? matrix_fragment_94 : matrix_fragment_95;

    matrix_fragment_93 = condition_1 ? matrix_fragment_88 : matrix_accumulator_107;

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    packed_half_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_102),
                                          select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                                          select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_fragment_25 = transpose_half_tile(matrix_fragment_25);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    matrix_accumulator_3 = transpose_half_tile(matrix_accumulator_3);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_42, matrix_accumulator_43, 0u, 0u);

    matrix_fragment_27 = transpose_half_tile(matrix_fragment_27);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(matrix_accumulator_104));

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_105));

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_24);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_2 = transpose_half_tile(matrix_accumulator_2);

    matrix_fragment_23 = transpose_half_tile(matrix_fragment_23);

    matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_9 = transpose_half_tile(matrix_fragment_9);

    matrix_fragment_29 = transpose_half_tile(matrix_fragment_29);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_42, matrix_accumulator_43, 0u, 0u);

    matrix_fragment_22 = transpose_half_tile(matrix_fragment_22);

    matrix_fragment_10 = transpose_half_tile(matrix_fragment_4);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = transpose_half_tile(matrix_fragment_26);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_6);

    matrix_fragment_20 = transpose_half_tile(matrix_fragment_8);

    matrix_fragment_21 = transpose_half_tile(matrix_fragment_28);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(packed_half_46));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<0, false, false>(packed_half_46));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<0, false, false>(packed_half_46));

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<0, false, false>(packed_half_46));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_72),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_40,
                             matrix_accumulator_41);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(packed_half_77));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(packed_half_77));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(matrix_fragment_90));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    packed_half_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                         select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(matrix_fragment_94));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_36,
                             matrix_fragment_37);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x3000u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_42,
                             matrix_accumulator_43);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x2000u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                   select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_40 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_41 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                      select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_42 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                      select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_43 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                      select_half_operand<2, false, false>(matrix_fragment_22));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x1000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_152 + 0x2000u, matrix_fragment_28,
                                         matrix_fragment_29, packed_half_30, matrix_fragment_31,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_152 + 0x3000u, matrix_accumulator_40, matrix_accumulator_41,
        matrix_accumulator_42, matrix_accumulator_43, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_90 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    packed_half_30 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_160));

    packed_half_46 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_80 = unpack_fp8_bits(matrix_fragment_12);

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        uint64_t(uint32_t(address_word_146)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_accumulator_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        combine_words(matrix_fragment_18, matrix_fragment_19) + 0x98440u);

    matrix_accumulator_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                             select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_0));

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    uniform_index_word_5 = 0u;

    __syncthreads();
    do {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) - 0x200ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_20, matrix_fragment_21),
                                            matrix_fragment_48, matrix_fragment_49,
                                            matrix_fragment_50, matrix_fragment_51, memory_ranges,
                                            error_counts);

        packed_half_0 = (address_word_147 << 0x4u) + uniform_index_word_10;

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0xe0u);

        uniform_index_word_9 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            combine_words(matrix_fragment_18, matrix_fragment_19) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0, matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
            matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_10 = uniform_index_word_10 + 0x200u;

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

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_50, matrix_fragment_51, packed_half_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_14,
                                 matrix_fragment_15);
    } while (!uniform_condition_0);
    uniform_index_word_5 = uint32_t(int32_t(uniform_parameter_word_15) >> 0x1fu);

    uniform_index_word_9 = uint32_t(int32_t(uniform_parameter_word_14) >> 0x1fu);

    matrix_fragment_24 = parameters[parameter_output_low_2];
    matrix_fragment_25 = parameters[parameter_output_high_3];

    matrix_fragment_31 = threadIdx.x;

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_parameter_word_15;

    uniform_index_word_9 =
        uint32_t(combine_words(uniform_index_word_9, 0u) << 0x2u >> 32) + uniform_parameter_word_14;

    packed_half_0 = address_word_146 * 0x80u;

    packed_half_30 = threadIdx.z;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_9) >> 0x2u);

    uniform_condition_2 = int32_t(uniform_index_word_7) >= int32_t(uniform_index_word_5);

    uniform_condition_0 = int32_t(uniform_index_word_8) >= int32_t(uniform_index_word_9);

    uniform_index_word_10 = uniform_index_word_7 * uniform_index_word_9 + uniform_index_word_8;

    uniform_condition_2 =
        int32_t(uniform_index_word_6) > int32_t(0u - 0x4u) && !uniform_condition_2;

    uniform_condition_0 =
        int32_t(uniform_index_word_16) > int32_t(0u - 0x4u) && !uniform_condition_0;

    matrix_fragment_19 = uniform_index_word_10;

    uniform_condition_1 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_19 = matrix_fragment_19 * 0x400u + packed_half_0;

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));
    if (uniform_condition_1) {
        matrix_fragment_20 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_35 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                          select_half_operand<2, false, false>(matrix_fragment_20));

        matrix_fragment_20 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_35));

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));

        matrix_fragment_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_22 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }
    uniform_index_word_10 = uniform_index_word_8 + 0x1u;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_1 = int32_t(uniform_index_word_10) >= int32_t(uniform_index_word_9);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_1 =
        int32_t(uniform_index_word_16) > int32_t(0u - 0x8u) && !uniform_condition_1;

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_2 = uniform_condition_2 & uniform_condition_1;

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_2) {
        matrix_fragment_20 = permute_signed(matrix_fragment_20, matrix_fragment_27, 0x5410u);

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));

        matrix_fragment_21 = permute_signed(matrix_fragment_21, matrix_fragment_26, 0x5410u);

        matrix_fragment_17 = permute_signed(matrix_fragment_17, matrix_fragment_22, 0x5410u);

        matrix_fragment_16 = permute_signed(matrix_fragment_16, matrix_fragment_23, 0x5410u);

        matrix_fragment_18 = matrix_fragment_21;

        matrix_fragment_19 = matrix_fragment_20;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }
    uniform_index_word_9 = uniform_index_word_7 * uniform_index_word_9 + uniform_index_word_9;

    uniform_index_word_7 = uniform_index_word_7 + 0x1u;

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_9 = uniform_index_word_8 + uniform_index_word_9;

    uniform_condition_2 = int32_t(uniform_index_word_7) >= int32_t(uniform_index_word_5);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    uniform_condition_2 =
        int32_t(uniform_index_word_6) > int32_t(0u - 0x8u) && !uniform_condition_2;

    matrix_fragment_7 = uniform_index_word_9;

    condition_0 = (address_word_146 | matrix_fragment_31 | packed_half_30) != 0;

    uniform_condition_0 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = matrix_fragment_7 * 0x400u + packed_half_0;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                         select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_18 = permute_signed(matrix_fragment_18, matrix_fragment_5, 0x5410u);

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));

        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_accumulator_2, 0x5410u);

        matrix_fragment_17 = permute_signed(matrix_fragment_17, packed_half_0, 0x5410u);

        matrix_fragment_16 = permute_signed(matrix_fragment_16, matrix_fragment_21, 0x5410u);

        store_four_words_streaming<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                                matrix_fragment_16, matrix_fragment_17,
                                                matrix_fragment_18, matrix_fragment_19,
                                                memory_ranges, error_counts);
    }
    uniform_condition_1 = uniform_condition_2 & uniform_condition_1;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));

        matrix_fragment_7 = permute_signed(matrix_fragment_4, matrix_fragment_15, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_10, matrix_fragment_11, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x1000ull,
            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
            memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_index_word_6 = parameters[parameter_publication_optional_low_16];
    uniform_index_word_7 = parameters[parameter_publication_optional_high_17];

    uniform_index_word_5 = uniform_block_index_17 + uniform_index_word_4;

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
extern "C" __global__
__maxnreg__(248) void phoenix_window_attention_channels_256_input_view(KernelArguments parameters) {
    run_window_attention_channels_256_input_view<false>(parameters.words, nullptr, nullptr);
}
