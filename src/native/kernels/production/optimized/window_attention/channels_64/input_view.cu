// DLSS5 Phoenix optimized kernel: window_attention/channels_64/input_view.cu
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
__device__ __forceinline__ void run_window_attention_channels_64_input_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t packed_half_8 = 0;
    uint32_t packed_half_9 = 0;
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
    uint32_t matrix_accumulator_24 = 0;
    uint32_t matrix_accumulator_25 = 0;
    uint32_t matrix_accumulator_26 = 0;
    uint32_t matrix_accumulator_27 = 0;
    uint32_t matrix_accumulator_28 = 0;
    uint32_t matrix_accumulator_29 = 0;
    uint32_t matrix_fragment_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_fragment_32 = 0;
    uint32_t matrix_fragment_33 = 0;
    uint32_t matrix_fragment_34 = 0;
    uint32_t matrix_fragment_35 = 0;
    uint32_t matrix_accumulator_36 = 0;
    uint32_t matrix_accumulator_37 = 0;
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
    uint32_t packed_half_60 = 0;
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
    uint32_t packed_half_82 = 0;
    uint32_t matrix_fragment_83 = 0;
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
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
    uint32_t matrix_accumulator_114 = 0;
    uint32_t matrix_accumulator_115 = 0;
    uint32_t matrix_fragment_116 = 0;
    uint32_t matrix_fragment_117 = 0;
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
    uint32_t matrix_fragment_136 = 0;
    uint32_t matrix_fragment_137 = 0;
    uint32_t matrix_fragment_138 = 0;
    uint32_t matrix_fragment_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t packed_half_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t matrix_accumulator_150 = 0;
    uint32_t matrix_accumulator_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t address_word_156 = 0;
    uint32_t index_word_157 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t index_word_163 = 0;
    uint32_t float_bits_164 = 0;
    uint32_t thread_row_165 = 0;
    uint32_t uniform_parameter_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_parameter_word_11 = 0;
    uint32_t uniform_reused_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_block_index_15 = 0;
    uint32_t uniform_block_index_16 = 0;
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

    address_word_156 = threadIdx.x & 31u;

    matrix_fragment_10 = parameters[parameter_optional_control_20];
    matrix_fragment_11 = parameters[parameter_optional_control_21];

    uniform_parameter_word_11 = parameters[parameter_tensor_height_8];

    condition_3 = false;

    thread_row_165 = threadIdx.y;

    uniform_parameter_word_4 = parameters[parameter_optional_control_10];
    uniform_index_word_5 = parameters[parameter_optional_control_11];

    matrix_fragment_6 = uint32_t(condition_3) << 3;

    uniform_block_index_16 = blockIdx.y;

    uniform_index_word_10 = parameters[parameter_tensor_width_9];

    condition_2 = int32_t(matrix_fragment_6) != int32_t(0u);

    uniform_block_index_15 = blockIdx.x;

    condition_0 = int32_t(matrix_fragment_10) > int32_t(0u);

    condition_1 = int32_t(matrix_fragment_11) > int32_t(0u);

    matrix_fragment_5 = condition_0 ? matrix_fragment_10 : uniform_parameter_word_11;

    matrix_fragment_11 = condition_1 ? matrix_fragment_11 : uniform_index_word_10;

    index_word_163 = address_word_156;

    uniform_index_word_5 = (uniform_block_index_16 << 0x3u) + uniform_index_word_5;

    matrix_fragment_7 = address_word_156;

    packed_half_0 = thread_row_165 * 0x4u;

    address_word_162 = uint32_t(int32_t(index_word_163) >> 0x2u);

    matrix_fragment_10 = matrix_fragment_11 * 0x4u;

    condition_0 = uint32_t(matrix_fragment_5) != uint32_t(0x1u);

    condition_1 = true;

    uniform_index_word_14 = (uniform_block_index_15 << 0x3u) + uniform_parameter_word_4;

    matrix_fragment_4 = packed_half_0 + 0x2u;

    matrix_accumulator_2 =
        uint32_t(combine_words(address_word_162, 0u) << 0x2u >> 32) + address_word_162;

    matrix_fragment_4 =
        uint32_t(int64_t(int32_t(matrix_fragment_7)) * (1ll << 0x1cu) >> 32) + matrix_fragment_4;

    matrix_accumulator_3 = matrix_accumulator_2 & 0xfffffffcu;

    matrix_accumulator_2 = 0u;

    matrix_fragment_13 = matrix_fragment_4 + uniform_index_word_5;

    matrix_fragment_22 = address_word_162 + uniform_index_word_14 + (0u - matrix_accumulator_3);

    condition_5 = int32_t(matrix_fragment_13) >= int32_t(matrix_fragment_5);

    condition_6 = int32_t(matrix_fragment_22) >= int32_t(matrix_fragment_11);
    if (condition_0) {
        matrix_accumulator_3 = address_word_156;

        matrix_accumulator_3 =
            uint32_t(int64_t(int32_t(matrix_accumulator_3)) * (1ll << 0x1cu) >> 32) + packed_half_0;

        matrix_accumulator_3 = matrix_accumulator_3 + uniform_index_word_5;

        condition_1 = int32_t(matrix_accumulator_3) >= int32_t(matrix_fragment_5);

        matrix_accumulator_2 = matrix_fragment_10 * matrix_accumulator_3;

        condition_2 = int32_t(matrix_accumulator_3) < int32_t(0u) || condition_1;

        matrix_fragment_6 = uint32_t(condition_2) << 2;

        condition_1 = !condition_2;
    }
    condition_5 = int32_t(matrix_fragment_13) < int32_t(0u) || condition_5;

    condition_6 = int32_t(matrix_fragment_22) > -1 && !condition_6;

    condition_2 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_2;

    packed_half_0 =
        uint32_t(int64_t(int32_t(matrix_fragment_7)) * (1ll << 0x1cu) >> 32) + packed_half_0;

    uniform_reused_word_12 = 0u;
    uniform_reused_word_13 = 0u;

    condition_3 = false;

    address_word_158 = parameters[parameter_trained_weights_low_4];
    address_word_159 = parameters[parameter_trained_weights_high_5];

    if (condition_0) {
        condition_3 = condition_5;
    }

    condition_2 = condition_1 & (condition_2 | condition_6);

    matrix_fragment_14 = packed_half_0 + uniform_index_word_5;

    packed_half_9 = uint32_t(condition_3) << 3;

    packed_half_8 = uint32_t(condition_2) << 2;

    condition_2 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_3;

    condition_3 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_5);

    condition_1 = true;

    if (condition_0) {
        condition_1 = !condition_5;
    }

    matrix_fragment_4 = matrix_fragment_22 + 0x4u;

    condition_3 = int32_t(matrix_fragment_14) < int32_t(0u) || condition_3;

    condition_1 = condition_1 & (condition_2 | condition_6);

    matrix_accumulator_3 = condition_6 ? 0x40u : 0u;

    condition_6 = false;

    condition_2 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_11);

    if (condition_0) {
        condition_6 = condition_3;
    }

    matrix_fragment_7 = uint32_t(condition_1) << 1;

    condition_4 = int32_t(matrix_fragment_4) > -1 && !condition_2;

    condition_1 = true;

    condition_2 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_6;

    if (condition_0) {
        condition_1 = !condition_3;
    }

    matrix_fragment_18 = uint32_t(condition_4) << 4;

    condition_1 = condition_1 & (condition_2 | condition_4);

    condition_4 = false;

    if (condition_0) {
        condition_4 = condition_5;
    }

    matrix_fragment_16 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(packed_half_8) != int32_t(0u);

    matrix_fragment_19 = matrix_fragment_19 & 0xffffffbfu;

    condition_2 = true;

    matrix_fragment_12 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_7) != int32_t(0u);

    if (condition_0) {
        condition_2 = !condition_5;
    }

    if (condition_4) {
        matrix_fragment_19 = 0x40u | matrix_fragment_19;
    }

    packed_half_0 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_18) != int32_t(0u);

    condition_4 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_4;

    matrix_fragment_15 = condition_6 ? 0x40u : 0u;

    matrix_fragment_19 = matrix_fragment_19 & 0xfffffbffu;

    condition_2 = condition_2 & (condition_4 | condition_1);

    condition_4 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_3;

    condition_6 = int32_t(matrix_accumulator_3) != int32_t(0u);

    matrix_fragment_17 = uint32_t(condition_2) << 2;

    condition_2 = int32_t(matrix_fragment_15) != int32_t(0u);

    matrix_fragment_7 = uint32_t(condition_0);

    condition_0 = int32_t(packed_half_9) != int32_t(0u);

    index_word_157 = index_word_163 & 0xfffffffcu;

    matrix_fragment_15 = uint32_t(condition_2) << 2;

    condition_2 = int32_t(matrix_fragment_6) != int32_t(0u);

    if (condition_4) {
        matrix_fragment_19 = 0x400u | matrix_fragment_19;
    }

    condition_4 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_4;

    condition_0 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_0;

    matrix_fragment_6 = uint32_t(condition_2) << 2;

    condition_2 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_3;

    matrix_fragment_19 = matrix_fragment_19 & 0xfffffdffu;

    condition_3 = condition_2 & (condition_4 | condition_6);

    condition_4 = condition_2 & (condition_4 | condition_1);

    matrix_accumulator_3 = uint32_t(condition_3) << 3;

    condition_3 = uint32_t(matrix_fragment_5) != uint32_t(0x1u) && condition_5;

    condition_1 = int32_t(packed_half_0) != int32_t(0u);

    condition_2 = uint32_t(matrix_fragment_5) == uint32_t(0x1u) || !condition_5;

    packed_half_0 = 0u;

    if (condition_3) {
        matrix_fragment_19 = 0x200u | matrix_fragment_19;
    }

    if (condition_1) {
        packed_half_8 = parameters[parameter_input_low_0];
        packed_half_9 = parameters[parameter_input_high_1];
    }

    condition_3 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) || condition_3;

    condition_5 = condition_2 & (condition_3 | condition_6);

    condition_6 = int32_t(matrix_fragment_18) != int32_t(0u);

    condition_6 = condition_2 & (condition_3 | condition_6);

    condition_3 = int32_t(matrix_accumulator_3) != int32_t(0u);

    if (condition_1) {
        matrix_accumulator_3 = matrix_fragment_22 * 0x4u;
    }

    condition_2 = int32_t(matrix_fragment_6) != int32_t(0u);

    if (condition_1) {
        matrix_accumulator_3 = !condition_0 ? matrix_accumulator_3 : 0u;
    }

    condition_0 = int32_t(matrix_fragment_7) != int32_t(0u);

    if (condition_0) {
        packed_half_0 = matrix_fragment_10 * matrix_fragment_13;
    }

    if (condition_1) {
        packed_half_0 = packed_half_0 + address_word_156 + (0u - index_word_157);

        matrix_accumulator_3 = matrix_accumulator_3 + packed_half_0;
    }

    packed_half_0 = 0u;

    if (condition_1) {
        split_address_words(packed_half_8, packed_half_9,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_8, packed_half_9));

        packed_half_0 = load_aligned_word<CheckBounds>(combine_words(packed_half_8, packed_half_9),
                                                       memory_ranges, error_counts);
    }

    condition_1 = int32_t(matrix_fragment_12) != int32_t(0u);

    condition_2 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_2;

    if (condition_1) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];

        matrix_accumulator_3 = matrix_fragment_22 * 0x4u;

        matrix_accumulator_2 = matrix_accumulator_2 + address_word_156 + (0u - index_word_157);

        matrix_accumulator_3 = !condition_2 ? matrix_accumulator_3 : 0u;
    }

    condition_2 = int32_t(matrix_fragment_15) != int32_t(0u);

    if (condition_1) {
        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_2;

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    matrix_accumulator_3 = 0u;

    if (condition_1) {
        matrix_accumulator_3 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    condition_1 = int32_t(matrix_fragment_16) != int32_t(0u);

    condition_2 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_2;

    if (condition_1) {
        packed_half_8 = parameters[parameter_input_low_0];
        packed_half_9 = parameters[parameter_input_high_1];

        matrix_accumulator_2 = matrix_fragment_4 * 0x4u;

        matrix_fragment_12 = !condition_2 ? matrix_accumulator_2 : 0u;
    }

    matrix_accumulator_2 = 0u;

    if (condition_0) {
        matrix_accumulator_2 = matrix_fragment_10 * matrix_fragment_14;
    }

    condition_2 = int32_t(matrix_fragment_17) != int32_t(0u);

    matrix_fragment_17 = 0u;

    if (condition_1) {
        matrix_fragment_15 = matrix_accumulator_2 + address_word_156 + (0u - index_word_157);
    }

    if (condition_0) {
        matrix_fragment_17 = matrix_fragment_10 * matrix_fragment_13;
    }

    matrix_fragment_6 = condition_0 ? matrix_fragment_14 : 0u;

    matrix_fragment_14 = condition_0 ? matrix_fragment_13 : 0u;

    condition_0 = (matrix_fragment_19 & 0x40u) != 0;

    if (condition_1) {
        matrix_fragment_15 = matrix_fragment_12 + matrix_fragment_15;
    }

    matrix_accumulator_2 = 0u;

    condition_0 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_0;

    if (condition_2) {
        matrix_fragment_18 = matrix_fragment_4 * 0x4u;
    }

    if (condition_1) {
        split_address_words(packed_half_8, packed_half_9,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_8, packed_half_9));
    }

    if (condition_2) {
        matrix_fragment_18 = !condition_0 ? matrix_fragment_18 : 0u;
    }

    condition_0 = (matrix_fragment_19 & 0x400u) != 0;

    if (condition_1) {
        matrix_accumulator_2 = load_aligned_word<CheckBounds>(
            combine_words(packed_half_8, packed_half_9), memory_ranges, error_counts);
    }

    condition_1 = (matrix_fragment_19 & 0x200u) != 0;

    if (condition_3) {
        matrix_fragment_16 = matrix_fragment_5 * 0x3u + matrix_fragment_6;

        matrix_fragment_12 = matrix_fragment_5 + matrix_fragment_6;
    }

    condition_0 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_0;

    if (condition_4) {
        matrix_accumulator_24 = matrix_fragment_5 + matrix_fragment_6;
    }

    condition_1 = uint32_t(matrix_fragment_11) == uint32_t(0x1u) && !condition_1;

    if (condition_4) {
        matrix_accumulator_27 = matrix_fragment_5 * 0x2u + matrix_fragment_6;
    }

    if (condition_5) {
        matrix_fragment_15 = matrix_fragment_5 + matrix_fragment_14;
    }

    if (condition_4) {
        matrix_accumulator_29 = matrix_fragment_5 * 0x3u + matrix_fragment_6;
    }

    if (condition_6) {
        matrix_accumulator_26 = matrix_fragment_5 + matrix_fragment_14;
    }

    if (condition_3) {
        packed_half_8 = matrix_fragment_5 * 0x2u + matrix_fragment_6;
    }

    matrix_fragment_11 = 0u - index_word_157 + address_word_156;

    if (condition_5) {
        matrix_fragment_20 = matrix_fragment_5 * 0x2u + matrix_fragment_14;
    }

    if (condition_3) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_21 = matrix_fragment_5 * 0x3u + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_17 = matrix_fragment_17 + address_word_156 + (0u - index_word_157);
    }

    if (condition_6) {
        matrix_accumulator_28 = matrix_fragment_5 * 0x2u + matrix_fragment_14;
    }

    matrix_fragment_34 = 0u;

    if (condition_6) {
        matrix_fragment_30 = matrix_fragment_5 * 0x3u + matrix_fragment_14;
    }

    if (condition_3) {
        packed_half_9 = matrix_fragment_22 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_23 = matrix_fragment_18 + matrix_fragment_17;
    }

    if (condition_3) {
        matrix_fragment_5 = matrix_fragment_22 * 0x4u;

        matrix_fragment_13 = matrix_fragment_10 * packed_half_8 + matrix_fragment_11;

        matrix_fragment_14 = !condition_0 ? packed_half_9 : 0u;

        matrix_fragment_12 = matrix_fragment_10 * matrix_fragment_12 + matrix_fragment_11;

        matrix_fragment_5 = !condition_0 ? matrix_fragment_5 : 0u;
    }

    if (condition_2) {
        packed_half_8 = parameters[parameter_input_low_0];
        packed_half_9 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_19 = matrix_fragment_22 * 0x4u;

        matrix_fragment_33 = matrix_fragment_14 + matrix_fragment_13;

        matrix_fragment_16 = matrix_fragment_10 * matrix_fragment_16 + matrix_fragment_11;

        matrix_fragment_5 = matrix_fragment_5 + matrix_fragment_12;
    }

    if (condition_5) {
        matrix_fragment_14 = matrix_fragment_22 * 0x4u;
    }

    if (condition_3) {
        matrix_fragment_19 = !condition_0 ? matrix_fragment_19 : 0u;
    }

    if (condition_5) {
        matrix_fragment_15 = matrix_fragment_10 * matrix_fragment_15 + matrix_fragment_11;

        matrix_fragment_12 = parameters[parameter_input_low_0];
        matrix_fragment_13 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_5) {
        matrix_fragment_14 = !condition_1 ? matrix_fragment_14 : 0u;
    }

    if (condition_3) {
        matrix_fragment_35 = matrix_fragment_19 + matrix_fragment_16;
    }

    if (condition_5) {
        matrix_fragment_5 = matrix_fragment_22 * 0x4u;

        matrix_fragment_32 = matrix_fragment_10 * matrix_fragment_21 + matrix_fragment_11;
    }

    if (condition_6) {
        matrix_fragment_18 = parameters[parameter_input_low_0];
        matrix_fragment_19 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_fragment_20 = matrix_fragment_10 * matrix_fragment_20 + matrix_fragment_11;

        matrix_fragment_21 = matrix_fragment_14 + matrix_fragment_15;

        matrix_fragment_5 = !condition_1 ? matrix_fragment_5 : 0u;
    }

    if (condition_4) {
        matrix_fragment_14 = matrix_fragment_4 * 0x4u;
    }

    if (condition_6) {
        matrix_fragment_15 = matrix_fragment_4 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_16 = parameters[parameter_input_low_0];
        matrix_fragment_17 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        split_address_words(packed_half_8, packed_half_9,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_8, packed_half_9));
    }

    if (condition_5) {
        matrix_fragment_5 = matrix_fragment_5 + matrix_fragment_20;
    }

    if (condition_4) {
        matrix_fragment_14 = !condition_0 ? matrix_fragment_14 : 0u;

        matrix_accumulator_25 = matrix_fragment_10 * matrix_accumulator_24 + matrix_fragment_11;
    }

    if (condition_2) {
        matrix_fragment_34 = load_aligned_word<CheckBounds>(
            combine_words(packed_half_8, packed_half_9), memory_ranges, error_counts);
    }

    if (condition_6) {
        matrix_accumulator_26 = matrix_fragment_10 * matrix_accumulator_26 + matrix_fragment_11;

        matrix_fragment_15 = !condition_1 ? matrix_fragment_15 : 0u;
    }

    if (condition_4) {
        matrix_fragment_43 = matrix_fragment_10 * matrix_accumulator_27 + matrix_fragment_11;

        matrix_fragment_23 = matrix_accumulator_25 + matrix_fragment_14;
    }

    if (condition_6) {
        matrix_fragment_45 = matrix_fragment_10 * matrix_accumulator_28 + matrix_fragment_11;
    }

    if (condition_4) {
        matrix_fragment_47 = matrix_fragment_10 * matrix_accumulator_29 + matrix_fragment_11;
    }

    if (condition_6) {
        matrix_fragment_49 = matrix_fragment_10 * matrix_fragment_30 + matrix_fragment_11;
    }

    if (condition_5) {
        split_address_words(packed_half_8, packed_half_9,
                            int64_t(int32_t(matrix_fragment_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));
    }

    if (condition_6) {
        matrix_accumulator_25 = matrix_accumulator_26 + matrix_fragment_15;
    }

    if (condition_3) {
        matrix_fragment_20 = parameters[parameter_input_low_0];
        matrix_fragment_21 = parameters[parameter_input_high_1];
    }

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_27 = address_word_156 + 0x4u + (0u - index_word_157);

    if (condition_6) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    matrix_fragment_19 = address_word_156 + 0x8u + (0u - index_word_157);

    matrix_accumulator_109 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

    if (condition_4) {
        matrix_accumulator_25 = matrix_fragment_4 * 0x4u;

        matrix_accumulator_28 = parameters[parameter_input_low_0];
        matrix_accumulator_29 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_30 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        uint64_t(uint32_t(matrix_accumulator_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    if (condition_5) {
        matrix_fragment_31 = load_aligned_word<CheckBounds>(
            combine_words(packed_half_8, packed_half_9), memory_ranges, error_counts);

        matrix_accumulator_24 = matrix_fragment_22 * 0x4u;
    }

    if (condition_4) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x7010ull, memory_ranges,
        error_counts);

    if (condition_3) {
        matrix_fragment_16 = parameters[parameter_input_low_0];
        matrix_fragment_17 = parameters[parameter_input_high_1];
    }

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_19 = address_word_156 + 0xcu + (0u - index_word_157);

    if (condition_6) {
        matrix_accumulator_37 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), memory_ranges, error_counts);
    }

    if (condition_4) {
        packed_half_8 = !condition_0 ? matrix_accumulator_25 : 0u;
    }

    if (condition_5) {
        matrix_fragment_39 = !condition_1 ? matrix_accumulator_24 : 0u;

        matrix_fragment_22 = parameters[parameter_input_low_0];
        matrix_fragment_23 = parameters[parameter_input_high_1];
    }

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    if (condition_4) {
        matrix_accumulator_36 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), memory_ranges, error_counts);

        matrix_fragment_13 = matrix_fragment_43 + packed_half_8;
    }

    if (condition_3) {
        split_address_words(packed_half_8, packed_half_9,
                            int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (condition_4) {
        matrix_accumulator_24 = parameters[parameter_input_low_0];
        matrix_accumulator_25 = parameters[parameter_input_high_1];
    }

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x7010ull, memory_ranges,
        error_counts);

    if (condition_6) {
        matrix_accumulator_26 = parameters[parameter_input_low_0];
        matrix_accumulator_27 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_12 = matrix_fragment_4 * 0x4u;
    }

    matrix_fragment_42 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x7010ull, memory_ranges,
        error_counts);

    if (condition_5) {
        matrix_fragment_18 = parameters[parameter_input_low_0];
        matrix_fragment_19 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_fragment_20 = parameters[parameter_input_low_0];
        matrix_fragment_21 = parameters[parameter_input_high_1];

        matrix_fragment_10 = matrix_fragment_4 * 0x4u;

        matrix_fragment_4 = matrix_fragment_4 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_12 = !condition_0 ? matrix_fragment_12 : 0u;
    }

    if (condition_6) {
        matrix_fragment_10 = !condition_1 ? matrix_fragment_10 : 0u;

        matrix_fragment_4 = !condition_1 ? matrix_fragment_4 : 0u;
    }

    if (condition_5) {
        matrix_fragment_11 = matrix_fragment_39 + matrix_fragment_32;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_33)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    if (condition_6) {
        matrix_fragment_15 = matrix_fragment_45 + matrix_fragment_10;

        matrix_fragment_35 = matrix_fragment_49 + matrix_fragment_4;
    }

    if (condition_4) {
        matrix_fragment_17 = matrix_fragment_47 + matrix_fragment_12;
    }

    if (condition_5) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    if (condition_3) {
        matrix_fragment_32 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    matrix_accumulator_24 = 0;
    matrix_accumulator_25 = 0;

    if (condition_3) {
        matrix_fragment_33 = load_aligned_word<CheckBounds>(
            combine_words(packed_half_8, packed_half_9), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    matrix_fragment_19 = address_word_156 + 0x14u + (0u - index_word_157);

    if (condition_5) {
        matrix_fragment_23 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    if (condition_5) {
        matrix_fragment_22 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    packed_half_9 = address_word_156 + 0x10u + (0u - index_word_157);

    if (condition_4) {
        matrix_accumulator_24 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), memory_ranges, error_counts);
    }

    matrix_fragment_21 = address_word_156 + 0x18u + (0u - index_word_157);

    if (condition_4) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_28, matrix_accumulator_29));
    }

    matrix_accumulator_29 = address_word_156 + 0x1cu + (0u - index_word_157);

    if (condition_6) {
        matrix_accumulator_25 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
    }

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(packed_half_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    if (condition_4) {
        matrix_accumulator_26 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), memory_ranges, error_counts);
    }

    split_address_words(packed_half_8, packed_half_9,
                        uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    if (condition_6) {
        matrix_accumulator_27 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_28 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_accumulator_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_29 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_8, packed_half_9) + 0x7010ull, memory_ranges, error_counts);

    matrix_fragment_35 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_fragment_44 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x7010ull, memory_ranges,
        error_counts);

    uniform_parameter_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_4));

    uniform_parameter_word_4 = 0u - uniform_parameter_word_4 + uniform_index_word_10;

    uniform_condition_1 = uint32_t(uniform_parameter_word_4) != 0u;

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);

    uniform_address_word_7 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);

    if (uniform_condition_1) {
        uniform_index_word_8 = uniform_parameter_word_4 + 0xffffffffu;
    }

    packed_half_152 = unpack_fp8_bits(packed_half_0);

    if (uniform_condition_1) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);
    }

    matrix_accumulator_151 = unpack_fp8_bits(packed_half_0 >> 16);

    if (uniform_condition_1) {
        uniform_index_word_9 =
            uint32_t(combine_words(uniform_index_word_9, 0u) << 0x3u >> 32) + uniform_index_word_8;
    }

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x2u >> 32) + uniform_index_word_5;

    uniform_address_word_7 =
        uint32_t(combine_words(uniform_address_word_7, 0u) << 0x2u >> 32) + uniform_index_word_14;

    matrix_accumulator_102 = 0u;

    if (uniform_condition_1) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_9) >> 0x3u);
    }

    uniform_parameter_word_4 = 0u;

    uniform_condition_0 = true;

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_6) >> 0x2u);

    uniform_address_word_7 = uint32_t(int32_t(uniform_address_word_7) >> 0x2u);

    if (uniform_condition_1) {
        uniform_parameter_word_4 =
            uniform_index_word_9 * uniform_block_index_16 + uniform_block_index_16;
    }

    packed_half_155 = unpack_fp8_bits(matrix_accumulator_3);

    packed_half_154 = unpack_fp8_bits(matrix_accumulator_3 >> 16);

    matrix_fragment_138 = unpack_fp8_bits(matrix_accumulator_2);

    matrix_fragment_137 = unpack_fp8_bits(matrix_accumulator_2 >> 16);

    matrix_fragment_136 = unpack_fp8_bits(matrix_fragment_34);

    packed_half_113 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_150 = unpack_fp8_bits(matrix_fragment_30);

    packed_half_149 = unpack_fp8_bits(matrix_fragment_30 >> 16);

    packed_half_148 = unpack_fp8_bits(matrix_fragment_31);

    packed_half_147 = unpack_fp8_bits(matrix_fragment_31 >> 16);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_138),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_110 = unpack_fp8_bits(matrix_accumulator_37);

    packed_half_0 = unpack_fp8_bits(matrix_accumulator_37 >> 16);

    packed_half_112 = unpack_fp8_bits(matrix_accumulator_36);

    matrix_accumulator_111 = unpack_fp8_bits(matrix_accumulator_36 >> 16);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_136),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_137),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_146 = unpack_fp8_bits(matrix_fragment_32);

    matrix_accumulator_145 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_accumulator_142 = unpack_fp8_bits(matrix_fragment_33);

    matrix_accumulator_141 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_accumulator_140 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_139 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_accumulator_144 = unpack_fp8_bits(matrix_fragment_22);

    matrix_accumulator_143 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_24);

    matrix_fragment_6 = unpack_fp8_bits(matrix_accumulator_24 >> 16);

    matrix_fragment_7 = unpack_fp8_bits(matrix_accumulator_25);

    packed_half_8 = unpack_fp8_bits(matrix_accumulator_25 >> 16);

    packed_half_9 = unpack_fp8_bits(matrix_accumulator_26);

    matrix_fragment_5 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

    matrix_fragment_4 = unpack_fp8_bits(matrix_accumulator_27);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_9),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_8),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_139),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_44));
    do {
        matrix_accumulator_105 = matrix_accumulator_102 * 0x800u;

        split_address_words(matrix_accumulator_104, matrix_accumulator_105,
                            uint64_t(uint32_t(matrix_accumulator_105)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        split_address_words(matrix_accumulator_104, matrix_accumulator_105,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_104, matrix_accumulator_105));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105), matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x200ull,
            matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1000ull,
            matrix_accumulator_80, matrix_accumulator_81, packed_half_82, matrix_fragment_83,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x400ull,
            matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x600ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1200ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1600ull,
            matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
            matrix_accumulator_95, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x800ull,
            matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1400ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1800ull,
            packed_half_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        matrix_accumulator_101 = matrix_accumulator_102 * 0x400u;

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4000ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4200ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_150),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_137),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_151),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_138),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_136),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_139),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_8),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_120 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                          select_half_operand<2, false, false>(matrix_fragment_52));

        packed_half_153 = 0x2b28u;

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_48, matrix_fragment_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_48, matrix_fragment_49, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_144),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_96, matrix_accumulator_97, 0u, 0u);

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_9),
                          select_half_operand<2, false, false>(matrix_accumulator_120));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_96, matrix_accumulator_97, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_98, matrix_fragment_99, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_106, matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_82, matrix_fragment_83, matrix_fragment_116,
                                 matrix_fragment_117);

        multiply_accumulate_half(packed_half_82, matrix_fragment_83, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 packed_half_82, matrix_fragment_83, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_98, matrix_fragment_99, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_116), 0x44004400u);

        matrix_fragment_75 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_117), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_78,
                                 matrix_fragment_79);

        matrix_fragment_78 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_92, matrix_accumulator_93,
                                 matrix_accumulator_114, matrix_accumulator_115);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_78),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_94, matrix_accumulator_95,
                                 matrix_accumulator_126, matrix_accumulator_127);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_106), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_107), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_132,
                                 matrix_accumulator_133);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0xc400c400u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_74),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_75), 0x3b283b28u);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x44004400u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78),
            select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_76), 0x44004400u);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_77), 0x44004400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_78 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_75 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0x44004400u);

        matrix_fragment_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_116),
                                select_half_operand<2, false, false>(matrix_fragment_79));

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_78),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0xc400c400u);

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_117),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_74),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78),
            select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_75), 0x3b283b28u);

        matrix_fragment_79 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<2, false, false>(matrix_fragment_79));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_75 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_77 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_60, matrix_fragment_61, matrix_accumulator_94,
                                 matrix_accumulator_95);

        matrix_fragment_78 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_78));

        matrix_fragment_57 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 packed_half_60, matrix_fragment_61, matrix_accumulator_92,
                                 matrix_accumulator_93);

        matrix_fragment_61 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_116),
                          select_half_operand<2, false, false>(0u));

        packed_half_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x44004400u);

        matrix_fragment_76 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_60 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_60),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        packed_half_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_86), 0x3b283b28u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<2, false, false>(matrix_fragment_86));

        packed_half_60 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_79 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4400ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_accumulator_81 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        packed_half_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_81 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81), 0xc400c400u);

        packed_half_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(packed_half_60), 0x3b283b28u);

        packed_half_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_82),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(packed_half_60));

        packed_half_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_81),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_80 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        packed_half_60 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0xc400c400u);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        packed_half_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_81 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_82),
                                              select_half_operand<2, false, false>(0u));

        matrix_accumulator_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(packed_half_60), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        matrix_fragment_56 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_accumulator_80));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0xa00ull,
            matrix_accumulator_80, matrix_accumulator_81, packed_half_82, matrix_fragment_83,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4600ull,
            matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
            memory_ranges, error_counts);

        matrix_fragment_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_accumulator_121 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_122 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(packed_half_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_58 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                          select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                          select_half_operand<2, false, false>(matrix_accumulator_121));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1a00ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        matrix_fragment_57 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_accumulator_122));

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x44004400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_62, matrix_fragment_63, packed_half_60,
                                 matrix_fragment_61);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x44004400u);

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        packed_half_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        packed_half_60 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

        matrix_fragment_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_59 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_60),
            select_half_operand<2, false, false>(matrix_accumulator_103), 0x3b283b28u);

        packed_half_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_96,
                                 matrix_accumulator_97);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_accumulator_130 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4a00ull,
            packed_half_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4800ull,
            matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
            memory_ranges, error_counts);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_130),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130),
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x3b283b28u);

        matrix_fragment_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_90 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0xc400c400u);

        matrix_fragment_91 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0xc400c400u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_90 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0x44004400u);

        matrix_accumulator_130 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_130),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130),
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x3b283b28u);

        matrix_accumulator_130 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_130 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x3b283b28u);

        matrix_accumulator_103 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_fragment_98 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_127 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                          select_half_operand<2, false, false>(matrix_fragment_98));

        matrix_fragment_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_accumulator_114 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0xc400c400u);

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                          select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_accumulator_130));

        matrix_accumulator_114 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x3b283b28u);

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_126 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_103 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(matrix_accumulator_126));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_accumulator_114));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_116,
                                 matrix_fragment_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_64,
                                 matrix_fragment_65);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_66,
                                 matrix_fragment_67);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0xc00ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1c00ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0xe00ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_104, matrix_accumulator_105) + 0x1e00ull,
            matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_80, matrix_accumulator_81, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_80, matrix_accumulator_81, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 packed_half_82, matrix_fragment_83, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 packed_half_82, matrix_fragment_83, 0u, 0u);

        packed_half_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

        packed_half_82 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_88,
                                 matrix_fragment_89);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_82),
            select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_90,
                                 matrix_fragment_91);

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_80,
                                 matrix_accumulator_81);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        packed_half_82 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0xc400c400u);

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_82),
            select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

        packed_half_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        packed_half_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(packed_half_82), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0x44004400u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(packed_half_82));

        packed_half_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_86 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x44004400u);

        matrix_fragment_85 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_accumulator_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_fragment_83 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        packed_half_82 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0xc400c400u);

        matrix_fragment_87 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_86), 0xc400c400u);

        matrix_fragment_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_85),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0xc400c400u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85),
            select_half_operand<2, false, false>(matrix_fragment_86), 0x3b283b28u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_87),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_82),
            select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

        packed_half_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_87),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                select_half_operand<2, false, false>(matrix_fragment_86));

        packed_half_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(packed_half_82), 0x3b283b28u);

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(packed_half_82));

        matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_85 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_86 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85), 0xc400c400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_fragment_83 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_accumulator_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

        matrix_accumulator_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                          select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_accumulator_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_accumulator_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0xc400c400u);

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                          select_half_operand<2, false, false>(matrix_accumulator_80));

        matrix_accumulator_95 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0xc400c400u);

        matrix_accumulator_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        matrix_fragment_87 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_86),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4c00ull,
            matrix_accumulator_80, matrix_accumulator_81, packed_half_82, matrix_fragment_83,
            memory_ranges, error_counts);

        matrix_accumulator_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(matrix_accumulator_94), 0x3b283b28u);

        matrix_accumulator_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x44004400u);

        matrix_accumulator_133 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_86),
            select_half_operand<2, false, false>(matrix_fragment_87), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4e00ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<2, false, false>(matrix_accumulator_94));

        matrix_accumulator_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0xc400c400u);

        matrix_accumulator_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_accumulator_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0xc400c400u);

        matrix_accumulator_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x3b283b28u);

        matrix_accumulator_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_94),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94),
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x3b283b28u);

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_accumulator_92));

        matrix_accumulator_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_105), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 packed_half_60, matrix_fragment_61, matrix_accumulator_106,
                                 matrix_accumulator_107);

        matrix_accumulator_101 = matrix_accumulator_102 * 0x200u;

        matrix_accumulator_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0xc400c400u);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_93 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_accumulator_133));

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_93 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0xc400c400u);

        matrix_accumulator_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_95),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x3b283b28u);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        matrix_accumulator_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95),
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x3b283b28u);

        matrix_accumulator_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_93),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_accumulator_96));

        packed_half_160 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93),
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6400ull,
            matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6200ull,
            matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
            matrix_accumulator_95, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6000ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6600ull,
            matrix_accumulator_100, matrix_accumulator_101, matrix_accumulator_102,
            matrix_accumulator_103, memory_ranges, error_counts);

        packed_half_160 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(packed_half_160));

        matrix_accumulator_133 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_accumulator_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_123 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(0u));

        packed_half_160 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_104 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_accumulator_133));

        matrix_accumulator_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(matrix_accumulator_105));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(matrix_accumulator_123));

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(packed_half_160));

        multiply_accumulate_half(
            matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_56, matrix_fragment_57, matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_accumulator_104,
                                 matrix_accumulator_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_116, matrix_fragment_117);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(
            matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107, packed_half_60,
            matrix_fragment_61, matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(packed_half_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

        multiply_accumulate_half(packed_half_60, matrix_fragment_61, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_76, matrix_fragment_77, packed_half_60,
                                 matrix_fragment_61);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                                 matrix_fragment_41);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_66,
                                 matrix_fragment_67);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_68,
                                 matrix_fragment_69);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_42 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        multiply_accumulate_half(
            matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_62, matrix_fragment_63, matrix_accumulator_124, matrix_accumulator_125);

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0x44004400u);

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        packed_half_153 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        packed_half_153 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(packed_half_153), 0x3b283b28u);

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                           select_half_operand<2, false, false>(0u));

        packed_half_153 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                select_half_operand<2, false, false>(packed_half_153));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 packed_half_82, matrix_fragment_83, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 packed_half_82, matrix_fragment_83, matrix_fragment_116,
                                 matrix_fragment_117);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_116),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_117),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, matrix_fragment_38,
                                 matrix_fragment_39);

        uniform_condition_1 = uniform_condition_0;

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_96, matrix_accumulator_97,
                                 matrix_accumulator_24, matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_36, matrix_accumulator_37);

        uniform_condition_0 = false;

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_34,
                                 matrix_fragment_35);

        matrix_accumulator_102 = 0x1u;
    } while (uniform_condition_1);
    uniform_index_word_9 = 0u;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                         select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = 0x400u;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                         select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                         select_half_operand<2, false, false>(0u));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = (uniform_index_word_9 << 0x18u) + uniform_index_word_8;

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = thread_row_165 * 0x800u;

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                         select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(packed_half_9));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_111 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_15 = matrix_accumulator_3 + uniform_index_word_8;

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = address_word_156 * 0x10u + matrix_fragment_15;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                           select_half_operand<2, false, false>(packed_half_9));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_15, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(matrix_fragment_11));

    packed_half_9 = thread_row_165 * 0x300u;

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(packed_half_0));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_15 + 0x200u, matrix_fragment_52, matrix_fragment_53,
        matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                      select_half_operand<2, false, false>(matrix_fragment_13));

    split_address_words(packed_half_8, packed_half_9,
                        uint64_t(uint32_t(packed_half_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_15 + 0x400u, matrix_accumulator_108, matrix_accumulator_109,
        matrix_accumulator_110, matrix_accumulator_111, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_15 + 0x600u, matrix_accumulator_24, matrix_accumulator_25,
        matrix_accumulator_26, matrix_accumulator_27, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_8, packed_half_9));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x72a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x70a0ull, matrix_accumulator_36,
        matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x74a0ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x76a0ull, packed_half_8,
        packed_half_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x78a0ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x7aa0ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8aa0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x88a0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x90a0ull, packed_half_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ca0ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ea0ull, matrix_accumulator_96,
        matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x92a0ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    packed_half_0 = (address_word_156 << 0x4u) + uniform_index_word_8;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x400u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0x600u, matrix_fragment_116, matrix_fragment_117,
        matrix_accumulator_118, matrix_accumulator_119, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xc00u, matrix_fragment_56,
                                        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                        shared_bytes, memory_ranges, error_counts);

    split_address_words(packed_half_154, packed_half_155,
                        uint64_t(uint32_t(thread_row_165)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    packed_half_154 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_154, packed_half_155) + 0xe0a0ull, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_36, matrix_accumulator_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_138, matrix_fragment_139, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_136, matrix_fragment_137, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_8, packed_half_9, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_36, matrix_accumulator_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(packed_half_82, matrix_fragment_83, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             packed_half_8, packed_half_9, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             packed_half_60, matrix_fragment_61, matrix_accumulator_24,
                             matrix_accumulator_25);

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_138, matrix_fragment_139, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_138,
                             matrix_fragment_139);

    multiply_accumulate_half(matrix_fragment_136, matrix_fragment_137, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_136,
                             matrix_fragment_137);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_64, matrix_fragment_65, packed_half_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_134,
                             matrix_accumulator_135);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             packed_half_60, matrix_fragment_61, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_62, matrix_fragment_63, packed_half_112,
                             packed_half_113);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_116,
                             matrix_fragment_117, matrix_accumulator_118, matrix_accumulator_119,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_138),
                                      select_half_operand<2, false, false>(matrix_fragment_138),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_136),
                                      select_half_operand<2, false, false>(matrix_fragment_136),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_81 = result_word;
    }

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_139),
                                      select_half_operand<2, false, false>(matrix_fragment_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_137),
                                      select_half_operand<2, false, false>(matrix_fragment_137),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_85 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_accumulator_2));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                          select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xe00u, matrix_fragment_76,
                                        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xa00u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_88 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_accumulator_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_accumulator_3 = permute_signed(matrix_accumulator_2, matrix_accumulator_2, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_105 = result_word;
    }

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             packed_half_8, packed_half_9, 0u, 0u);

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                          select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_86));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_142 = result_word;
    }

    matrix_fragment_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                       select_half_operand<2, false, false>(matrix_fragment_86));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_87 = permute_signed(matrix_fragment_84, matrix_fragment_84, 0x5432u);

    matrix_accumulator_105 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_126,
                             matrix_accumulator_127);

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_2 = half_to_float_bits((matrix_fragment_85 & 65535u) * 65537u);

    matrix_fragment_89 = half_to_float_bits((matrix_fragment_85 >> 16) * 65537u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_128,
                             matrix_accumulator_129);

    matrix_accumulator_142 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_accumulator_2 = rsqrt_bits(matrix_accumulator_2);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_104 = half_to_float_bits((matrix_fragment_90 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_118,
                             matrix_accumulator_119);

    matrix_fragment_89 = rsqrt_bits(matrix_fragment_89);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_90 >> 16) * 65537u);

    packed_half_147 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_130,
                             matrix_accumulator_131);

    matrix_fragment_4 = permute_signed(matrix_accumulator_105, matrix_accumulator_105, 0x5432u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_124,
                             matrix_accumulator_125);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_3 = permute_signed(matrix_accumulator_142, matrix_accumulator_142, 0x5432u);

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_88 = rsqrt_bits(matrix_accumulator_104);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                     select_half_operand<2, false, false>(packed_half_147));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_105 = rsqrt_bits(matrix_fragment_90);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_151 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_8, packed_half_9, 0u, 0u);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_147), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_8 = result_word;
    }

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_12, matrix_fragment_13));

    __syncthreads();

    packed_half_146 = half_to_float_bits((matrix_fragment_5 & 65535u) * 65537u);

    packed_half_149 = half_to_float_bits((matrix_fragment_5 >> 16) * 65537u);

    matrix_fragment_5 = pack_half(matrix_fragment_89, matrix_accumulator_2);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                      select_half_operand<2, false, false>(matrix_accumulator_118),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_9 = pack_half(matrix_accumulator_105, matrix_fragment_88);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xa2a0ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xa4a0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    packed_half_152 = half_to_float_bits((matrix_accumulator_3 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_104), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_105 = result_word;
    }

    matrix_accumulator_107 = half_to_float_bits((matrix_accumulator_3 >> 16) * 65537u);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_146 = rsqrt_bits(packed_half_146);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_151));

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_140,
                             matrix_accumulator_141);

    matrix_accumulator_100 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                            select_half_operand<2, false, false>(packed_half_8));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xa0a0ull, matrix_accumulator_80,
        matrix_accumulator_81, packed_half_82, matrix_fragment_83, memory_ranges, error_counts);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(packed_half_9));

    matrix_accumulator_103 =
        permute_signed(matrix_accumulator_106, matrix_accumulator_106, 0x5432u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(packed_half_9));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_100), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_103));

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_accumulator_106 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x4100410u);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    packed_half_146 = pack_half(packed_half_149, packed_half_146);

    packed_half_149 = half_to_float_bits((matrix_accumulator_106 & 65535u) * 65537u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_150 = half_to_float_bits((matrix_accumulator_106 >> 16) * 65537u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_138),
                            select_half_operand<2, false, false>(packed_half_146));

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_142,
                             matrix_accumulator_143);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_136),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_104), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_153 = result_word;
    }

    matrix_fragment_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(packed_half_146));

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_26));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    packed_half_146 = rsqrt_bits(matrix_accumulator_150);

    matrix_accumulator_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_102 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_97 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_103 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_accumulator_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                     select_half_operand<2, false, false>(matrix_accumulator_96));

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                     select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_accumulator_101 =
        permute_signed(matrix_accumulator_100, matrix_accumulator_100, 0x5432u);

    matrix_accumulator_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                      select_half_operand<2, false, false>(matrix_accumulator_119),
                                      select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_97 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_97));

    packed_half_152 = rsqrt_bits(packed_half_152);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                        select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                        select_half_operand<2, false, false>(packed_half_9));

    matrix_accumulator_107 = rsqrt_bits(matrix_accumulator_107);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xa6a0ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_144,
                             matrix_accumulator_145);

    packed_half_149 = pack_half(packed_half_146, packed_half_149);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xa8a0ull, matrix_accumulator_96,
        matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    packed_half_146 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xaaa0ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_148), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_155 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_108), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_110 = result_word;
    }

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                     select_half_operand<2, false, false>(packed_half_153));

    packed_half_152 = pack_half(matrix_accumulator_107, packed_half_152);

    matrix_accumulator_151 = half_to_float_bits((packed_half_146 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xaca0ull, matrix_accumulator_104,
        matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107, memory_ranges,
        error_counts);

    matrix_accumulator_150 = permute_signed(packed_half_153, packed_half_153, 0x5432u);

    matrix_fragment_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_139),
                            select_half_operand<2, false, false>(packed_half_152));

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_150 = half_to_float_bits((packed_half_146 & 65535u) * 65537u);

    matrix_fragment_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_137),
                            select_half_operand<2, false, false>(packed_half_152));

    packed_half_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(packed_half_152));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                     select_half_operand<2, false, false>(packed_half_155));

    packed_half_161 = half_to_float_bits((matrix_accumulator_109 & 65535u) * 65537u);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                     select_half_operand<2, false, false>(matrix_accumulator_110));

    float_bits_164 = half_to_float_bits((matrix_accumulator_109 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xaea0ull, matrix_accumulator_108,
        matrix_accumulator_109, matrix_accumulator_110, matrix_accumulator_111, memory_ranges,
        error_counts);

    matrix_accumulator_150 = rsqrt_bits(matrix_accumulator_150);

    matrix_accumulator_151 = rsqrt_bits(matrix_accumulator_151);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_147), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    packed_half_153 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_132 = rsqrt_bits(packed_half_161);

    matrix_accumulator_151 = pack_half(matrix_accumulator_151, matrix_accumulator_150);

    matrix_accumulator_150 = rsqrt_bits(float_bits_164);

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                     select_half_operand<2, false, false>(packed_half_160));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_149));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_accumulator_150 = pack_half(matrix_accumulator_150, matrix_accumulator_132);

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_accumulator_144));

    matrix_accumulator_123 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_150 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_accumulator_123 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                       select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_accumulator_121 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                            select_half_operand<2, false, false>(packed_half_160));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_155 = result_word;
    }

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_accumulator_151 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_138),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_136),
                      select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_fragment_136 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_137),
                                        select_half_operand<2, false, false>(0u));

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                    select_half_operand<2, false, false>(0u));

    packed_half_154 = pack_half(0u, packed_half_154);

    matrix_fragment_137 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_136 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_139),
                                        select_half_operand<2, false, false>(matrix_fragment_136));

    matrix_fragment_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_accumulator_145));

    matrix_fragment_138 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                      select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_143));

    packed_half_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_137 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                        select_half_operand<2, false, false>(matrix_fragment_137));

    matrix_accumulator_120 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_120));

    packed_half_146 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_fragment_139));

    matrix_fragment_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<0, false, false>(packed_half_154));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_accumulator_120));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                     select_half_operand<2, false, false>(packed_half_155));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_146), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_4 = permute_signed(matrix_accumulator_121, matrix_accumulator_121, 0x5432u);

    matrix_fragment_7 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_6 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                            select_half_operand<0, false, false>(packed_half_154));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_147), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_161 = result_word;
    }

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    packed_half_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_9),
                                        select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_121 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                           select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_8),
                                        select_half_operand<0, false, false>(packed_half_154));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_9),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_accumulator_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_123 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_8),
                                           select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_7 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    packed_half_8 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_7);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    packed_half_8 = rsqrt_bits(packed_half_8);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_148), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                     select_half_operand<2, false, false>(packed_half_161));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_146 = permute_signed(packed_half_147, packed_half_147, 0x5432u);

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                  select_half_operand<2, false, false>(0u));

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                     select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_132 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_7 = pack_half(packed_half_8, matrix_fragment_7);

    packed_half_8 = permute_signed(packed_half_155, packed_half_155, 0x5432u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_147), 0x4100410u);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(packed_half_8));

    packed_half_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                   select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = half_to_float_bits((packed_half_9 & 65535u) * 65537u);

    matrix_fragment_5 = half_to_float_bits((packed_half_9 >> 16) * 65537u);

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                     select_half_operand<2, false, false>(packed_half_160));

    packed_half_9 = permute_signed(packed_half_8, packed_half_8, 0x5432u);

    packed_half_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x4100410u);

    packed_half_8 = add_half_pairs(select_half_operand<2, false, false>(packed_half_8),
                                   select_half_operand<2, false, false>(packed_half_9));

    packed_half_9 = permute_signed(packed_half_148, packed_half_148, 0x5432u);

    matrix_fragment_6 = half_to_float_bits((packed_half_155 & 65535u) * 65537u);

    matrix_accumulator_134 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(0u));

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                     select_half_operand<2, false, false>(packed_half_9));

    packed_half_9 = half_to_float_bits((packed_half_155 >> 16) * 65537u);

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = rsqrt_bits(matrix_fragment_6);

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_accumulator_134 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(0u));

    packed_half_9 = rsqrt_bits(packed_half_9);

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(packed_half_146, packed_half_147, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_148 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_148), 0x4100410u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0x4100410u);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                             matrix_fragment_89);

    packed_half_8 = half_to_float_bits((matrix_fragment_85 & 65535u) * 65537u);

    packed_half_9 = pack_half(packed_half_9, matrix_fragment_6);

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    multiply_accumulate_half(packed_half_152, packed_half_153, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_86 = half_to_float_bits((matrix_fragment_85 >> 16) * 65537u);

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_7 = half_to_float_bits((packed_half_148 & 65535u) * 65537u);

    packed_half_148 = half_to_float_bits((packed_half_148 >> 16) * 65537u);

    matrix_accumulator_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(packed_half_9));

    packed_half_8 = rsqrt_bits(packed_half_8);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_9));

    matrix_accumulator_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_87 = rsqrt_bits(matrix_fragment_86);

    matrix_fragment_6 = rsqrt_bits(matrix_fragment_7);

    packed_half_9 = rsqrt_bits(packed_half_148);

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_half(matrix_fragment_5, matrix_fragment_4);

    packed_half_8 = pack_half(matrix_fragment_87, packed_half_8);

    matrix_fragment_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                      select_half_operand<2, false, false>(matrix_accumulator_125));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 = pack_half(packed_half_9, matrix_fragment_6);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_8));

    matrix_accumulator_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(packed_half_8));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(packed_half_82, matrix_fragment_83, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_136, matrix_fragment_137, packed_half_82,
                             matrix_fragment_83);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_141 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_8));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(packed_half_8));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_144),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_88 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                      select_half_operand<2, false, false>(matrix_accumulator_141));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(packed_half_8));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_24));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_92,
                             matrix_accumulator_93);

    packed_half_8 = 0x29c0u;

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_accumulator_120,
                             matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_82),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_146 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_146),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_152 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_152),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_136, matrix_fragment_137, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_102,
                             matrix_accumulator_103);

    packed_half_146 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_146), 0x3c203c20u);

    matrix_accumulator_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x3c203c20u);

    matrix_accumulator_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_96,
                             matrix_accumulator_97);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_138, matrix_fragment_139, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    packed_half_147 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_147),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_82), 0x3e473e47u);

    packed_half_153 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_153),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_146 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_146), 0x3e473e47u);

    packed_half_152 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x3e473e47u);

    matrix_accumulator_118 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3c203c20u);

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3c203c20u);

    matrix_fragment_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_accumulator_27 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_accumulator_118 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3e473e47u);

    matrix_accumulator_124 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3e473e47u);

    matrix_accumulator_80 = (packed_half_82 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_25 = (packed_half_146 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_28 = (packed_half_152 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_147 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_147), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_104,
                             matrix_accumulator_105);

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x3c203c20u);

    matrix_accumulator_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                       select_half_operand<2, false, false>(matrix_accumulator_28));

    packed_half_82 = (matrix_fragment_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    packed_half_147 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_147), 0x3e473e47u);

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_153 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x3e473e47u);

    matrix_accumulator_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    packed_half_9 = (matrix_accumulator_118 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_23 = (matrix_accumulator_124 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_108,
                             matrix_accumulator_109);

    matrix_accumulator_92 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3c203c20u);

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_94 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3c203c20u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_26 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_accumulator_29 = (packed_half_147 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_9),
                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_81 = (packed_half_153 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_accumulator_119 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3c203c20u);

    matrix_accumulator_125 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3c203c20u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3e473e47u);

    matrix_accumulator_94 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3e473e47u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3c203c20u);

    matrix_accumulator_120 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120), 0x3c203c20u);

    matrix_accumulator_97 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3c203c20u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_accumulator_121 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3c203c20u);

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_accumulator_119 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3e473e47u);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                       select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_125 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3e473e47u);

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3e473e47u);

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3e473e47u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_120 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120), 0x3e473e47u);

    matrix_fragment_98 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3e473e47u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_121 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3e473e47u);

    matrix_fragment_5 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = (matrix_accumulator_94 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_22 = (matrix_accumulator_119 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_24 = (matrix_accumulator_125 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_108 = (matrix_accumulator_120 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_120 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_96 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3c203c20u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_fragment_98 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_106 = (matrix_accumulator_106 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 = (matrix_accumulator_121 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                       select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3e473e47u);

    matrix_accumulator_120 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_122 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_97 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_99 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_107 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_accumulator_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_122 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_104 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x3c203c20u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_6),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_93 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3c203c20u);

    matrix_accumulator_121 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_95 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3c203c20u);

    matrix_accumulator_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x3c203c20u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_7),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_104 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x3e473e47u);

    matrix_accumulator_121 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                       select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3c203c20u);

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x3c203c20u);

    matrix_accumulator_93 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3e473e47u);

    matrix_accumulator_95 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3e473e47u);

    matrix_accumulator_105 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x3e473e47u);

    matrix_accumulator_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_6 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3e473e47u);

    matrix_fragment_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x3e473e47u);

    matrix_fragment_83 = (matrix_accumulator_104 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_94 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_accumulator_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_accumulator_111 = (matrix_accumulator_93 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_118 = (matrix_accumulator_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = (matrix_accumulator_105 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_92 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_93 = (matrix_fragment_6 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_95 = (matrix_fragment_7 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_91));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_accumulator_94 = (matrix_accumulator_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                       select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_96 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_accumulator_96));

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_fragment_48 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_48 = address_word_156 * 0x4u;

    condition_0 = (address_word_156 & 0x1u) != 0;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    condition_1 = (address_word_156 & 0x2u) != 0;

    matrix_fragment_49 = matrix_fragment_48 & 0x1cu;

    matrix_fragment_48 = condition_0 ? matrix_accumulator_119 : matrix_accumulator_120;

    matrix_accumulator_121 = condition_0 ? matrix_fragment_6 : matrix_fragment_7;

    matrix_accumulator_119 = condition_0 ? matrix_accumulator_120 : matrix_accumulator_119;

    matrix_fragment_6 = condition_0 ? matrix_fragment_7 : matrix_fragment_6;

    matrix_fragment_49 =
        uint32_t(int64_t(int32_t(address_word_156)) * (1ll << 0x1du) >> 32) + matrix_fragment_49;

    matrix_accumulator_120 = !condition_1 ? matrix_fragment_48 : matrix_accumulator_121;

    matrix_accumulator_122 = !condition_1 ? matrix_accumulator_121 : matrix_fragment_48;

    matrix_accumulator_121 = !condition_1 ? matrix_accumulator_119 : matrix_fragment_6;

    matrix_accumulator_119 = !condition_1 ? matrix_fragment_6 : matrix_accumulator_119;

    matrix_fragment_48 = 0x1u ^ matrix_fragment_49;

    matrix_fragment_7 = 0x2u ^ matrix_fragment_49;

    matrix_fragment_6 = 0x3u ^ matrix_fragment_49;

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_49), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_120 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_121), "r"(matrix_fragment_48), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_121 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_122 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_119 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    condition_2 = (address_word_156 & 0x8u) != 0;

    condition_3 = (address_word_156 & 0x10u) != 0;

    matrix_accumulator_123 = !condition_2 ? matrix_accumulator_120 : matrix_accumulator_121;

    matrix_accumulator_124 = !condition_2 ? matrix_accumulator_121 : matrix_accumulator_120;

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             packed_half_60, matrix_fragment_61, matrix_accumulator_150,
                             matrix_accumulator_151);

    matrix_accumulator_126 = !condition_2 ? matrix_accumulator_122 : matrix_accumulator_119;

    matrix_accumulator_125 = !condition_2 ? matrix_accumulator_119 : matrix_accumulator_122;

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             packed_half_60, matrix_fragment_61, matrix_fragment_44,
                             matrix_fragment_45);

    packed_half_60 = !condition_3 ? matrix_accumulator_123 : matrix_accumulator_126;

    matrix_fragment_61 = !condition_3 ? matrix_accumulator_124 : matrix_accumulator_125;

    matrix_accumulator_123 = !condition_3 ? matrix_accumulator_126 : matrix_accumulator_123;

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_60 = add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                    select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_124 = !condition_3 ? matrix_accumulator_125 : matrix_accumulator_124;

    packed_half_60 = add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                    select_half_operand<2, false, false>(matrix_accumulator_123));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_60 = add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                    select_half_operand<2, false, false>(matrix_accumulator_124));

    packed_half_60 = add_half_pairs(select_half_operand<0, false, false>(packed_half_60),
                                    select_half_operand<1, false, false>(packed_half_60));

    packed_half_60 = permute_signed(packed_half_60, packed_half_60, 0x5410u);

    packed_half_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_61 = half_to_float_bits((packed_half_60 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_61 = reciprocal_bits(matrix_fragment_61);

    matrix_fragment_62 = half_to_float_bits((packed_half_60 >> 16) * 65537u);

    packed_half_60 = address_word_162 + 0x8u;

    matrix_fragment_62 = reciprocal_bits(matrix_fragment_62);

    matrix_accumulator_119 = uint32_t(int32_t(packed_half_60) >> 0x1fu);

    matrix_fragment_63 = uint32_t(int32_t(index_word_163) >> 0x1fu);

    matrix_accumulator_120 =
        uint32_t(combine_words(matrix_accumulator_119, 0u) << 0x5u >> 32) + packed_half_60;

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_63 =
        uint32_t(combine_words(matrix_fragment_63, 0u) << 0x5u >> 32) + address_word_162;

    packed_half_113 = transpose_half_tile(packed_half_113);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_69 = matrix_accumulator_120 & 0xffffffe0u;

    matrix_accumulator_120 = pack_half(matrix_fragment_62, matrix_fragment_61);

    matrix_fragment_61 = matrix_fragment_63 & 0xffffffe0u;

    packed_half_112 = transpose_half_tile(packed_half_112);

    matrix_fragment_63 = packed_half_60 + (0u - matrix_fragment_69);

    matrix_fragment_61 = address_word_162 + (0u - matrix_fragment_61);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_61), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_63), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    matrix_fragment_62 = transpose_half_tile(matrix_fragment_15);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(packed_half_113));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    packed_half_112 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5410u);

    packed_half_113 = permute_signed(matrix_fragment_63, matrix_fragment_63, 0x5410u);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));

    packed_half_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_82),
                                         select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_62 = transpose_half_tile(matrix_fragment_30);

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = transpose_half_tile(matrix_fragment_17);

    matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_82),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_25 =
        uint32_t(int64_t(int32_t(index_word_163)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_16);

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_accumulator_26 = uint32_t(int32_t(matrix_accumulator_25) >> 0x1fu);

    matrix_fragment_63 = transpose_half_tile(matrix_fragment_31);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_104,
                             matrix_accumulator_105);

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(matrix_accumulator_80));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_71 = uint32_t(int64_t(int32_t(index_word_163)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_70 =
        uint32_t(combine_words(matrix_accumulator_26, 0u) << 0x5u >> 32) + matrix_accumulator_25;

    matrix_accumulator_80 = uint32_t(int32_t(matrix_fragment_71) >> 0x1fu);

    matrix_fragment_70 = matrix_fragment_70 & 0xffffffe0u;

    matrix_accumulator_80 =
        uint32_t(combine_words(matrix_accumulator_80, 0u) << 0x5u >> 32) + matrix_fragment_71;

    matrix_accumulator_25 = matrix_accumulator_25 + (0u - matrix_fragment_70);

    matrix_fragment_70 = matrix_accumulator_80 & 0xffffffe0u;

    matrix_accumulator_115 = transpose_half_tile(matrix_accumulator_115);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_121 = matrix_fragment_71 + (0u - matrix_fragment_70);

    matrix_accumulator_114 = transpose_half_tile(matrix_accumulator_114);

    matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_21);

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                      select_half_operand<2, false, false>(matrix_accumulator_27));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_accumulator_121), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_121 = result_word;
    }

    matrix_fragment_62 = transpose_half_tile(matrix_fragment_20);

    matrix_fragment_117 = transpose_half_tile(matrix_fragment_117);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_accumulator_25), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_25 = result_word;
    }

    matrix_fragment_63 = transpose_half_tile(matrix_fragment_116);

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_19);

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_18);

    matrix_fragment_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 =
        permute_signed(matrix_accumulator_121, matrix_accumulator_121, 0x5410u);

    matrix_accumulator_3 = transpose_half_tile(matrix_accumulator_3);

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_117),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    packed_half_82 = permute_signed(matrix_accumulator_25, matrix_accumulator_25, 0x5410u);

    matrix_accumulator_2 = transpose_half_tile(matrix_accumulator_2);

    matrix_fragment_98 = transpose_half_tile(matrix_fragment_47);

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_97 = transpose_half_tile(matrix_fragment_46);

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                      select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                      select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_151 = transpose_half_tile(matrix_accumulator_151);

    matrix_accumulator_110 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_26, matrix_accumulator_27, 0u, 0u);

    matrix_accumulator_101 = transpose_half_tile(matrix_accumulator_101);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_103 = transpose_half_tile(matrix_accumulator_103);

    matrix_accumulator_105 = transpose_half_tile(matrix_accumulator_105);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_24 = transpose_half_tile(matrix_accumulator_150);

    matrix_accumulator_25 = transpose_half_tile(matrix_fragment_44);

    matrix_accumulator_100 = transpose_half_tile(matrix_accumulator_100);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_accumulator_102 = transpose_half_tile(matrix_accumulator_102);

    matrix_fragment_98 = transpose_half_tile(matrix_accumulator_104);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_26, matrix_accumulator_27, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = transpose_half_tile(matrix_fragment_51);

    matrix_fragment_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                      select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_99 = transpose_half_tile(matrix_fragment_50);

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(packed_half_113));

    packed_half_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                                       select_half_operand<2, false, false>(0u));

    packed_half_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_9),
                                        select_half_operand<2, false, false>(packed_half_112));

    matrix_accumulator_151 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_151),
                      select_half_operand<2, false, false>(0u));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_9),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_50 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_51 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_98 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(packed_half_82));

    packed_half_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(packed_half_82));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_accumulator_80,
                             matrix_accumulator_81);

    packed_half_9 = thread_row_165 * 0x20u;

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_70,
                             matrix_fragment_71);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_fragment_4 = uint32_t(uint32_t(packed_half_9) >> 0x1u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_3 = index_word_163 & 0x7ffffffcu;

    matrix_accumulator_2 = matrix_fragment_4 + 0x4u;

    multiply_accumulate_half(packed_half_82, matrix_fragment_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_fragment_5 = thread_row_165 * 0x100u;

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_30 = 0u - matrix_accumulator_3 + address_word_156;

    matrix_accumulator_3 = matrix_fragment_4 + 0x8u;

    matrix_fragment_4 = matrix_fragment_4 + 0xcu;

    matrix_fragment_30 = matrix_fragment_30 * 0x2u + packed_half_9;

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_accumulator_93 = matrix_accumulator_3 + address_word_156 + (0u - index_word_157);

    matrix_accumulator_95 = matrix_fragment_4 + address_word_156 + (0u - index_word_157);

    matrix_fragment_19 = matrix_accumulator_2 + address_word_156 + (0u - index_word_157);

    matrix_fragment_17 = uint32_t(int32_t(matrix_fragment_30) >> 0x1u);

    split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                        uint64_t(uint32_t(matrix_accumulator_93)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                        uint64_t(uint32_t(matrix_accumulator_95)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_18, matrix_fragment_19) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_92, matrix_accumulator_93) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(address_word_158, address_word_159,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_16, matrix_fragment_17) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_104 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_5 = (thread_row_165 << 0x9u) + uniform_index_word_8;

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_5 = address_word_156 * 0x10u + matrix_fragment_5;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_5, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_5, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_5 + 0x400u, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_5 + 0x400u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_90, matrix_fragment_91,
                        int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_158, address_word_159));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe0b0ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe2b0ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe8b0ull, matrix_accumulator_80,
        matrix_accumulator_81, packed_half_82, matrix_fragment_83, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xeab0ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_36, matrix_accumulator_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_36, matrix_accumulator_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_36,
                             matrix_accumulator_37);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_fragment_16);

    matrix_accumulator_37 = unpack_fp8_bits(matrix_fragment_17);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_53 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_52 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_36,
                             matrix_accumulator_37);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0x400u, matrix_accumulator_36, matrix_accumulator_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    matrix_fragment_16 = unpack_fp8_bits(matrix_accumulator_28);

    matrix_fragment_17 = unpack_fp8_bits(matrix_accumulator_29);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_accumulator_28 = unpack_fp8_bits(matrix_accumulator_28 >> 16);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x600u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_accumulator_29 >> 16);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_16,
                             matrix_fragment_17);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_accumulator_28 = unpack_fp8_bits(matrix_fragment_30);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_fragment_31);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_30 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_31 >> 16);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_28,
                             matrix_accumulator_29);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_38 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_39 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_82, matrix_fragment_83, matrix_fragment_58,
                             matrix_fragment_59);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_45 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    multiply_accumulate_half(packed_half_82, matrix_fragment_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             packed_half_82, matrix_fragment_83, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_44 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                      select_half_operand<2, false, false>(matrix_fragment_75),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_38 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_54,
                             matrix_fragment_55);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_45), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_52,
                             matrix_fragment_53);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_39), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);

    matrix_fragment_16 = parameters[parameter_output_low_2];
    matrix_fragment_17 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_38), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    uniform_index_word_9 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_index_word_10;

    uniform_index_word_8 = uint32_t(int32_t(uniform_parameter_word_11) >> 0x1fu);

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_44));

    uniform_index_word_8 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_parameter_word_11;

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_68,
                             matrix_fragment_69);

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_9) >> 0x2u);

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_8) >> 0x2u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_accumulator_36,
                             matrix_accumulator_37);

    uniform_condition_2 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_8);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_94 = result_word;
    }

    uniform_condition_0 = int32_t(uniform_address_word_7) >= int32_t(uniform_index_word_9);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    uniform_index_word_10 = uniform_index_word_6 * uniform_index_word_9 + uniform_address_word_7;

    uniform_condition_2 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x4u) && !uniform_condition_2;

    uniform_condition_0 =
        int32_t(uniform_index_word_14) > int32_t(0u - 0x4u) && !uniform_condition_0;

    packed_half_0 = thread_row_165 * 0x80u;

    uniform_condition_1 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_45 = uniform_index_word_10;

    matrix_fragment_39 = matrix_fragment_45 * 0x100u + packed_half_0;

    split_address_words(matrix_fragment_32, matrix_fragment_33,
                        int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_16, matrix_fragment_17));
    if (uniform_condition_1) {
        matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_30 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                          select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_fragment_17 = matrix_fragment_30;

        matrix_fragment_19 = matrix_fragment_55;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_28, matrix_accumulator_29), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }
    uniform_index_word_10 = uniform_address_word_7 + 0x1u;

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(0u));

    uniform_condition_1 = int32_t(uniform_index_word_10) >= int32_t(uniform_index_word_9);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_1 =
        int32_t(uniform_index_word_14) > int32_t(0u - 0x8u) && !uniform_condition_1;

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_93 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                      select_half_operand<2, false, false>(0u));

    uniform_condition_2 = uniform_condition_2 & uniform_condition_1;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_2) {
        matrix_accumulator_93 = permute_signed(matrix_accumulator_93, matrix_fragment_18, 0x5410u);

        split_address_words(matrix_fragment_32, matrix_fragment_33,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));

        matrix_accumulator_81 = permute_signed(matrix_accumulator_81, matrix_fragment_16, 0x5410u);

        matrix_fragment_18 = permute_signed(matrix_accumulator_92, matrix_fragment_19, 0x5410u);

        matrix_fragment_16 = permute_signed(matrix_accumulator_80, matrix_fragment_17, 0x5410u);

        matrix_fragment_17 = matrix_accumulator_81;

        matrix_fragment_19 = matrix_accumulator_93;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_32, matrix_fragment_33) + 0x400ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xb6a0ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xb0a0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xb2a0ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xb4a0ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xb8a0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xbaa0ull, matrix_accumulator_36,
        matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xbca0ull, matrix_accumulator_28,
        matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xbea0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_78 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_12 = permute_signed(matrix_fragment_79, matrix_fragment_79, 0x5432u);

    matrix_fragment_13 = permute_signed(matrix_fragment_78, matrix_fragment_78, 0x5432u);

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x4100410u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x4100410u);

    matrix_accumulator_80 = half_to_float_bits((matrix_fragment_79 & 65535u) * 65537u);

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_78 & 65535u) * 65537u);

    matrix_fragment_79 = half_to_float_bits((matrix_fragment_79 >> 16) * 65537u);

    matrix_fragment_78 = half_to_float_bits((matrix_fragment_78 >> 16) * 65537u);

    matrix_accumulator_80 = rsqrt_bits(matrix_accumulator_80);

    matrix_fragment_12 = rsqrt_bits(matrix_fragment_12);

    matrix_fragment_13 = rsqrt_bits(matrix_fragment_78);

    matrix_fragment_79 = rsqrt_bits(matrix_fragment_79);

    matrix_fragment_13 = pack_half(matrix_fragment_13, matrix_fragment_12);

    matrix_accumulator_81 = pack_half(matrix_fragment_79, matrix_accumulator_80);

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = permute_signed(matrix_accumulator_97, matrix_fragment_12, 0x5410u);

    matrix_fragment_40 = permute_signed(matrix_accumulator_96, matrix_fragment_13, 0x5410u);

    matrix_fragment_12 = permute_signed(matrix_fragment_69, matrix_fragment_69, 0x5432u);

    matrix_fragment_13 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5432u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x4100410u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x4100410u);

    matrix_fragment_70 = half_to_float_bits((matrix_fragment_69 & 65535u) * 65537u);

    matrix_fragment_69 = half_to_float_bits((matrix_fragment_69 >> 16) * 65537u);

    matrix_fragment_71 = half_to_float_bits((matrix_fragment_68 & 65535u) * 65537u);

    matrix_fragment_68 = half_to_float_bits((matrix_fragment_68 >> 16) * 65537u);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_70 = rsqrt_bits(matrix_fragment_70);

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_69 = rsqrt_bits(matrix_fragment_69);

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = rsqrt_bits(matrix_fragment_71);

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = rsqrt_bits(matrix_fragment_68);

    matrix_fragment_42 = permute_signed(matrix_fragment_42, matrix_fragment_79, 0x5410u);

    matrix_fragment_43 = permute_signed(matrix_fragment_43, matrix_fragment_72, 0x5410u);

    matrix_fragment_70 = pack_half(matrix_fragment_69, matrix_fragment_70);

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_16 = pack_half(matrix_fragment_68, matrix_fragment_71);

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(packed_half_154));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(packed_half_154));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_136, matrix_fragment_137, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_16 = permute_signed(matrix_fragment_16, matrix_fragment_17, 0x5410u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_138, matrix_fragment_139, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = permute_signed(matrix_fragment_75, matrix_fragment_18, 0x5410u);

    matrix_fragment_18 = permute_signed(matrix_accumulator_102, matrix_fragment_19, 0x5410u);

    matrix_fragment_19 = permute_signed(matrix_accumulator_103, matrix_fragment_42, 0x5410u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_136, matrix_fragment_137, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_38,
                             matrix_fragment_39);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_138, matrix_fragment_139, matrix_accumulator_36,
                             matrix_accumulator_37);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_35),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3c203c20u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3c203c20u);

    matrix_accumulator_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3c203c20u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3c203c20u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3e473e47u);

    matrix_fragment_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3e473e47u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_accumulator_36 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    matrix_fragment_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3e473e47u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3e473e47u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_fragment_35 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3e473e47u);

    matrix_fragment_34 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_38 = (matrix_fragment_32 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = (matrix_fragment_10 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_36 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_59 = (matrix_fragment_11 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_42 = (matrix_fragment_33 << 0x5u) + 0x7ff88000u;

    matrix_fragment_10 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_11 = (matrix_fragment_54 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_32 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_33 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_accumulator_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_fragment_54 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_37 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3c203c20u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_accumulator_28 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28), 0x3c203c20u);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3c203c20u);

    matrix_fragment_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3e473e47u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_fragment_17 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28), 0x3e473e47u);

    matrix_accumulator_28 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_37 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3e473e47u);

    matrix_fragment_46 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = (matrix_fragment_16 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_accumulator_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_30 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3c203c20u);

    matrix_fragment_45 = (matrix_fragment_18 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_39 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_13),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3e473e47u);

    matrix_fragment_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_29 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3c203c20u);

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3c203c20u);

    packed_half_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_8), 0x3d343d34u);

    matrix_fragment_44 = (matrix_fragment_17 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3c203c20u);

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3c203c20u);

    matrix_accumulator_29 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3e473e47u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3e473e47u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0x3c203c20u);

    matrix_fragment_47 = (matrix_fragment_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3c203c20u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_55 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3e473e47u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3e473e47u);

    matrix_fragment_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0x3e473e47u);

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3e473e47u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    packed_half_8 = (matrix_accumulator_29 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = (matrix_fragment_31 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_56 = (matrix_fragment_16 << 0x5u) + 0x7ff88000u;

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_54 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(packed_half_8),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_41 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_fragment_55 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_29 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = (matrix_fragment_17 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_18 = condition_0 ? matrix_fragment_17 : matrix_fragment_16;

    matrix_fragment_16 = condition_0 ? matrix_fragment_16 : matrix_fragment_17;

    matrix_fragment_17 = condition_0 ? matrix_fragment_65 : matrix_fragment_58;

    matrix_fragment_65 = condition_0 ? matrix_fragment_58 : matrix_fragment_65;

    matrix_fragment_58 = !condition_1 ? matrix_fragment_18 : matrix_fragment_17;

    matrix_fragment_19 = !condition_1 ? matrix_fragment_16 : matrix_fragment_65;

    matrix_fragment_18 = !condition_1 ? matrix_fragment_17 : matrix_fragment_18;

    matrix_fragment_65 = !condition_1 ? matrix_fragment_65 : matrix_fragment_16;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_58), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_19), "r"(matrix_fragment_48), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_18), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_65 = result_word;
    }

    matrix_fragment_16 = !condition_2 ? matrix_fragment_58 : matrix_fragment_19;

    matrix_fragment_17 = !condition_2 ? matrix_fragment_19 : matrix_fragment_58;

    matrix_fragment_67 = !condition_2 ? matrix_fragment_18 : matrix_fragment_65;

    matrix_fragment_64 = !condition_2 ? matrix_fragment_65 : matrix_fragment_18;

    matrix_fragment_66 = !condition_3 ? matrix_fragment_16 : matrix_fragment_67;

    matrix_fragment_69 = !condition_3 ? matrix_fragment_17 : matrix_fragment_64;

    matrix_fragment_16 = !condition_3 ? matrix_fragment_67 : matrix_fragment_16;

    matrix_fragment_17 = !condition_3 ? matrix_fragment_64 : matrix_fragment_17;

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_66 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_66),
                                        select_half_operand<1, false, false>(matrix_fragment_66));

    matrix_fragment_66 = permute_signed(matrix_fragment_66, matrix_fragment_66, 0x5410u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x4100410u);

    matrix_fragment_6 = half_to_float_bits((matrix_fragment_66 & 65535u) * 65537u);

    matrix_fragment_66 = half_to_float_bits((matrix_fragment_66 >> 16) * 65537u);

    matrix_fragment_6 = reciprocal_bits(matrix_fragment_6);

    matrix_fragment_7 = reciprocal_bits(matrix_fragment_66);

    matrix_accumulator_119 =
        uint32_t(combine_words(matrix_accumulator_119, 0u) << 0x5u >> 32) + packed_half_60;

    matrix_accumulator_119 = matrix_accumulator_119 & 0xffffffe0u;

    matrix_fragment_17 = uint32_t(int64_t(int32_t(index_word_163)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_18 = uint32_t(int64_t(int32_t(index_word_163)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_18) >> 0x1fu);

    matrix_fragment_16 = pack_half(matrix_fragment_7, matrix_fragment_6);

    matrix_fragment_7 = packed_half_60 + (0u - matrix_accumulator_119);

    matrix_fragment_6 = uint32_t(int32_t(matrix_fragment_17) >> 0x1fu);

    matrix_fragment_19 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x5u >> 32) + matrix_fragment_18;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_61), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_61 = result_word;
    }

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_fragment_6, 0u) << 0x5u >> 32) + matrix_fragment_17;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    matrix_fragment_6 = matrix_fragment_6 & 0xffffffe0u;

    matrix_fragment_19 = matrix_fragment_19 & 0xffffffe0u;

    matrix_fragment_49 = matrix_fragment_17 + (0u - matrix_fragment_6);

    matrix_fragment_65 = matrix_fragment_18 + (0u - matrix_fragment_19);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        packed_half_60 = result_word;
    }

    matrix_fragment_61 = permute_signed(matrix_fragment_61, matrix_fragment_61, 0x5410u);

    matrix_fragment_58 = permute_signed(matrix_fragment_7, matrix_fragment_7, 0x5410u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_36 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_36 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5410u);

    packed_half_60 = permute_signed(packed_half_60, packed_half_60, 0x5410u);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_60));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_26, matrix_accumulator_27, 0u, 0u);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_37 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_26, matrix_accumulator_27, 0u, 0u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    packed_half_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_8),
                                        select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_8),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_48,
                             matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_7 = index_word_163 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_14,
                             matrix_fragment_15);

    packed_half_8 = 0u - matrix_fragment_7 + address_word_156;

    matrix_fragment_6 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_7 = parameters[parameter_trained_weights_high_5];

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_62,
                             matrix_fragment_63);

    index_word_163 = index_word_163 & 0xfffffffcu;

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_accumulator_29 = matrix_accumulator_2 + address_word_156 + (0u - index_word_163);

    packed_half_8 = packed_half_8 * 0x2u + packed_half_9;

    matrix_fragment_31 = matrix_accumulator_3 + address_word_156 + (0u - index_word_163);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                         select_half_operand<2, false, false>(0u));

    index_word_163 = matrix_fragment_4 + address_word_156 + (0u - index_word_163);

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = uint32_t(int32_t(packed_half_8) >> 0x1u);

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                  select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                        uint64_t(uint32_t(matrix_accumulator_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_5 + 0x800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    split_address_words(matrix_fragment_30, matrix_fragment_31,
                        uint64_t(uint32_t(matrix_fragment_31)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_5, matrix_fragment_48,
                                         matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(packed_half_9));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_37 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_28, matrix_accumulator_29) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(index_word_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    matrix_accumulator_36 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0xf0b0ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_5 + 0x400u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_30, matrix_fragment_31) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0xf0b0ull, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_5 + 0xc00u, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe0b0ull, matrix_accumulator_24,
        matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe2b0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe8b0ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xeab0ull, packed_half_8,
        packed_half_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    uniform_parameter_word_11 = 0u;

    uniform_index_word_10 = 0x400u;

    uniform_index_word_10 = (uniform_parameter_word_11 << 0x18u) + uniform_index_word_10;

    matrix_fragment_48 = (address_word_156 << 0x4u) + uniform_index_word_10;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_48, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_32);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_48 + 0x400u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_48 + 0x600u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    uniform_index_word_9 = uniform_index_word_6 * uniform_index_word_9 + uniform_index_word_9;

    uniform_index_word_6 = uniform_index_word_6 + 0x1u;

    uniform_condition_2 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_8);

    uniform_index_word_9 = uniform_address_word_7 + uniform_index_word_9;

    uniform_condition_2 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x8u) && !uniform_condition_2;

    uniform_condition_0 = uniform_condition_2 & uniform_condition_0;

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_35);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_20);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_21);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_22);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_23);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_48 + 0x200u, matrix_accumulator_28, matrix_accumulator_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_20,
                             matrix_fragment_21);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_4 = threadIdx.x;

    matrix_fragment_5 = threadIdx.z;

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             packed_half_8, packed_half_9, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_40,
                             matrix_fragment_41);

    condition_0 = (matrix_fragment_4 | matrix_fragment_5 | thread_row_165) != 0;

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_20,
                             matrix_fragment_21);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_22,
                             matrix_fragment_23);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_8, packed_half_9, matrix_fragment_42, matrix_fragment_43);

    packed_half_9 = uniform_index_word_9;

    matrix_fragment_5 = packed_half_9 * 0x100u + packed_half_0;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));
    if (uniform_condition_0) {
        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

        packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                           select_half_operand<2, false, false>(matrix_fragment_4));

        packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                           select_half_operand<2, false, false>(packed_half_9));

        packed_half_9 = packed_half_0;

        store_four_words_streaming<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                                packed_half_8, packed_half_9, matrix_fragment_10,
                                                matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_1 = uniform_condition_2 & uniform_condition_1;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_1) {
        matrix_fragment_23 = permute_signed(matrix_fragment_4, matrix_fragment_23, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_156)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        packed_half_0 = permute_signed(matrix_accumulator_25, packed_half_0, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_accumulator_24, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = packed_half_0;

        matrix_fragment_7 = matrix_fragment_23;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x400ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_index_word_6 = parameters[parameter_publication_optional_low_16];
    uniform_address_word_7 = parameters[parameter_publication_optional_high_17];

    uniform_index_word_5 = uniform_block_index_15 + uniform_parameter_word_4;

    split_address_words(uniform_index_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_index_word_6, uniform_address_word_7));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_6, uniform_address_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__ __launch_bounds__(64, 7) void phoenix_window_attention_channels_64_input_view(
    KernelArguments parameters) {
    run_window_attention_channels_64_input_view<false>(parameters.words, nullptr, nullptr);
}
