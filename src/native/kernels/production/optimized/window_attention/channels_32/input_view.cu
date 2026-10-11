// DLSS5 Phoenix optimized kernel: window_attention/channels_32/input_view.cu
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
    parameter_optional_control_18 = 18,
    parameter_optional_control_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[24];
};
static_assert(sizeof(KernelArguments) == 96, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_32_input_view(
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
    uint32_t packed_half_34 = 0;
    uint32_t packed_half_35 = 0;
    uint32_t packed_half_36 = 0;
    uint32_t packed_half_37 = 0;
    uint32_t matrix_accumulator_38 = 0;
    uint32_t matrix_accumulator_39 = 0;
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
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
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
    uint32_t matrix_fragment_96 = 0;
    uint32_t matrix_fragment_97 = 0;
    uint32_t packed_half_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t packed_half_100 = 0;
    uint32_t packed_half_101 = 0;
    uint32_t packed_half_102 = 0;
    uint32_t packed_half_103 = 0;
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
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
    uint32_t packed_half_116 = 0;
    uint32_t packed_half_117 = 0;
    uint32_t packed_half_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t packed_half_120 = 0;
    uint32_t packed_half_121 = 0;
    uint32_t matrix_accumulator_122 = 0;
    uint32_t matrix_accumulator_123 = 0;
    uint32_t matrix_accumulator_124 = 0;
    uint32_t matrix_accumulator_125 = 0;
    uint32_t matrix_accumulator_126 = 0;
    uint32_t matrix_accumulator_127 = 0;
    uint32_t packed_half_128 = 0;
    uint32_t packed_half_129 = 0;
    uint32_t packed_half_130 = 0;
    uint32_t packed_half_131 = 0;
    uint32_t matrix_fragment_132 = 0;
    uint32_t matrix_fragment_133 = 0;
    uint32_t matrix_fragment_134 = 0;
    uint32_t matrix_fragment_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t matrix_accumulator_146 = 0;
    uint32_t matrix_accumulator_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t matrix_fragment_150 = 0;
    uint32_t matrix_fragment_151 = 0;
    uint32_t matrix_accumulator_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t packed_half_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t float_bits_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_reused_word_14 = 0;
    uint32_t uniform_reused_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_block_index_17 = 0;
    uint32_t uniform_block_index_18 = 0;
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

    packed_half_0 = threadIdx.x & 31u;

    matrix_fragment_6 = parameters[parameter_optional_control_18];
    matrix_fragment_7 = parameters[parameter_optional_control_19];

    uniform_index_word_8 = parameters[parameter_optional_control_9];

    condition_4 = false;

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffeffffu;

    uniform_index_word_13 = parameters[parameter_tensor_height_6];

    condition_5 = false;

    uniform_block_index_18 = blockIdx.y;

    uniform_index_word_4 = parameters[parameter_optional_control_8];

    uniform_index_word_12 = parameters[parameter_tensor_width_7];

    uniform_block_index_17 = blockIdx.x;

    matrix_fragment_9 = 0u;

    uniform_reused_word_14 = 0u;
    uniform_reused_word_15 = 0u;

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_29 = 0u;

    uniform_address_word_6 = parameters[parameter_output_low_2];
    uniform_address_word_7 = parameters[parameter_output_high_3];

    condition_0 = int32_t(matrix_fragment_6) > int32_t(0u);

    condition_1 = int32_t(matrix_fragment_7) > int32_t(0u);

    packed_half_120 = parameters[parameter_trained_weights_low_4];
    packed_half_121 = parameters[parameter_trained_weights_high_5];

    packed_half_2 = condition_0 ? matrix_fragment_6 : uniform_index_word_13;

    matrix_fragment_6 = 0u;

    matrix_fragment_5 = packed_half_0;

    uniform_index_word_8 = (uniform_block_index_18 << 0x3u) + uniform_index_word_8;

    packed_half_3 = packed_half_0;

    condition_0 = uint32_t(packed_half_2) != uint32_t(0x1u);

    matrix_fragment_10 = condition_1 ? matrix_fragment_7 : uniform_index_word_12;

    matrix_fragment_4 = uint32_t(int32_t(packed_half_3) >> 0x2u);

    matrix_fragment_8 =
        uint32_t(int64_t(int32_t(matrix_fragment_5)) * (1ll << 0x1cu) >> 32) + uniform_index_word_8;

    uniform_index_word_16 = (uniform_block_index_17 << 0x3u) + uniform_index_word_4;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x2u >> 32) + matrix_fragment_4;

    matrix_fragment_11 = matrix_fragment_10 * 0x4u;

    condition_1 = true;

    condition_2 = int32_t(matrix_fragment_8) >= int32_t(packed_half_2);

    matrix_fragment_5 = matrix_fragment_5 & 0xfffffffcu;

    matrix_fragment_16 = matrix_fragment_8 + 0x2u;

    matrix_fragment_18 = matrix_fragment_8 + 0x4u;

    condition_2 = int32_t(matrix_fragment_8) < int32_t(0u) || condition_2;

    matrix_fragment_17 = matrix_fragment_4 + uniform_index_word_16 + (0u - matrix_fragment_5);

    condition_3 = int32_t(matrix_fragment_16) >= int32_t(packed_half_2);

    matrix_fragment_20 = matrix_fragment_8 + 0x6u;

    if (condition_0) {
        condition_4 = condition_2;
    }

    condition_6 = int32_t(matrix_fragment_17) >= int32_t(matrix_fragment_10);

    if (condition_0) {
        condition_1 = !condition_2;
    }

    condition_3 = int32_t(matrix_fragment_16) < int32_t(0u) || condition_3;

    matrix_fragment_19 = matrix_fragment_17 + 0x4u;

    condition_6 = int32_t(matrix_fragment_17) > -1 && !condition_6;

    matrix_fragment_5 = uint32_t(condition_0);

    if (condition_0) {
        condition_5 = condition_3;
    }

    packed_half_3 = packed_half_3 & 0xfffffffcu;

    if (condition_4) {
        matrix_accumulator_33 = 0x10000u | matrix_accumulator_33;
    }

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffdffu;

    condition_1 = condition_1 & (condition_4 | condition_6);

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_5;

    if (condition_1) {
        matrix_accumulator_33 = 0x200u | matrix_accumulator_33;
    }

    condition_1 = true;

    if (condition_0) {
        condition_1 = !condition_3;
    }

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffbffffu;

    condition_1 = condition_1 & (condition_4 | condition_6);

    condition_4 = int32_t(matrix_fragment_18) >= int32_t(packed_half_2);

    if (condition_5) {
        matrix_accumulator_33 = 0x40000u | matrix_accumulator_33;
    }

    condition_5 = false;

    condition_4 = int32_t(matrix_fragment_18) < int32_t(0u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffff7fu;

    if (condition_0) {
        condition_5 = condition_4;
    }

    if (condition_1) {
        matrix_accumulator_33 = 0x80u | matrix_accumulator_33;
    }

    condition_1 = true;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfff7ffffu;

    if (condition_0) {
        condition_1 = !condition_4;
    }

    matrix_fragment_4 = uint32_t(condition_4) << 4;

    if (condition_5) {
        matrix_accumulator_33 = 0x80000u | matrix_accumulator_33;
    }

    condition_5 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_5;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffffefu;

    condition_4 = condition_1 & (condition_5 | condition_6);

    condition_1 = int32_t(matrix_fragment_20) >= int32_t(packed_half_2);

    condition_5 = int32_t(matrix_fragment_20) < int32_t(0u) || condition_1;

    condition_1 = true;

    if (condition_0) {
        condition_1 = !condition_5;
    }

    if (condition_4) {
        matrix_accumulator_33 = 0x10u | matrix_accumulator_33;
    }

    condition_4 = false;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffeffu;

    if (condition_0) {
        condition_4 = condition_5;
    }

    if (condition_5) {
        matrix_accumulator_33 = 0x100u | matrix_accumulator_33;
    }

    condition_5 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffdffffu;

    condition_1 = condition_1 & (condition_5 | condition_6);

    condition_5 = true;

    if (condition_4) {
        matrix_accumulator_33 = 0x20000u | matrix_accumulator_33;
    }

    condition_4 = false;

    if (condition_0) {
        condition_4 = condition_2;
    }

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffbffu;

    if (condition_0) {
        condition_5 = !condition_2;
    }

    if (condition_1) {
        matrix_accumulator_33 = 0x400u | matrix_accumulator_33;
    }

    condition_1 = int32_t(matrix_fragment_19) >= int32_t(matrix_fragment_10);

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffff7fffu;

    condition_1 = int32_t(matrix_fragment_19) > -1 && !condition_1;

    if (condition_4) {
        matrix_accumulator_33 = 0x8000u | matrix_accumulator_33;
    }

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffffbfu;

    condition_5 = condition_5 & (condition_4 | condition_1);

    condition_4 = false;

    if (condition_0) {
        condition_4 = condition_3;
    }

    if (condition_5) {
        matrix_accumulator_33 = 0x40u | matrix_accumulator_33;
    }

    condition_5 = true;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffbfffu;

    if (condition_0) {
        condition_5 = !condition_3;
    }

    if (condition_4) {
        matrix_accumulator_33 = 0x4000u | matrix_accumulator_33;
    }

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffffdfu;

    condition_5 = condition_5 & (condition_4 | condition_1);

    condition_4 = uint32_t(packed_half_2) != uint32_t(0x1u) && condition_2;

    if (condition_5) {
        matrix_accumulator_33 = 0x20u | matrix_accumulator_33;
    }

    condition_5 = uint32_t(packed_half_2) == uint32_t(0x1u) || !condition_2;

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xff7fffffu;

    if (condition_4) {
        matrix_accumulator_33 = 0x800000u | matrix_accumulator_33;
    }

    condition_4 = condition_5 & (condition_2 | condition_6);

    condition_2 = condition_5 & (condition_2 | condition_1);

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffffdu;

    matrix_fragment_22 = uint32_t(condition_2) << 2;

    condition_2 = uint32_t(packed_half_2) != uint32_t(0x1u) && condition_3;

    condition_5 = uint32_t(packed_half_2) == uint32_t(0x1u) || !condition_3;

    if (condition_4) {
        matrix_accumulator_33 = 0x2u | matrix_accumulator_33;
    }

    condition_3 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_2;

    condition_4 = int32_t(matrix_fragment_4) != int32_t(0u);

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffbfffffu;

    if (condition_2) {
        matrix_accumulator_33 = 0x400000u | matrix_accumulator_33;
    }

    condition_2 = condition_5 & (condition_3 | condition_6);

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffffeu;

    condition_3 = condition_5 & (condition_3 | condition_1);

    condition_5 = true;

    matrix_fragment_21 = uint32_t(condition_3) << 3;

    if (condition_0) {
        condition_5 = !condition_4;
    }

    if (condition_2) {
        matrix_accumulator_33 = 0x1u | matrix_accumulator_33;
    }

    condition_2 = false;

    if (condition_0) {
        condition_2 = condition_4;
    }

    condition_3 = (matrix_accumulator_33 & 0x10000u) != 0;

    matrix_fragment_4 = uint32_t(condition_3) << 3;

    condition_3 = (matrix_accumulator_33 & 0x100u) != 0;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffdfffu;

    if (condition_2) {
        matrix_accumulator_33 = 0x2000u | matrix_accumulator_33;
    }

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_2;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffff7u;

    condition_5 = condition_5 & (condition_2 | condition_1);

    condition_2 = false;

    if (condition_0) {
        condition_2 = condition_3;
    }

    if (condition_5) {
        matrix_accumulator_33 = 0x8u | matrix_accumulator_33;
    }

    condition_5 = true;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffffefffu;

    if (condition_0) {
        condition_5 = !condition_3;
    }

    if (condition_2) {
        matrix_accumulator_33 = 0x1000u | matrix_accumulator_33;
    }

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_2;

    condition_0 = uint32_t(packed_half_2) == uint32_t(0x1u) || !condition_4;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xfffffffbu;

    condition_5 = condition_5 & (condition_2 | condition_1);

    condition_2 = uint32_t(packed_half_2) != uint32_t(0x1u) && condition_4;

    if (condition_5) {
        matrix_accumulator_33 = 0x4u | matrix_accumulator_33;
    }

    condition_5 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_2;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffdfffffu;

    condition_4 = condition_0 & (condition_5 | condition_6);

    condition_5 = condition_0 & (condition_5 | condition_1);

    condition_0 = uint32_t(packed_half_2) != uint32_t(0x1u) && condition_3;

    if (condition_2) {
        matrix_accumulator_33 = 0x200000u | matrix_accumulator_33;
    }

    matrix_fragment_23 = uint32_t(condition_5) << 5;

    condition_5 = (matrix_accumulator_33 & 0x200u) != 0;

    condition_2 = (matrix_accumulator_33 & 0x40000u) != 0;

    matrix_accumulator_33 = matrix_accumulator_33 & 0xffefffffu;

    matrix_fragment_28 = uint32_t(condition_4) << 4;

    condition_4 = uint32_t(packed_half_2) == uint32_t(0x1u) || !condition_3;

    condition_3 = int32_t(matrix_fragment_4) != int32_t(0u);

    if (condition_0) {
        matrix_accumulator_33 = 0x100000u | matrix_accumulator_33;
    }

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) || condition_0;

    if (condition_5) {
        matrix_fragment_4 = matrix_fragment_17 * 0x4u;
    }

    condition_3 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_3;

    condition_6 = condition_4 & (condition_0 | condition_6);

    condition_1 = condition_4 & (condition_0 | condition_1);

    condition_0 = int32_t(matrix_fragment_5) != int32_t(0u);

    if (condition_5) {
        matrix_fragment_7 = !condition_3 ? matrix_fragment_4 : 0u;
    }

    matrix_fragment_14 = uint32_t(condition_1) << 1;

    if (condition_5) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    condition_1 = (matrix_accumulator_33 & 0x8000u) != 0;

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_2;

    matrix_fragment_15 = condition_6 ? 0x40u : 0u;

    matrix_fragment_13 = uint32_t(condition_1) << 1;

    condition_1 = (matrix_accumulator_33 & 0x80u) != 0;

    if (condition_0) {
        matrix_fragment_6 = matrix_fragment_11 * matrix_fragment_8;
    }

    condition_6 = (matrix_accumulator_33 & 0x40u) != 0;

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_16;
    }

    condition_3 = (matrix_accumulator_33 & 0x20u) != 0;

    if (condition_5) {
        matrix_fragment_6 = matrix_fragment_6 + packed_half_0 + (0u - packed_half_3);
    }

    condition_4 = (matrix_accumulator_33 & 0x4000u) != 0;

    if (condition_5) {
        matrix_fragment_7 = matrix_fragment_7 + matrix_fragment_6;

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_1) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];

        matrix_fragment_12 = matrix_fragment_17 * 0x4u;

        matrix_fragment_9 = matrix_fragment_9 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_5) {
        matrix_fragment_24 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_4;

    condition_5 = (matrix_accumulator_33 & 0x800000u) != 0;

    if (condition_1) {
        matrix_fragment_12 = !condition_2 ? matrix_fragment_12 : 0u;
    }

    condition_2 = (matrix_accumulator_33 & 0x2u) != 0;

    if (condition_1) {
        matrix_fragment_9 = matrix_fragment_12 + matrix_fragment_9;

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));

        matrix_fragment_25 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    condition_1 = int32_t(matrix_fragment_13) != int32_t(0u);

    if (condition_6) {
        matrix_fragment_4 = matrix_fragment_19 * 0x4u;
    }

    matrix_fragment_9 = 0u;

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_8;
    }

    condition_1 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_1;

    if (condition_6) {
        matrix_fragment_9 = matrix_fragment_9 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_3) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        matrix_fragment_12 = !condition_1 ? matrix_fragment_4 : 0u;
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_19 * 0x4u;
    }

    condition_1 = int32_t(matrix_fragment_14) != int32_t(0u);

    if (condition_6) {
        matrix_fragment_9 = matrix_fragment_12 + matrix_fragment_9;
    }

    matrix_fragment_12 = 0u;

    if (condition_6) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_13 = !condition_4 ? matrix_fragment_13 : 0u;
    }

    packed_half_36 = 0u - packed_half_3 + packed_half_0;

    condition_5 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_5;

    if (condition_0) {
        matrix_fragment_29 = matrix_fragment_11 * matrix_fragment_20;
    }

    if (condition_1) {
        matrix_fragment_46 = parameters[parameter_input_low_0];
        matrix_fragment_47 = parameters[parameter_input_high_1];
    }

    if (condition_6) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_fragment_9 = 0u;

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_16;
    }

    if (condition_6) {
        matrix_fragment_12 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    condition_6 = int32_t(matrix_fragment_15) != int32_t(0u);

    condition_4 = (matrix_accumulator_33 & 0x1u) != 0;

    if (condition_3) {
        matrix_fragment_14 = matrix_fragment_9 + packed_half_0 + (0u - packed_half_3);
    }

    matrix_fragment_9 = condition_0 ? matrix_fragment_8 : 0u;

    if (condition_2) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_13 + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_8 = matrix_fragment_17 * 0x4u;

        matrix_fragment_14 = packed_half_2 + matrix_fragment_9;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_2) {
        matrix_fragment_8 = !condition_5 ? matrix_fragment_8 : 0u;
    }

    matrix_fragment_13 = 0u;

    if (condition_6) {
        matrix_fragment_40 = parameters[parameter_input_low_0];
        matrix_fragment_41 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_11 * matrix_fragment_14 + packed_half_36;
    }

    if (condition_3) {
        matrix_fragment_13 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_8 + matrix_fragment_15;
    }

    matrix_fragment_14 = 0u;

    condition_3 = int32_t(matrix_fragment_21) != int32_t(0u);

    if (condition_2) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        matrix_fragment_26 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    condition_2 = int32_t(matrix_fragment_22) != int32_t(0u);

    if (condition_2) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];

        matrix_fragment_8 = matrix_fragment_19 * 0x4u;

        matrix_fragment_9 = packed_half_2 + matrix_fragment_9;

        matrix_fragment_8 = !condition_5 ? matrix_fragment_8 : 0u;

        matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_9 + packed_half_36;
    }

    matrix_fragment_15 = condition_0 ? matrix_fragment_16 : 0u;

    if (condition_4) {
        matrix_fragment_4 = matrix_fragment_17 * 0x4u;
    }

    condition_5 = (matrix_accumulator_33 & 0x10u) != 0;

    if (condition_2) {
        matrix_fragment_9 = matrix_fragment_9 + matrix_fragment_8;
    }

    if (condition_4) {
        matrix_fragment_5 = packed_half_2 + matrix_fragment_15;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_4) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
    }

    if (condition_2) {
        matrix_fragment_14 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    condition_2 = (matrix_accumulator_33 & 0x400000u) != 0;

    if (condition_4) {
        matrix_fragment_5 = matrix_fragment_11 * matrix_fragment_5 + packed_half_36;
    }

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_2;

    if (condition_3) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_4 = !condition_2 ? matrix_fragment_4 : 0u;

        matrix_fragment_5 = matrix_fragment_4 + matrix_fragment_5;
    }

    if (condition_3) {
        matrix_fragment_15 = packed_half_2 + matrix_fragment_15;
    }

    if (condition_4) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_3) {
        matrix_fragment_8 = matrix_fragment_19 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_27 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    if (condition_3) {
        matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_15 + packed_half_36;
    }

    matrix_fragment_15 = 0u;

    matrix_fragment_16 = 0u;

    if (condition_3) {
        matrix_fragment_8 = !condition_2 ? matrix_fragment_8 : 0u;
    }

    condition_2 = (matrix_accumulator_33 & 0x80000u) != 0;

    condition_4 = (matrix_accumulator_33 & 0x400u) != 0;

    if (condition_3) {
        matrix_fragment_9 = matrix_fragment_9 + matrix_fragment_8;
    }

    if (condition_5) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    matrix_fragment_8 = 0u;

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_2;

    if (condition_0) {
        matrix_fragment_8 = matrix_fragment_11 * matrix_fragment_18;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_5) {
        matrix_fragment_9 = matrix_fragment_17 * 0x4u;

        matrix_fragment_8 = matrix_fragment_8 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_3) {
        matrix_fragment_15 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), memory_ranges, error_counts);
    }

    matrix_fragment_22 = 0u;

    if (condition_5) {
        matrix_fragment_9 = !condition_2 ? matrix_fragment_9 : 0u;
    }

    if (condition_0) {
        matrix_fragment_22 = matrix_fragment_11 * matrix_fragment_18;
    }

    condition_2 = (matrix_accumulator_33 & 0x8u) != 0;

    condition_3 = (matrix_accumulator_33 & 0x4u) != 0;

    if (condition_5) {
        matrix_fragment_9 = matrix_fragment_9 + matrix_fragment_8;

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_fragment_6 = 0u;

    matrix_fragment_9 = condition_0 ? matrix_fragment_18 : 0u;

    if (condition_5) {
        matrix_fragment_16 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_23) != int32_t(0u);

    if (condition_0) {
        matrix_fragment_6 = matrix_fragment_11 * matrix_fragment_20;
    }

    matrix_fragment_23 = condition_0 ? matrix_fragment_20 : 0u;

    condition_0 = (matrix_accumulator_33 & 0x20000u) != 0;

    if (condition_4) {
        matrix_fragment_7 = matrix_fragment_17 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_20 = parameters[parameter_input_low_0];
        matrix_fragment_21 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_30 = matrix_fragment_19 * 0x4u;

        matrix_fragment_29 = matrix_fragment_29 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_6) {
        matrix_fragment_18 = matrix_fragment_17 * 0x4u;
    }

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_0;

    if (condition_1) {
        matrix_accumulator_32 = matrix_fragment_19 * 0x4u;
    }

    if (condition_5) {
        matrix_fragment_31 = matrix_fragment_19 * 0x4u;
    }

    if (condition_4) {
        matrix_fragment_7 = !condition_0 ? matrix_fragment_7 : 0u;
    }

    condition_0 = (matrix_accumulator_33 & 0x2000u) != 0;

    if (condition_3) {
        matrix_fragment_42 = parameters[parameter_input_low_0];
        matrix_fragment_43 = parameters[parameter_input_high_1];
    }

    condition_4 = int32_t(matrix_fragment_28) != int32_t(0u);

    if (condition_2) {
        matrix_fragment_28 = matrix_fragment_19 * 0x4u;
    }

    if (condition_6) {
        matrix_fragment_19 = packed_half_2 + matrix_fragment_23;
    }

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_0;

    if (condition_6) {
        matrix_fragment_19 = matrix_fragment_11 * matrix_fragment_19 + packed_half_36;
    }

    if (condition_2) {
        matrix_fragment_28 = !condition_0 ? matrix_fragment_28 : 0u;
    }

    condition_0 = (matrix_accumulator_33 & 0x1000u) != 0;

    if (condition_5) {
        matrix_fragment_44 = parameters[parameter_input_low_0];
        matrix_fragment_45 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_8 = matrix_fragment_17 * 0x4u;
    }

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_0;

    if (condition_4) {
        matrix_accumulator_38 = parameters[parameter_input_low_0];
        matrix_accumulator_39 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_30 = !condition_0 ? matrix_fragment_30 : 0u;
    }

    condition_0 = (matrix_accumulator_33 & 0x200000u) != 0;

    if (condition_3) {
        matrix_fragment_29 = matrix_fragment_30 + matrix_fragment_29;
    }

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_0;

    if (condition_3) {
        split_address_words(matrix_fragment_42, matrix_fragment_43,
                            int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_42, matrix_fragment_43));
    }

    if (condition_4) {
        matrix_fragment_8 = !condition_0 ? matrix_fragment_8 : 0u;
    }

    if (condition_5) {
        matrix_fragment_31 = !condition_0 ? matrix_fragment_31 : 0u;
    }

    condition_0 = (matrix_accumulator_33 & 0x100000u) != 0;

    condition_0 = uint32_t(matrix_fragment_10) == uint32_t(0x1u) && !condition_0;

    if (condition_4) {
        matrix_fragment_10 = packed_half_2 + matrix_fragment_9;
    }

    if (condition_6) {
        matrix_fragment_18 = !condition_0 ? matrix_fragment_18 : 0u;
    }

    if (condition_1) {
        matrix_accumulator_32 = !condition_0 ? matrix_accumulator_32 : 0u;
    }

    condition_0 = (matrix_accumulator_33 & 0x400u) != 0;

    if (condition_4) {
        matrix_fragment_17 = matrix_fragment_11 * matrix_fragment_10 + packed_half_36;
    }

    if (condition_5) {
        matrix_fragment_10 = packed_half_2 + matrix_fragment_9;
    }

    if (condition_6) {
        matrix_fragment_19 = matrix_fragment_18 + matrix_fragment_19;
    }

    if (condition_1) {
        packed_half_2 = packed_half_2 + matrix_fragment_23;
    }

    if (condition_2) {
        matrix_fragment_9 = matrix_fragment_22 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_4) {
        matrix_fragment_17 = matrix_fragment_8 + matrix_fragment_17;
    }

    if (condition_6) {
        split_address_words(matrix_fragment_40, matrix_fragment_41,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_40, matrix_fragment_41));
    }

    if (condition_0) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];

        matrix_fragment_6 = matrix_fragment_6 + packed_half_0 + (0u - packed_half_3);
    }

    if (condition_4) {
        split_address_words(matrix_accumulator_38, matrix_accumulator_39,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_38, matrix_accumulator_39));
    }

    matrix_fragment_17 = 0u;

    if (condition_2) {
        matrix_fragment_9 = matrix_fragment_28 + matrix_fragment_9;
    }

    if (condition_0) {
        matrix_fragment_7 = matrix_fragment_7 + matrix_fragment_6;
    }

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    if (condition_5) {
        matrix_fragment_10 = matrix_fragment_11 * matrix_fragment_10 + packed_half_36;
    }

    if (condition_1) {
        matrix_fragment_11 = matrix_fragment_11 * packed_half_2 + packed_half_36;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (condition_4) {
        matrix_fragment_18 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_38, matrix_accumulator_39), memory_ranges,
            error_counts);
    }

    if (condition_5) {
        matrix_fragment_31 = matrix_fragment_10 + matrix_fragment_31;
    }

    if (condition_1) {
        matrix_fragment_11 = matrix_fragment_11 + matrix_accumulator_32;
    }

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    if (condition_6) {
        matrix_fragment_19 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_40, matrix_fragment_41), memory_ranges, error_counts);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            int64_t(int32_t(matrix_fragment_31)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_44, matrix_fragment_45));
    }

    if (condition_2) {
        matrix_fragment_8 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), memory_ranges, error_counts);
    }

    if (condition_1) {
        split_address_words(matrix_fragment_46, matrix_fragment_47,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_46, matrix_fragment_47));
    }

    if (condition_0) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_3) {
        matrix_fragment_9 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_42, matrix_fragment_43), memory_ranges, error_counts);
    }

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(packed_half_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_120, packed_half_121));

    if (condition_0) {
        matrix_fragment_17 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
    }

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x200ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    if (condition_5) {
        matrix_fragment_10 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_44, matrix_fragment_45), memory_ranges, error_counts);
    }

    if (condition_1) {
        matrix_fragment_11 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_46, matrix_fragment_47), memory_ranges, error_counts);
    }

    split_address_words(packed_half_36, packed_half_37,
                        int64_t(int32_t(packed_half_36)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_120, packed_half_121));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1000ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    matrix_accumulator_33 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_36, packed_half_37) + 0x2010ull, memory_ranges, error_counts);

    packed_half_35 = packed_half_0 + 0x4u + (0u - packed_half_3);

    packed_half_109 = packed_half_0 + 0x8u + (0u - packed_half_3);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1200ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    split_address_words(packed_half_34, packed_half_35,
                        uint64_t(uint32_t(packed_half_35)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_120, packed_half_121));

    packed_half_2 = 0x2b28u;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2460ull, matrix_fragment_132,
        matrix_fragment_133, matrix_fragment_134, matrix_fragment_135, memory_ranges, error_counts);

    matrix_accumulator_32 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_34, packed_half_35) + 0x2010ull, memory_ranges, error_counts);

    split_address_words(packed_half_108, packed_half_109,
                        uint64_t(uint32_t(packed_half_109)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_120, packed_half_121));

    packed_half_36 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_36, packed_half_37) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_68 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_108, packed_half_109) + 0x2010ull, memory_ranges, error_counts);

    packed_half_3 = packed_half_0 + 0xcu + (0u - packed_half_3);

    packed_half_34 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_34, packed_half_35) + 0x5070ull, memory_ranges, error_counts);

    split_address_words(packed_half_120, packed_half_121,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_120, packed_half_121));

    packed_half_108 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_108, packed_half_109) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_69 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_120, packed_half_121) + 0x2010ull, memory_ranges, error_counts);

    packed_half_0 = permute_signed(packed_half_2, packed_half_0, 0x7610u);

    packed_half_120 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_120, packed_half_121) + 0x5070ull, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_64),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_72),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_73 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3b283b28u);

    matrix_accumulator_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0xc400c400u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_67),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_76),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0xc400c400u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0xc400c400u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3b283b28u);

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_63),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_65),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_67),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_16, matrix_fragment_17,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_28,
                             matrix_fragment_29, 0u, 0u);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0xc400c400u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_65),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_accumulator_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52),
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3b283b28u);

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_65));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x400ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_73),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_28 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_29 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x44004400u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_43),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x44004400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0xc400c400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_80),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_81),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_82),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_39 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0xc400c400u);

    matrix_accumulator_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_accumulator_38 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0xc400c400u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3b283b28u);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0xc400c400u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0xc400c400u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_38),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_81),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39),
        select_half_operand<2, false, false>(matrix_fragment_80), 0x3b283b28u);

    matrix_accumulator_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38),
        select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_30),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_39 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0xc400c400u);

    matrix_accumulator_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_82),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3b283b28u);

    matrix_accumulator_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_accumulator_84));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x600ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_38 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0xc400c400u);

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_5),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39),
        select_half_operand<2, false, false>(packed_half_3), 0x3b283b28u);

    packed_half_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_5),
        select_half_operand<2, false, false>(packed_half_2), 0x3b283b28u);

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_38),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0xc400c400u);

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38),
        select_half_operand<2, false, false>(packed_half_2), 0x3b283b28u);

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_5),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_5),
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3b283b28u);

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_16);

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_38,
                             matrix_accumulator_39);

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_88 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_8);

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                       select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_24 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_25 >> 16);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    packed_half_2 = unpack_fp8_bits(matrix_fragment_24);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_25);

    matrix_accumulator_53 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_53 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                      select_half_operand<2, false, false>(matrix_accumulator_53));

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_40, matrix_fragment_41,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_20,
                             matrix_fragment_21, packed_half_2, packed_half_3);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_accumulator_52,
                             matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_38,
                             matrix_accumulator_39);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_26);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_27);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_accumulator_52,
                             matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1400ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    matrix_accumulator_38 = unpack_fp8_bits(matrix_fragment_26 >> 16);

    matrix_accumulator_39 = unpack_fp8_bits(matrix_fragment_27 >> 16);

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_38,
                             matrix_accumulator_39);

    matrix_accumulator_38 = unpack_fp8_bits(matrix_fragment_18);

    matrix_accumulator_39 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_56 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_57 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_accumulator_52,
                             matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1600ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_accumulator_52,
                             matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_52 = unpack_fp8_bits(matrix_fragment_10);

    matrix_accumulator_53 = unpack_fp8_bits(matrix_fragment_11);

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x800ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(packed_half_98, matrix_fragment_99, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0xa00ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    packed_half_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0x44004400u);

    matrix_fragment_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0xc400c400u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_104), 0x3b283b28u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_102), 0x3b283b28u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    packed_half_107 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0xc400c400u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    packed_half_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_98),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x44004400u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    packed_half_111 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

    packed_half_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0xc400c400u);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                          select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0xc400c400u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_98),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0xc400c400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_98),
        select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_104), 0x3b283b28u);

    packed_half_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_100));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    packed_half_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(packed_half_102));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(packed_half_114));

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(packed_half_107), 0x3b283b28u);

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(packed_half_115));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_accumulator_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

    matrix_fragment_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_accumulator_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(packed_half_116));

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_accumulator_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_94),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_115),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94),
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3b283b28u);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_115),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                          select_half_operand<2, false, false>(packed_half_111));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_28 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

    multiply_accumulate_half(packed_half_98, matrix_fragment_99, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0xc400c400u);

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_28 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_64),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_69),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_accumulator_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_53 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_117),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

    packed_half_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_117),
        select_half_operand<2, false, false>(packed_half_118), 0x3b283b28u);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3b283b28u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_53),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(packed_half_101));

    packed_half_118 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                          select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_69),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                       select_half_operand<2, false, false>(0u));

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3b283b28u);

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_20, matrix_fragment_21, packed_half_2, packed_half_3);

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                          select_half_operand<2, false, false>(packed_half_117));

    matrix_accumulator_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                          select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1800ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0x44004400u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0xc400c400u);

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3b283b28u);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                       select_half_operand<2, false, false>(packed_half_100));

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_3),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_98),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_3),
        select_half_operand<2, false, false>(packed_half_2), 0x3b283b28u);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_64, matrix_fragment_65,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_62, matrix_fragment_63);

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(matrix_fragment_63));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1a00ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                       select_half_operand<2, false, false>(packed_half_110));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_82,
                             matrix_fragment_83);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0xe00ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_86,
                             matrix_accumulator_87);

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                       select_half_operand<2, false, false>(packed_half_114));

    multiply_accumulate_half(packed_half_98, matrix_fragment_99, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_76,
                             matrix_accumulator_77);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0xc00ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_102), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_46),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0x44004400u);

    packed_half_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0xc400c400u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                          select_half_operand<2, false, false>(packed_half_100));

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_61),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    packed_half_102 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                          select_half_operand<2, false, false>(packed_half_102));

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    packed_half_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_98),
                                         select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<2, false, false>(packed_half_103));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                          select_half_operand<2, false, false>(packed_half_112));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

    packed_half_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(matrix_fragment_96), 0x3b283b28u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0xc400c400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_96),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(packed_half_107), 0x3b283b28u);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<2, false, false>(matrix_fragment_104), 0x3b283b28u);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                          select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(packed_half_118));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                          select_half_operand<2, false, false>(packed_half_114));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_44),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

    packed_half_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0xc400c400u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_123 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    packed_half_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    matrix_accumulator_123 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0xc400c400u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_116),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0xc400c400u);

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                          select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    packed_half_118 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                          select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_45),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                    select_half_operand<2, false, false>(0u));

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1c00ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_123),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_116),
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3b283b28u);

    packed_half_119 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    matrix_fragment_22 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                       select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

    packed_half_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1e00ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_115),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_115),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(0u));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(packed_half_101));

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                       select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_92,
                             matrix_accumulator_93);

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_48, matrix_fragment_49,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                             matrix_fragment_29, packed_half_2, packed_half_3);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5);

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_76,
                             matrix_accumulator_77);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2260ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(matrix_fragment_61));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_82,
                             matrix_fragment_83);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2060ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_98),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_98),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                          select_half_operand<2, false, false>(packed_half_103));

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                          select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                       select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                       select_half_operand<2, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    matrix_fragment_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_accumulator_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    packed_half_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_72),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_74),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0xc400c400u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2660ull, matrix_fragment_72,
        matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    packed_half_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    packed_half_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0xc400c400u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(packed_half_98));

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0xc400c400u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_96),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_98),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<2, false, false>(matrix_fragment_97), 0x3b283b28u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_98),
        select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

    matrix_fragment_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_17), 0x3b283b28u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_18),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_18),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_98 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                         select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_104), 0x3b283b28u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    matrix_fragment_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(packed_half_102));

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(matrix_fragment_16), 0x3b283b28u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_16 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x44004400u);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_19 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x44004400u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_29),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x44004400u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_27),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_31),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_105));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_13),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

    packed_half_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_13),
        select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2860ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_63));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_40, matrix_fragment_41, packed_half_2, packed_half_3);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(packed_half_116, packed_half_117, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(packed_half_118, packed_half_119, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_38,
                             matrix_accumulator_39);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2a60ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_76,
                             matrix_accumulator_77);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_20, matrix_fragment_21, packed_half_114,
                             packed_half_115);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                       select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_90,
                             matrix_accumulator_91);

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_60,
                             matrix_fragment_61);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = parameters[parameter_trained_weights_low_4];
    packed_half_3 = parameters[parameter_trained_weights_high_5];

    packed_half_3 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, packed_half_3) + 0x4c60ull, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_162, packed_half_163, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_162),
                                        select_half_operand<2, false, false>(packed_half_162));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                      select_half_operand<2, false, false>(packed_half_100),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(packed_half_130, packed_half_131, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_accumulator_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_163),
                                        select_half_operand<2, false, false>(packed_half_163));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                                      select_half_operand<2, false, false>(packed_half_101),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_accumulator_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                        select_half_operand<2, false, false>(packed_half_131));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                            select_half_operand<2, false, false>(packed_half_129));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(packed_half_2));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_79 = result_word;
    }

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                        select_half_operand<2, false, false>(packed_half_130));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                            select_half_operand<2, false, false>(packed_half_128));

    matrix_accumulator_39 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(matrix_accumulator_39));

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                   select_half_operand<2, false, false>(packed_half_2));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_93 = result_word;
    }

    matrix_accumulator_38 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                                      select_half_operand<2, false, false>(matrix_accumulator_32),
                                      select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_accumulator_93 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                       select_half_operand<2, false, false>(matrix_accumulator_93));

    multiply_accumulate_half(matrix_fragment_150, matrix_fragment_151, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_132, matrix_fragment_133, 0u, 0u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                   select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_134, matrix_fragment_135, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_fragment_72 = permute_signed(matrix_accumulator_79, matrix_accumulator_79, 0x5432u);

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_fragment_72));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_93), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_fragment_73 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_86), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_87 = result_word;
    }

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_132, matrix_fragment_133, 0u, 0u);

    matrix_accumulator_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                                      select_half_operand<2, false, false>(matrix_fragment_150),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_134, matrix_fragment_135, 0u, 0u);

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79), 0x4100410u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                   select_half_operand<2, false, false>(matrix_fragment_72));

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_fragment_72 = half_to_float_bits((matrix_fragment_73 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_73 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_132, matrix_fragment_133, 0u, 0u);

    packed_half_161 = half_to_float_bits((matrix_accumulator_79 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_134, matrix_fragment_135, 0u, 0u);

    matrix_accumulator_79 = half_to_float_bits((matrix_accumulator_79 >> 16) * 65537u);

    packed_half_107 = rsqrt_bits(matrix_fragment_72);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    multiply_accumulate_half(packed_half_148, packed_half_149, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                                      select_half_operand<2, false, false>(matrix_fragment_151),
                                      select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_accumulator_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_74 = rsqrt_bits(matrix_fragment_73);

    matrix_accumulator_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_78 = rsqrt_bits(matrix_accumulator_79);

    matrix_accumulator_87 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(matrix_accumulator_87));

    multiply_accumulate_half(matrix_fragment_132, matrix_fragment_133, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_132, matrix_fragment_133, 0u, 0u);

    packed_half_103 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    multiply_accumulate_half(matrix_fragment_134, matrix_fragment_135, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_134, matrix_fragment_135, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_87), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_2 = result_word;
    }

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_accumulator_86));

    packed_half_161 = rsqrt_bits(packed_half_161);

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_95), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_106 = result_word;
    }

    packed_half_107 = pack_half(matrix_accumulator_74, packed_half_107);

    packed_half_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                    select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_86), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    packed_half_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                     select_half_operand<2, false, false>(matrix_fragment_97));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2c60ull, matrix_fragment_72,
        matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    packed_half_103 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                     select_half_operand<2, false, false>(packed_half_103));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_147 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_102), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_153 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_103), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_146 = result_word;
    }

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    packed_half_161 = pack_half(matrix_accumulator_78, packed_half_161);

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                            select_half_operand<2, false, false>(packed_half_148));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                   select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_132),
                                      select_half_operand<2, false, false>(matrix_fragment_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_134),
                                      select_half_operand<2, false, false>(matrix_fragment_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_accumulator_141));

    matrix_accumulator_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_135),
                                      select_half_operand<2, false, false>(matrix_fragment_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_accumulator_94 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_133),
                                      select_half_operand<2, false, false>(matrix_fragment_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                       select_half_operand<2, false, false>(packed_half_106));

    matrix_accumulator_93 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(matrix_fragment_99));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2e60ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    matrix_accumulator_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_accumulator_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_94), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    packed_half_98 = add_half_pairs(select_half_operand<2, false, false>(packed_half_98),
                                    select_half_operand<2, false, false>(matrix_accumulator_147));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_95), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_96 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_93), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_106 = result_word;
    }

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_102),
                                     select_half_operand<2, false, false>(packed_half_153));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_86), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_87 = result_word;
    }

    packed_half_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(packed_half_107));

    packed_half_103 = add_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                                     select_half_operand<2, false, false>(matrix_accumulator_146));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                          select_half_operand<2, false, false>(packed_half_161));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_85 = result_word;
    }

    packed_half_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(packed_half_161));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_103), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_84 = result_word;
    }

    packed_half_162 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_162),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_161 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_4 = permute_signed(packed_half_153, packed_half_153, 0x5432u);

    matrix_accumulator_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                       select_half_operand<2, false, false>(matrix_fragment_96));

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_106 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                     select_half_operand<2, false, false>(packed_half_106));

    matrix_accumulator_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(matrix_accumulator_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    packed_half_164 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_163),
                                          select_half_operand<2, false, false>(packed_half_107));

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_52, matrix_accumulator_53, 0u, 0u);

    packed_half_163 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(packed_half_107));

    matrix_accumulator_94 = permute_signed(matrix_accumulator_95, matrix_accumulator_95, 0x5432u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_92 = half_to_float_bits((matrix_accumulator_93 & 65535u) * 65537u);

    matrix_fragment_97 = permute_signed(matrix_accumulator_86, matrix_accumulator_86, 0x5432u);

    matrix_accumulator_85 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_98),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_93 = half_to_float_bits((matrix_accumulator_93 >> 16) * 65537u);

    matrix_accumulator_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_106), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_87 = result_word;
    }

    packed_half_103 = add_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                                     select_half_operand<2, false, false>(matrix_accumulator_84));

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_84 = permute_signed(matrix_accumulator_85, matrix_accumulator_85, 0x5432u);

    matrix_accumulator_92 = rsqrt_bits(matrix_accumulator_92);

    matrix_accumulator_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_accumulator_93 = rsqrt_bits(matrix_accumulator_93);

    matrix_accumulator_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86), 0x4100410u);

    matrix_accumulator_85 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                       select_half_operand<2, false, false>(matrix_accumulator_84));

    multiply_accumulate_half(matrix_accumulator_152, packed_half_153, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3060ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_accumulator_94 = half_to_float_bits((matrix_accumulator_95 & 65535u) * 65537u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                                          select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_97 = half_to_float_bits((matrix_accumulator_95 >> 16) * 65537u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85), 0x4100410u);

    matrix_accumulator_95 = half_to_float_bits((matrix_accumulator_86 & 65535u) * 65537u);

    matrix_fragment_96 = half_to_float_bits((matrix_accumulator_86 >> 16) * 65537u);

    matrix_accumulator_94 = rsqrt_bits(matrix_accumulator_94);

    packed_half_154 = half_to_float_bits((packed_half_107 & 65535u) * 65537u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_99 = half_to_float_bits((packed_half_107 >> 16) * 65537u);

    matrix_fragment_97 = rsqrt_bits(matrix_fragment_97);

    matrix_accumulator_92 = pack_half(matrix_accumulator_93, matrix_accumulator_92);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_98 = result_word;
    }

    matrix_accumulator_95 = rsqrt_bits(matrix_accumulator_95);

    packed_half_106 = add_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                     select_half_operand<2, false, false>(matrix_accumulator_87));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3260ull, matrix_accumulator_84,
        matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87, memory_ranges,
        error_counts);

    matrix_fragment_96 = rsqrt_bits(matrix_fragment_96);

    packed_half_154 = rsqrt_bits(packed_half_154);

    matrix_accumulator_93 = rsqrt_bits(matrix_fragment_99);

    packed_half_3 = pack_half(0u, packed_half_3);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_94 = pack_half(matrix_fragment_97, matrix_accumulator_94);

    packed_half_159 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_95 = pack_half(matrix_fragment_96, matrix_accumulator_95);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_102),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_159),
                                        select_half_operand<2, false, false>(0u));

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    packed_half_158 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    packed_half_154 = pack_half(matrix_accumulator_93, packed_half_154);

    packed_half_160 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3460ull, matrix_accumulator_88,
        matrix_accumulator_89, matrix_accumulator_90, matrix_accumulator_91, memory_ranges,
        error_counts);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_102 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                          select_half_operand<0, false, false>(packed_half_3));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3660ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                                          select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_150 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_150),
                                        select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_151 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                        select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                        select_half_operand<2, false, false>(packed_half_100));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3860ull, matrix_fragment_96,
        matrix_fragment_97, packed_half_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                        select_half_operand<2, false, false>(packed_half_102));

    packed_half_100 = permute_signed(packed_half_103, packed_half_103, 0x5432u);

    packed_half_101 = permute_signed(packed_half_106, packed_half_106, 0x5432u);

    packed_half_107 = add_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                                     select_half_operand<2, false, false>(packed_half_100));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                     select_half_operand<2, false, false>(packed_half_101));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3a60ull, packed_half_100,
        packed_half_101, packed_half_102, packed_half_103, memory_ranges, error_counts);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_161),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_161 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x4100410u);

    packed_half_162 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_162),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    float_bits_165 = half_to_float_bits((packed_half_161 & 65535u) * 65537u);

    packed_half_161 = half_to_float_bits((packed_half_161 >> 16) * 65537u);

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_162),
                                    select_half_operand<2, false, false>(packed_half_155));

    packed_half_155 = rsqrt_bits(float_bits_165);

    packed_half_163 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_163),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_161 = rsqrt_bits(packed_half_161);

    packed_half_159 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x4100410u);

    packed_half_164 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_164),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_163),
                                    select_half_operand<2, false, false>(0u));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_164),
                                    select_half_operand<2, false, false>(packed_half_107));

    packed_half_162 = half_to_float_bits((packed_half_159 & 65535u) * 65537u);

    packed_half_157 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                    select_half_operand<2, false, false>(0u));

    packed_half_164 = half_to_float_bits((packed_half_159 >> 16) * 65537u);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_154));

    packed_half_159 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(packed_half_154));

    packed_half_161 = pack_half(packed_half_161, packed_half_155);

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_161));

    multiply_accumulate_half(packed_half_154, packed_half_155, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_150, matrix_fragment_151, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                       select_half_operand<2, false, false>(packed_half_157));

    packed_half_157 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_39 = permute_signed(packed_half_2, packed_half_2, 0x5432u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                      select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5432u);

    packed_half_163 = rsqrt_bits(packed_half_162);

    packed_half_161 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_139 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    packed_half_162 = rsqrt_bits(packed_half_164);

    matrix_fragment_42 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_138 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_161),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_26 = permute_signed(matrix_fragment_5, matrix_fragment_5, 0x5432u);

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_159),
                      select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x4100410u);

    matrix_accumulator_138 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x4100410u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_158),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    packed_half_159 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    packed_half_2 = half_to_float_bits((matrix_accumulator_138 & 65535u) * 65537u);

    packed_half_157 = half_to_float_bits((matrix_accumulator_138 >> 16) * 65537u);

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_78,
                             matrix_accumulator_79);

    packed_half_156 = pack_half(packed_half_162, packed_half_163);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x4100410u);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    matrix_accumulator_79 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_159 = rsqrt_bits(packed_half_159);

    matrix_fragment_26 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    packed_half_0 = half_to_float_bits((matrix_fragment_28 & 65535u) * 65537u);

    matrix_fragment_28 = half_to_float_bits((matrix_fragment_28 >> 16) * 65537u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(0u));

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x4100410u);

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_29 = rsqrt_bits(matrix_fragment_28);

    matrix_fragment_25 = half_to_float_bits((matrix_fragment_26 & 65535u) * 65537u);

    matrix_fragment_26 = half_to_float_bits((matrix_fragment_26 >> 16) * 65537u);

    packed_half_156 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    packed_half_2 = rsqrt_bits(packed_half_2);

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                      select_half_operand<2, false, false>(matrix_accumulator_79));

    packed_half_157 = rsqrt_bits(packed_half_157);

    matrix_fragment_27 = rsqrt_bits(matrix_fragment_26);

    packed_half_0 = pack_half(matrix_fragment_29, packed_half_0);

    matrix_fragment_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_135),
                            select_half_operand<2, false, false>(packed_half_0));

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_81 = pack_half(packed_half_159, matrix_fragment_24);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_25);

    matrix_fragment_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_134),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_132),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_82,
                             matrix_fragment_83);

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(matrix_fragment_81));

    packed_half_2 = pack_half(packed_half_157, packed_half_2);

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_133),
                            select_half_operand<2, false, false>(packed_half_0));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_135 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_135),
                                        select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_24 = pack_half(matrix_fragment_27, matrix_fragment_24);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(packed_half_148));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_135));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(matrix_fragment_24));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_2));

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                          select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<0, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(0u));

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                  select_half_operand<2, false, false>(0u));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_104,
                             matrix_fragment_105, packed_half_106, packed_half_107,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_0 = 0x29c0u;

    packed_half_3 = permute_signed(packed_half_3, packed_half_0, 0x5410u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_90,
                             matrix_accumulator_91);

    packed_half_154 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_154),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_138 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_150, matrix_fragment_151, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_accumulator_136 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x3c203c20u);

    packed_half_155 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_155),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_138 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3c203c20u);

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_accumulator_92,
                             matrix_accumulator_93);

    matrix_accumulator_139 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3c203c20u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_5),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_154 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x3e473e47u);

    multiply_accumulate_half(packed_half_130, packed_half_131, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_accumulator_137 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_accumulator_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_138 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3e473e47u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_136 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3c203c20u);

    packed_half_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_accumulator_139 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3c203c20u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3e473e47u);

    multiply_accumulate_half(packed_half_98, matrix_fragment_99, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_78, matrix_accumulator_79, packed_half_98,
                             matrix_fragment_99);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3c203c20u);

    matrix_accumulator_136 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3e473e47u);

    packed_half_155 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x3e473e47u);

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_80, matrix_fragment_81, packed_half_100,
                             packed_half_101);

    matrix_accumulator_139 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3e473e47u);

    matrix_fragment_28 = (packed_half_154 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_92 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_90 = (matrix_accumulator_138 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(packed_half_102, packed_half_103, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_82, matrix_fragment_83, packed_half_102,
                             packed_half_103);

    matrix_accumulator_137 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3c203c20u);

    matrix_accumulator_84 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3c203c20u);

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0x3c203c20u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3e473e47u);

    packed_half_159 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                   select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_137 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3e473e47u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                   select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_4 = (matrix_accumulator_136 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (packed_half_155 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_87 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_86 = (matrix_accumulator_139 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_84 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3e473e47u);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104), 0x3e473e47u);

    matrix_accumulator_74 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                           select_half_operand<2, false, false>(packed_half_2));

    packed_half_161 = (matrix_fragment_5 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_159),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_160 = (matrix_accumulator_137 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                       select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_94 = (matrix_accumulator_84 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                       select_half_operand<2, false, false>(matrix_accumulator_86));

    packed_half_2 = (matrix_fragment_104 << 0x5u) + 0x7ff88000u;

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_128),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_129),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_130),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_131),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_140 = threadIdx.x & 31u;

    matrix_accumulator_74 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                       select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3c203c20u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                   select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_161),
                                       select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3c203c20u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                        select_half_operand<2, false, false>(packed_half_2));

    packed_half_128 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_128), 0x3c203c20u);

    matrix_accumulator_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_129 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_129), 0x3c203c20u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_98),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_130 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_130), 0x3c203c20u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_89 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3c203c20u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3c203c20u);

    packed_half_131 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_131), 0x3c203c20u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                   select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3e473e47u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_88 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3e473e47u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_128), 0x3e473e47u);

    matrix_accumulator_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_129 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_129), 0x3e473e47u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_130), 0x3e473e47u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_89 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3e473e47u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_134 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3e473e47u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_131), 0x3e473e47u);

    matrix_accumulator_32 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3c203c20u);

    packed_half_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0x3c203c20u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_107 = (packed_half_106 << 0x5u) + 0x7ff88000u;

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105), 0x3c203c20u);

    packed_half_130 = (matrix_accumulator_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    packed_half_106 = (matrix_fragment_24 << 0x5u) + 0x7ff88000u;

    packed_half_131 = (packed_half_129 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_33 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    packed_half_128 = (matrix_accumulator_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_134 = (matrix_fragment_134 << 0x5u) + 0x7ff88000u;

    packed_half_129 = (matrix_fragment_25 << 0x5u) + 0x7ff88000u;

    packed_half_158 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3e473e47u);

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0x3e473e47u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0x3c203c20u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0x3c203c20u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3e473e47u);

    matrix_fragment_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105), 0x3e473e47u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                        select_half_operand<2, false, false>(packed_half_107));

    packed_half_157 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3e473e47u);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0x3c203c20u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0x3c203c20u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                        select_half_operand<2, false, false>(matrix_fragment_134));

    packed_half_154 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0x3e473e47u);

    matrix_accumulator_88 = add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                           select_half_operand<2, false, false>(packed_half_129));

    packed_half_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0x3e473e47u);

    packed_half_158 = (packed_half_158 << 0x5u) + 0x7ff88000u;

    packed_half_156 = (packed_half_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_149 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0x3e473e47u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0x3e473e47u);

    matrix_fragment_96 = (matrix_fragment_26 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_95 = (matrix_fragment_105 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                       select_half_operand<2, false, false>(matrix_accumulator_88));

    packed_half_157 = (packed_half_157 << 0x5u) + 0x7ff88000u;

    packed_half_155 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                                        select_half_operand<2, false, false>(packed_half_156));

    packed_half_154 = (packed_half_154 << 0x5u) + 0x7ff88000u;

    packed_half_148 = (packed_half_102 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_149 = (packed_half_149 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_88 = add_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                           select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_135 = (packed_half_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                        select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_25 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                       select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                        select_half_operand<2, false, false>(matrix_fragment_135));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 = matrix_accumulator_140 * 0x4u;

    condition_0 = (matrix_accumulator_140 & 0x1u) != 0;

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    condition_1 = (matrix_accumulator_140 & 0x2u) != 0;

    matrix_accumulator_88 = matrix_fragment_27 & 0x1cu;

    matrix_fragment_26 = condition_0 ? matrix_fragment_5 : packed_half_0;

    matrix_fragment_27 = condition_0 ? matrix_fragment_25 : matrix_fragment_24;

    matrix_fragment_5 = condition_0 ? packed_half_0 : matrix_fragment_5;

    matrix_fragment_24 = condition_0 ? matrix_fragment_24 : matrix_fragment_25;

    packed_half_0 = uint32_t(int64_t(int32_t(matrix_accumulator_140)) * (1ll << 0x1du) >> 32) +
                    matrix_accumulator_88;

    matrix_fragment_25 = !condition_1 ? matrix_fragment_26 : matrix_fragment_27;

    matrix_fragment_27 = !condition_1 ? matrix_fragment_27 : matrix_fragment_26;

    matrix_fragment_26 = !condition_1 ? matrix_fragment_5 : matrix_fragment_24;

    matrix_accumulator_88 = !condition_1 ? matrix_fragment_24 : matrix_fragment_5;

    float_bits_165 = 0x1u ^ packed_half_0;

    packed_half_164 = 0x2u ^ packed_half_0;

    packed_half_163 = 0x3u ^ packed_half_0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_25), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_25 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_26), "r"(float_bits_165), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_27), "r"(packed_half_164), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_27 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_88), "r"(packed_half_163), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_88 = result_word;
    }

    condition_2 = (matrix_accumulator_140 & 0x8u) != 0;

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    condition_3 = (matrix_accumulator_140 & 0x10u) != 0;

    matrix_fragment_5 = !condition_2 ? matrix_fragment_25 : matrix_fragment_26;

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_52, matrix_accumulator_53, 0u, 0u);

    multiply_accumulate_half(packed_half_102, packed_half_103, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(packed_half_98, matrix_fragment_99, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_60 = !condition_2 ? matrix_fragment_26 : matrix_fragment_25;

    matrix_fragment_62 = !condition_2 ? matrix_fragment_27 : matrix_accumulator_88;

    matrix_fragment_61 = !condition_2 ? matrix_accumulator_88 : matrix_fragment_27;

    matrix_fragment_26 = !condition_3 ? matrix_fragment_5 : matrix_fragment_62;

    matrix_fragment_27 = !condition_3 ? matrix_fragment_60 : matrix_fragment_61;

    matrix_fragment_5 = !condition_3 ? matrix_fragment_62 : matrix_fragment_5;

    matrix_fragment_60 = !condition_3 ? matrix_fragment_61 : matrix_fragment_60;

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_26 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_26),
                                        select_half_operand<1, false, false>(matrix_fragment_26));

    matrix_fragment_26 = permute_signed(matrix_fragment_26, matrix_fragment_26, 0x5410u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_52, matrix_accumulator_53, 0u, 0u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x4100410u);

    matrix_fragment_5 = half_to_float_bits((matrix_fragment_26 & 65535u) * 65537u);

    matrix_fragment_26 = half_to_float_bits((matrix_fragment_26 >> 16) * 65537u);

    packed_half_162 = matrix_accumulator_140;

    matrix_fragment_5 = reciprocal_bits(matrix_fragment_5);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_27 = uint32_t(int32_t(packed_half_162) >> 0x1fu);

    matrix_fragment_26 = reciprocal_bits(matrix_fragment_26);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_25 = transpose_half_tile(matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_64 = uint32_t(int32_t(packed_half_162) >> 0x2u);

    matrix_fragment_27 =
        uint32_t(combine_words(matrix_fragment_27, 0u) << 0x5u >> 32) + matrix_fragment_64;

    matrix_fragment_27 = matrix_fragment_27 & 0xffffffe0u;

    matrix_accumulator_91 = pack_half(matrix_fragment_26, matrix_fragment_5);

    packed_half_162 = matrix_fragment_64 + (0u - matrix_fragment_27);

    matrix_fragment_5 = matrix_fragment_64 + 0x8u;

    matrix_fragment_24 = transpose_half_tile(matrix_fragment_24);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_91), "r"(packed_half_162), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    matrix_fragment_66 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    matrix_fragment_66 =
        uint32_t(combine_words(matrix_fragment_66, 0u) << 0x5u >> 32) + matrix_fragment_5;

    matrix_fragment_66 = matrix_fragment_66 & 0xffffffe0u;

    matrix_fragment_66 = matrix_fragment_5 + (0u - matrix_fragment_66);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_91), "r"(matrix_fragment_66), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_37 = permute_signed(matrix_fragment_26, matrix_fragment_26, 0x5410u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4c70ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_63 = transpose_half_tile(matrix_fragment_63);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_37));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = transpose_half_tile(matrix_fragment_62);

    packed_half_35 = permute_signed(matrix_fragment_66, matrix_fragment_66, 0x5410u);

    matrix_accumulator_89 = transpose_half_tile(matrix_accumulator_89);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_87 = transpose_half_tile(matrix_accumulator_88);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_92 = transpose_half_tile(matrix_fragment_60);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_90 = transpose_half_tile(matrix_fragment_61);

    matrix_accumulator_87 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(matrix_fragment_62));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4e70ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_35));

    matrix_accumulator_143 = transpose_half_tile(matrix_accumulator_143);

    matrix_accumulator_86 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_145 = transpose_half_tile(matrix_accumulator_145);

    matrix_accumulator_147 = transpose_half_tile(matrix_accumulator_147);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_accumulator_86));

    packed_half_153 = transpose_half_tile(packed_half_153);

    matrix_fragment_64 = transpose_half_tile(matrix_accumulator_142);

    matrix_fragment_66 = transpose_half_tile(matrix_accumulator_144);

    matrix_accumulator_86 = transpose_half_tile(matrix_accumulator_146);

    matrix_accumulator_152 = transpose_half_tile(matrix_accumulator_152);

    matrix_accumulator_143 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_145 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_147 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_89 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_89 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_fragment_64 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                      select_half_operand<2, false, false>(matrix_accumulator_143));

    matrix_fragment_66 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                      select_half_operand<2, false, false>(matrix_accumulator_145));

    matrix_accumulator_86 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                      select_half_operand<2, false, false>(matrix_accumulator_147));

    matrix_accumulator_88 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_152),
                      select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_143 = matrix_accumulator_140;

    multiply_accumulate_half(matrix_fragment_132, matrix_fragment_133, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_86, matrix_accumulator_87, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_88, matrix_accumulator_89, 0u, 0u);

    matrix_fragment_28 =
        uint32_t(int64_t(int32_t(matrix_accumulator_143)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_29 = uint32_t(int32_t(matrix_fragment_28) >> 0x1fu);

    matrix_fragment_30 =
        uint32_t(int64_t(int32_t(matrix_accumulator_143)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_29 =
        uint32_t(combine_words(matrix_fragment_29, 0u) << 0x5u >> 32) + matrix_fragment_28;

    matrix_fragment_31 = uint32_t(int32_t(matrix_fragment_30) >> 0x1fu);

    matrix_fragment_29 = matrix_fragment_29 & 0xffffffe0u;

    matrix_fragment_31 =
        uint32_t(combine_words(matrix_fragment_31, 0u) << 0x5u >> 32) + matrix_fragment_30;

    matrix_fragment_28 = matrix_fragment_28 + (0u - matrix_fragment_29);

    matrix_fragment_31 = matrix_fragment_31 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_91), "r"(matrix_fragment_28), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_28 = result_word;
    }

    matrix_fragment_30 = matrix_fragment_30 + (0u - matrix_fragment_31);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_52, matrix_accumulator_53, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_91), "r"(matrix_fragment_30), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_30 = result_word;
    }

    packed_half_101 = transpose_half_tile(packed_half_101);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    packed_half_109 = permute_signed(matrix_fragment_28, matrix_fragment_28, 0x5410u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_121 = permute_signed(matrix_fragment_30, matrix_fragment_30, 0x5410u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    packed_half_100 = transpose_half_tile(packed_half_100);

    matrix_fragment_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                             select_half_operand<2, false, false>(packed_half_121));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                          select_half_operand<2, false, false>(packed_half_37));

    packed_half_103 = transpose_half_tile(packed_half_103);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                       select_half_operand<2, false, false>(packed_half_107));

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_105 = transpose_half_tile(matrix_fragment_105);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_159 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_159),
                                          select_half_operand<2, false, false>(packed_half_37));

    packed_half_2 = transpose_half_tile(matrix_fragment_55);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_99 = transpose_half_tile(matrix_fragment_99);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    packed_half_106 = transpose_half_tile(matrix_accumulator_53);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_57 = transpose_half_tile(matrix_fragment_57);

    matrix_fragment_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_134),
                            select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_59 = transpose_half_tile(matrix_fragment_59);

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(packed_half_121));

    packed_half_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                   select_half_operand<2, false, false>(0u));

    packed_half_102 = transpose_half_tile(packed_half_102);

    matrix_fragment_104 = transpose_half_tile(matrix_fragment_104);

    matrix_fragment_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                             select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_97 = transpose_half_tile(matrix_accumulator_52);

    packed_half_101 = transpose_half_tile(matrix_fragment_54);

    packed_half_98 = transpose_half_tile(packed_half_98);

    matrix_fragment_56 = transpose_half_tile(matrix_fragment_56);

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_58);

    matrix_fragment_134 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_134),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_134));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_161 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_161),
                                          select_half_operand<2, false, false>(packed_half_35));

    packed_half_160 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_160),
                                          select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(packed_half_35));

    packed_half_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(packed_half_35));

    matrix_accumulator_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                          select_half_operand<2, false, false>(packed_half_37));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                        select_half_operand<2, false, false>(0u));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    packed_half_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                   select_half_operand<2, false, false>(0u));

    packed_half_160 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                    select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_88, matrix_accumulator_89, 0u, 0u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_accumulator_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                          select_half_operand<2, false, false>(packed_half_106));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    packed_half_158 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(packed_half_109));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_86, matrix_accumulator_87, 0u, 0u);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_159),
                                       select_half_operand<2, false, false>(packed_half_100));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                    select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_161),
                                       select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    packed_half_102 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                    select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                    select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                          select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_135),
                            select_half_operand<2, false, false>(packed_half_121));

    multiply_accumulate_half(matrix_fragment_132, matrix_fragment_133, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_132,
                             matrix_fragment_133);

    packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                   select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                  select_half_operand<2, false, false>(0u));

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_135),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_100, packed_half_101, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_102, packed_half_103, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_104, matrix_fragment_105, matrix_accumulator_140,
                             matrix_accumulator_141);

    matrix_fragment_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                             select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                             select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_158),
                                       select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_100, packed_half_101, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_104, matrix_fragment_105, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_102, packed_half_103, matrix_accumulator_52,
                             matrix_accumulator_53);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_132),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_133),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                   select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                  select_half_operand<2, false, false>(0u));

    packed_half_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                             select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_49 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                             select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_36));

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                            select_half_operand<2, false, false>(packed_half_34));

    matrix_accumulator_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                            select_half_operand<2, false, false>(packed_half_34));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_48,
                             matrix_fragment_49);

    packed_half_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_34));

    matrix_fragment_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                             select_half_operand<2, false, false>(packed_half_34));

    matrix_fragment_54 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(packed_half_37));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_52,
                             matrix_accumulator_53);

    packed_half_35 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                      select_half_operand<2, false, false>(matrix_accumulator_77),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                        select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                            select_half_operand<2, false, false>(packed_half_107));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                             select_half_operand<2, false, false>(packed_half_129));

    matrix_fragment_55 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                             select_half_operand<2, false, false>(packed_half_128));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                      select_half_operand<2, false, false>(matrix_accumulator_92),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_27 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                      select_half_operand<2, false, false>(matrix_accumulator_93),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                    select_half_operand<2, false, false>(packed_half_35));

    packed_half_37 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                    select_half_operand<2, false, false>(packed_half_37));

    matrix_fragment_54 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_55 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                        select_half_operand<2, false, false>(matrix_fragment_4));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_35), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_37), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_54), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_55), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_13) >> 0x1fu);

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_index_word_4));

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_12) >> 0x1fu);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_16) >> 0x1fu);

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_13;

    uniform_index_word_4 = 0u - uniform_index_word_4 + uniform_index_word_12;

    uniform_index_word_9 =
        uint32_t(combine_words(uniform_index_word_9, 0u) << 0x2u >> 32) + uniform_index_word_8;

    matrix_fragment_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                             select_half_operand<2, false, false>(packed_half_108));

    uniform_index_word_12 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x2u >> 32) + uniform_index_word_12;

    matrix_fragment_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                             select_half_operand<2, false, false>(packed_half_108));

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_16;

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                          select_half_operand<2, false, false>(packed_half_108));

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_10) >> 0x2u);

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                          select_half_operand<2, false, false>(packed_half_108));

    packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                    select_half_operand<2, false, false>(packed_half_2));

    packed_half_37 = add_half_pairs(select_half_operand<2, false, false>(packed_half_37),
                                    select_half_operand<2, false, false>(matrix_fragment_4));

    uniform_condition_0 = uint32_t(uniform_index_word_4) != 0u;

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_120));

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_9) >> 0x2u);

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                   select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_120));

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_12) >> 0x2u);

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_26,
                             matrix_fragment_27);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_35), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_98 = result_word;
    }

    uniform_condition_2 = int32_t(uniform_index_word_10) >= int32_t(uniform_index_word_11);

    uniform_condition_1 = int32_t(uniform_index_word_9) >= int32_t(uniform_index_word_12);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_37), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_96 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_111 = result_word;
    }

    matrix_accumulator_142 = threadIdx.x & 31u;

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_60, matrix_fragment_61, packed_half_114,
                             packed_half_115);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_2), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_109 = result_word;
    }

    uniform_condition_2 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x4u) && !uniform_condition_2;

    uniform_condition_1 =
        int32_t(uniform_index_word_16) > int32_t(0u - 0x4u) && !uniform_condition_1;

    if (uniform_condition_0) {
        uniform_index_word_4 = uniform_index_word_4 + 0xffffffffu;
    }

    uniform_condition_3 = uniform_condition_2 & uniform_condition_1;

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                             matrix_accumulator_123);

    if (uniform_condition_0) {
        uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);
    }

    uniform_index_word_13 = uniform_index_word_10 * uniform_index_word_12 + uniform_index_word_9;

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_124,
                             matrix_accumulator_125);

    if (uniform_condition_0) {
        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x3u >> 32) + uniform_index_word_4;
    }

    uniform_index_word_13 = uniform_index_word_13 << (0x7u & 31u);

    if (uniform_condition_0) {
        uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x3u);
    }

    uniform_index_word_4 = 0u;

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_13)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uniform_index_word_5 * uniform_block_index_18 + uniform_block_index_18;
    }
    if (uniform_condition_3) {
        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                           select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                          select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                          select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(matrix_accumulator_142)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_28, matrix_fragment_29));

        matrix_fragment_26 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29), matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);
    }
    uniform_index_word_5 = uniform_index_word_9 + 0x1u;

    split_address_words(packed_half_112, packed_half_113,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_0 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_12);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_0 =
        int32_t(uniform_index_word_16) > int32_t(0u - 0x8u) && !uniform_condition_0;

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_2 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(0u));

    packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                    select_half_operand<2, false, false>(packed_half_98));

    uniform_address_word_6 = parameters[parameter_output_low_2];
    uniform_address_word_7 = parameters[parameter_output_high_3];

    condition_4 = uniform_condition_2;

    if (condition_4) {
        matrix_fragment_27 = permute_signed(matrix_fragment_27, matrix_fragment_30, 0x5410u);

        split_address_words(packed_half_112, packed_half_113,
                            int64_t(int32_t(matrix_accumulator_142)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_112, packed_half_113));

        matrix_fragment_26 = permute_signed(matrix_fragment_26, matrix_fragment_31, 0x5410u);

        matrix_fragment_25 = permute_signed(matrix_fragment_25, matrix_fragment_28, 0x5410u);

        matrix_fragment_24 = permute_signed(matrix_fragment_24, matrix_fragment_29, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_112, packed_half_113) + 0x200ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3e60ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4460ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4660ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4060ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x3c60ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4860ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4260ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4a60ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    packed_half_98 = permute_signed(packed_half_35, packed_half_35, 0x5432u);

    packed_half_37 = add_half_pairs(select_half_operand<2, false, false>(packed_half_37),
                                    select_half_operand<2, false, false>(matrix_fragment_96));

    packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                    select_half_operand<2, false, false>(packed_half_98));

    packed_half_98 = permute_signed(packed_half_37, packed_half_37, 0x5432u);

    packed_half_37 = add_half_pairs(select_half_operand<2, false, false>(packed_half_37),
                                    select_half_operand<2, false, false>(packed_half_98));

    packed_half_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_35), 0x4100410u);

    packed_half_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_37), 0x4100410u);

    matrix_fragment_96 = half_to_float_bits((packed_half_35 & 65535u) * 65537u);

    packed_half_35 = half_to_float_bits((packed_half_35 >> 16) * 65537u);

    matrix_fragment_97 = half_to_float_bits((packed_half_37 & 65535u) * 65537u);

    packed_half_37 = half_to_float_bits((packed_half_37 >> 16) * 65537u);

    matrix_fragment_96 = rsqrt_bits(matrix_fragment_96);

    packed_half_35 = rsqrt_bits(packed_half_35);

    matrix_fragment_97 = rsqrt_bits(matrix_fragment_97);

    packed_half_98 = rsqrt_bits(packed_half_37);

    matrix_fragment_99 = pack_half(packed_half_35, matrix_fragment_96);

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_98 = pack_half(packed_half_98, matrix_fragment_97);

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(packed_half_98));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(packed_half_98));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(packed_half_98));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(packed_half_98));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                   select_half_operand<2, false, false>(0u));

    packed_half_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_77 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_74 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = permute_signed(matrix_accumulator_76, packed_half_35, 0x5410u);

    matrix_fragment_97 = permute_signed(matrix_accumulator_77, matrix_accumulator_32, 0x5410u);

    packed_half_98 = permute_signed(matrix_accumulator_84, packed_half_37, 0x5410u);

    matrix_fragment_99 = permute_signed(matrix_accumulator_85, matrix_accumulator_74, 0x5410u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(packed_half_111));

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(packed_half_109));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_150, matrix_fragment_151, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_25 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5432u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_25 = permute_signed(packed_half_2, packed_half_2, 0x5432u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x4100410u);

    packed_half_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_24 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    matrix_fragment_26 = half_to_float_bits((packed_half_2 & 65535u) * 65537u);

    packed_half_2 = half_to_float_bits((packed_half_2 >> 16) * 65537u);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    matrix_fragment_25 = rsqrt_bits(matrix_fragment_4);

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    matrix_fragment_27 = rsqrt_bits(packed_half_2);

    matrix_fragment_24 = pack_half(matrix_fragment_25, matrix_fragment_24);

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_27 = pack_half(matrix_fragment_27, matrix_fragment_26);

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                          select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<0, false, false>(packed_half_3));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<0, false, false>(packed_half_3));

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                          select_half_operand<0, false, false>(packed_half_3));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<0, false, false>(packed_half_3));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_91 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                  select_half_operand<2, false, false>(0u));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = permute_signed(matrix_fragment_24, matrix_fragment_25, 0x5410u);

    matrix_fragment_26 = permute_signed(matrix_fragment_26, matrix_fragment_27, 0x5410u);

    matrix_fragment_25 = permute_signed(matrix_accumulator_91, packed_half_2, 0x5410u);

    matrix_fragment_27 = permute_signed(packed_half_107, matrix_fragment_4, 0x5410u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_150, matrix_fragment_151, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_accumulator_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_60,
                             matrix_fragment_61);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_32 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3c203c20u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_29),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x3c203c20u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3c203c20u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_33 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3c203c20u);

    matrix_accumulator_32 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_96,
                             matrix_fragment_97, packed_half_98, matrix_fragment_99,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_fragment_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x3c203c20u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x3e473e47u);

    matrix_fragment_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_52 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3c203c20u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_accumulator_33 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3e473e47u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x3e473e47u);

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3e473e47u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 = (matrix_accumulator_32 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_39 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_40 = (matrix_fragment_28 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_43 = (matrix_fragment_30 << 0x5u) + 0x7ff88000u;

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3c203c20u);

    matrix_fragment_28 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    packed_half_37 = (matrix_accumulator_33 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_fragment_29 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_2 = (matrix_accumulator_52 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_33 = (matrix_fragment_54 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_32 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    packed_half_35 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_accumulator_53 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3c203c20u);

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3e473e47u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3c203c20u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_fragment_28 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_29 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3e473e47u);

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3e473e47u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_accumulator_38 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_fragment_31 << 0x5u) + 0x7ff88000u;

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    matrix_fragment_29 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_47 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_54 = (matrix_accumulator_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_44 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_45 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_37),
                       select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_fragment_46 = (matrix_fragment_24 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_55 = (matrix_fragment_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_56 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_52 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_53 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<1, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_57 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    matrix_fragment_26 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_28 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_49 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4c70ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    packed_half_3 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_61 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_58 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_59 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_28 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_29 = condition_0 ? matrix_fragment_31 : matrix_fragment_28;

    matrix_fragment_28 = condition_0 ? matrix_fragment_28 : matrix_fragment_31;

    matrix_fragment_30 = condition_0 ? matrix_fragment_68 : matrix_fragment_63;

    matrix_fragment_63 = condition_0 ? matrix_fragment_63 : matrix_fragment_68;

    matrix_fragment_69 = !condition_1 ? matrix_fragment_29 : matrix_fragment_30;

    matrix_fragment_68 = !condition_1 ? matrix_fragment_28 : matrix_fragment_63;

    matrix_fragment_71 = !condition_1 ? matrix_fragment_30 : matrix_fragment_29;

    matrix_fragment_70 = !condition_1 ? matrix_fragment_63 : matrix_fragment_28;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x4e70ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_69 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(float_bits_165), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(packed_half_164), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(packed_half_163), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_fragment_63 = !condition_2 ? matrix_fragment_69 : matrix_fragment_68;

    matrix_fragment_72 = !condition_2 ? matrix_fragment_68 : matrix_fragment_69;

    matrix_accumulator_74 = !condition_2 ? matrix_fragment_71 : matrix_fragment_70;

    matrix_fragment_73 = !condition_2 ? matrix_fragment_70 : matrix_fragment_71;

    matrix_accumulator_75 = !condition_3 ? matrix_fragment_63 : matrix_accumulator_74;

    matrix_accumulator_76 = !condition_3 ? matrix_fragment_72 : matrix_fragment_73;

    matrix_fragment_63 = !condition_3 ? matrix_accumulator_74 : matrix_fragment_63;

    matrix_fragment_72 = !condition_3 ? matrix_fragment_73 : matrix_fragment_72;

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_75),
                       select_half_operand<1, false, false>(matrix_accumulator_75));

    matrix_accumulator_75 = permute_signed(matrix_accumulator_75, matrix_accumulator_75, 0x5410u);

    matrix_accumulator_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75), 0x4100410u);

    packed_half_0 = half_to_float_bits((matrix_accumulator_75 & 65535u) * 65537u);

    matrix_accumulator_75 = half_to_float_bits((matrix_accumulator_75 >> 16) * 65537u);

    packed_half_0 = reciprocal_bits(packed_half_0);

    matrix_fragment_6 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    matrix_accumulator_75 = reciprocal_bits(matrix_accumulator_75);

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_fragment_6, 0u) << 0x5u >> 32) + matrix_fragment_5;

    matrix_fragment_69 =
        uint32_t(int64_t(int32_t(matrix_accumulator_143)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_6 = matrix_fragment_6 & 0xffffffe0u;

    matrix_fragment_63 =
        uint32_t(int64_t(int32_t(matrix_accumulator_143)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_68 = uint32_t(int32_t(matrix_fragment_69) >> 0x1fu);

    matrix_fragment_6 = matrix_fragment_5 + (0u - matrix_fragment_6);

    matrix_fragment_68 =
        uint32_t(combine_words(matrix_fragment_68, 0u) << 0x5u >> 32) + matrix_fragment_69;

    matrix_fragment_7 = pack_half(matrix_accumulator_75, packed_half_0);

    packed_half_0 = uint32_t(int32_t(matrix_fragment_63) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(packed_half_162), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    packed_half_0 = uint32_t(combine_words(packed_half_0, 0u) << 0x5u >> 32) + matrix_fragment_63;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_70 = matrix_fragment_68 & 0xffffffe0u;

    packed_half_0 = packed_half_0 & 0xffffffe0u;

    matrix_fragment_70 = matrix_fragment_69 + (0u - matrix_fragment_70);

    matrix_fragment_68 = matrix_fragment_63 + (0u - packed_half_0);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_70), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_68), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    packed_half_162 = permute_signed(packed_half_162, packed_half_162, 0x5410u);

    packed_half_0 = permute_signed(matrix_fragment_5, matrix_fragment_5, 0x5410u);

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_37),
                                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_63 = permute_signed(matrix_fragment_63, matrix_fragment_63, 0x5410u);

    matrix_fragment_70 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5410u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    packed_half_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                         select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_35),
                                   select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_86, matrix_accumulator_87, 0u, 0u);

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_88, matrix_accumulator_89, 0u, 0u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(packed_half_35));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, 0u, 0u);

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_88, matrix_accumulator_89, 0u, 0u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_accumulator_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_86, matrix_accumulator_87, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_53 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                      select_half_operand<2, false, false>(0u));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_104, matrix_fragment_105, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_62));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_100, packed_half_101, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_102, packed_half_103, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_100, packed_half_101, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_104, matrix_fragment_105, matrix_accumulator_88,
                             matrix_accumulator_89);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_64,
                             matrix_fragment_65);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_102, packed_half_103, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                            select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                            select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_36));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_0 = threadIdx.x;

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_11 = threadIdx.y;

    matrix_fragment_10 = threadIdx.z;

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_34));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(packed_half_34));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(packed_half_34));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(packed_half_34));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(packed_half_120));

    uniform_index_word_5 = uniform_index_word_10 + 0x1u;

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_26,
                             matrix_fragment_27, packed_half_2, packed_half_3);

    uniform_condition_2 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_11);

    uniform_condition_2 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x8u) && !uniform_condition_2;

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9);

    uniform_condition_1 = uniform_condition_2 & uniform_condition_1;

    uniform_index_word_10 = uniform_index_word_10 * uniform_index_word_12 + uniform_index_word_12;

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_14,
                             matrix_fragment_15);

    uniform_index_word_10 = uniform_index_word_9 + uniform_index_word_10;

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_22,
                             matrix_fragment_23);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_18,
                             matrix_fragment_19);

    uniform_index_word_10 = uniform_index_word_10 << (0x7u & 31u);

    condition_0 = (matrix_fragment_10 | matrix_fragment_11 | packed_half_0) != 0;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_10)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));
    if (uniform_condition_1) {
        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                           select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                          select_half_operand<2, false, false>(packed_half_0));

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_accumulator_142)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(matrix_fragment_13));

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_4, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_9, 0x5410u);

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_accumulator_142)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_24, matrix_fragment_11, 0x5410u);

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

    uniform_index_word_5 = uniform_block_index_17 + uniform_index_word_4;

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
extern "C" __global__
__maxnreg__(128) void phoenix_window_attention_channels_32_input_view(KernelArguments parameters) {
    run_window_attention_channels_32_input_view<false>(parameters.words, nullptr, nullptr);
}
