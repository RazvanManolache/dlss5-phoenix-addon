// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/query_key_value.cu
// 512-channel split-head stage. Separate projection/feed-forward modules use the same persistent feature layout.
// The filename identifies which stage is executed; wait/view variants preserve producer-consumer synchronization.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 9232
#include "../../../common/async_memory.cuh"
#include "../../../common/cached_vector_access.cuh"

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
    parameter_geometry_or_control_8 = 8,
    parameter_geometry_or_control_9 = 9,
    parameter_wait_optional_low_10 = 10,
    parameter_wait_optional_high_11 = 11,
    parameter_publication_low_12 = 12,
    parameter_publication_high_13 = 13,
};

struct alignas(8) KernelArguments {
    uint32_t words[14];
};
static_assert(sizeof(KernelArguments) == 56, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_query_key_value(
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
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t matrix_accumulator_16 = 0;
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
    uint32_t matrix_accumulator_28 = 0;
    uint32_t matrix_accumulator_29 = 0;
    uint32_t matrix_accumulator_30 = 0;
    uint32_t matrix_accumulator_31 = 0;
    uint32_t matrix_accumulator_32 = 0;
    uint32_t matrix_accumulator_33 = 0;
    uint32_t matrix_accumulator_34 = 0;
    uint32_t matrix_accumulator_35 = 0;
    uint32_t matrix_accumulator_36 = 0;
    uint32_t matrix_accumulator_37 = 0;
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
    uint32_t matrix_accumulator_54 = 0;
    uint32_t matrix_accumulator_55 = 0;
    uint32_t matrix_accumulator_56 = 0;
    uint32_t matrix_accumulator_57 = 0;
    uint32_t matrix_accumulator_58 = 0;
    uint32_t matrix_accumulator_59 = 0;
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
    uint32_t matrix_fragment_100 = 0;
    uint32_t matrix_fragment_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
    uint32_t matrix_fragment_106 = 0;
    uint32_t matrix_fragment_107 = 0;
    uint32_t matrix_fragment_108 = 0;
    uint32_t matrix_fragment_109 = 0;
    uint32_t matrix_fragment_110 = 0;
    uint32_t matrix_fragment_111 = 0;
    uint32_t matrix_fragment_112 = 0;
    uint32_t matrix_fragment_113 = 0;
    uint32_t matrix_fragment_114 = 0;
    uint32_t matrix_fragment_115 = 0;
    uint32_t matrix_fragment_116 = 0;
    uint32_t matrix_fragment_117 = 0;
    uint32_t matrix_fragment_118 = 0;
    uint32_t matrix_fragment_119 = 0;
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
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t matrix_accumulator_146 = 0;
    uint32_t matrix_accumulator_147 = 0;
    uint32_t matrix_accumulator_148 = 0;
    uint32_t matrix_accumulator_149 = 0;
    uint32_t matrix_accumulator_150 = 0;
    uint32_t matrix_accumulator_151 = 0;
    uint32_t matrix_accumulator_152 = 0;
    uint32_t matrix_accumulator_153 = 0;
    uint32_t matrix_accumulator_154 = 0;
    uint32_t matrix_accumulator_155 = 0;
    uint32_t matrix_accumulator_156 = 0;
    uint32_t matrix_accumulator_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t matrix_accumulator_159 = 0;
    uint32_t matrix_accumulator_160 = 0;
    uint32_t matrix_accumulator_161 = 0;
    uint32_t matrix_accumulator_162 = 0;
    uint32_t matrix_accumulator_163 = 0;
    uint32_t matrix_accumulator_164 = 0;
    uint32_t matrix_accumulator_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_block_index_9 = 0;
    uint32_t uniform_address_word_10 = 0;
    uint32_t uniform_address_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_index_word_15 = 0;
    uint32_t uniform_block_index_16 = 0;
    uint32_t uniform_index_word_17 = 0;
    uint32_t uniform_reused_word_18 = 0;
    uint32_t uniform_reused_word_19 = 0;
    uint32_t uniform_index_word_20 = 0;
    uint32_t uniform_index_word_21 = 0;
    uint32_t uniform_address_word_22 = 0;
    uint32_t uniform_address_word_23 = 0;
    uint32_t uniform_index_word_79 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    bool uniform_condition_2 = false;
    bool uniform_condition_3 = false;
    bool uniform_condition_4 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    packed_half_0 = threadIdx.x;

    matrix_fragment_10 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_11 = parameters[parameter_trained_weights_high_5];

    uniform_reused_word_18 = 0u;
    uniform_reused_word_19 = 0u;

    reused_word_1 = reused_word_1 + (0u - 0x40u);

    matrix_fragment_15 = threadIdx.y;

    uniform_index_word_12 = parameters[parameter_tensor_width_7];

    matrix_accumulator_2 = blockIdx.z;

    matrix_accumulator_17 = threadIdx.x & 31u;

    uniform_index_word_7 = blockIdx.y;

    uniform_block_index_16 = parameters[parameter_tensor_height_6];

    packed_half_0 = matrix_fragment_15 | packed_half_0;
    condition_0 = packed_half_0 != 0;

    matrix_accumulator_2 = matrix_accumulator_2 * 0x4u + matrix_fragment_15;

    matrix_accumulator_2 = matrix_accumulator_2 * 0x60u;

    if (!condition_0) {
        matrix_accumulator_3 = blockDim.x;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    uniform_condition_1 = uniform_condition_0;

    if (!uniform_condition_0) {
        uniform_index_word_5 = 0x400u;
    }

    if (!condition_0) {
        matrix_fragment_4 = blockDim.y;
    }

    if (!uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_5 + 0x2000u;

        uniform_index_word_6 = 0u;
    }

    if (!condition_0) {
        matrix_fragment_4 = matrix_accumulator_3 * matrix_fragment_4;
    }

    matrix_accumulator_3 = matrix_accumulator_2 * 0x8u;

    uniform_index_word_4 =
        broadcast_warp_word(matrix_fragment_4, !condition_0, uniform_index_word_4);

    matrix_fragment_5 = matrix_accumulator_3 + 0x80u;

    if (!uniform_condition_0) {
        uniform_index_word_6 = (uniform_index_word_6 << 0x18u) + uniform_index_word_5;
    }

    matrix_fragment_7 = matrix_accumulator_3 + 0x100u;

    matrix_fragment_9 = matrix_accumulator_3 + 0x200u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_accumulator_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    if (!uniform_condition_0) {
        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x200u);

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x200u);

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_6);
#endif
    }

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_6 + 0x8u);
#endif
    }

    __syncthreads();

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    uniform_index_word_4 = parameters[parameter_geometry_or_control_8];
    uniform_index_word_5 = parameters[parameter_geometry_or_control_9];

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                        matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0xc000ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0xc000ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xc000ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                        matrix_accumulator_52, matrix_accumulator_53,
                                        matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0xc000ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_12, matrix_accumulator_13),
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0xc000ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                        matrix_fragment_84, matrix_fragment_85, matrix_fragment_86,
                                        matrix_fragment_87, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0xc000ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_accumulator_3 = threadIdx.z;

    uniform_block_index_9 = blockIdx.x;

    uniform_condition_0 = uint32_t(uniform_index_word_12) != 0u;

    uniform_condition_1 = uint32_t(uniform_block_index_16) != 0u;

    uniform_address_word_11 = min(int32_t(0u), int32_t(uniform_index_word_4));

    uniform_index_word_7 = (uniform_index_word_7 << 0x3u) + uniform_index_word_5;

    uniform_index_word_13 = 0u - uniform_address_word_11 + uniform_index_word_12;

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_7) >> 0x1fu);

    uniform_index_word_14 = uniform_index_word_7 + 0x7u;

    if (uniform_condition_0) {
        uniform_address_word_11 = uniform_index_word_12 + 0xffffffffu;
    }

    if (uniform_condition_1) {
        uniform_index_word_12 = uniform_block_index_16 + 0xffffffffu;
    }

    uniform_condition_3 = int32_t(uniform_index_word_7) >= int32_t(0u - 0x7u);

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_index_word_14, 0u) << 0x1du >> 32) + 0x1u;

    uniform_index_word_6 = 0u;

    uniform_block_index_9 = (uniform_block_index_9 << 0x3u) + uniform_index_word_4;

    condition_0 = (matrix_accumulator_3 | packed_half_0) != 0;

    packed_half_0 = max(int32_t(0u), int32_t(uniform_index_word_7));

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_index_word_7;

    if (uniform_condition_1) {
        uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_12) >> 0x1fu);
    }

    uniform_index_word_15 = uniform_block_index_9 + 0x7u;

    if (uniform_condition_1) {
        uniform_index_word_8 =
            uint32_t(combine_words(uniform_index_word_8, 0u) << 0x3u >> 32) + uniform_index_word_12;
    }

    uniform_condition_2 = uint32_t(uniform_index_word_13) != 0u;

    uniform_block_index_16 =
        uint32_t(combine_words(uniform_index_word_15, 0u) << 0x1du >> 32) + 0x1u;

    uniform_index_word_15 = uniform_condition_3 ? uniform_index_word_14 : 0u;

    if (uniform_condition_1) {
        uniform_index_word_6 =
            uint32_t(int64_t(int32_t(uniform_index_word_8)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0x3u);

    uniform_index_word_8 = min(int32_t(uniform_index_word_15), int32_t(uniform_index_word_6));

    if (uniform_condition_2) {
        uniform_index_word_13 = uniform_index_word_13 + 0xffffffffu;
    }

    condition_0 = int32_t(packed_half_0) >= int32_t(uniform_index_word_8) || condition_0;

    uniform_address_word_10 = uint32_t(int32_t(uniform_block_index_9) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_11) >> 0x1fu);
    }

    if (uniform_condition_2) {
        uniform_index_word_14 = uint32_t(int32_t(uniform_index_word_13) >> 0x1fu);
    }

    uniform_condition_4 = int32_t(uniform_block_index_9) >= int32_t(0u - 0x7u);

    uniform_address_word_10 =
        uint32_t(combine_words(uniform_address_word_10, 0u) << 0x2u >> 32) + uniform_block_index_9;

    if (uniform_condition_0) {
        uniform_index_word_12 = uint32_t(combine_words(uniform_index_word_12, 0u) << 0x3u >> 32) +
                                uniform_address_word_11;
    }

    if (uniform_condition_2) {
        uniform_index_word_14 = uint32_t(combine_words(uniform_index_word_14, 0u) << 0x3u >> 32) +
                                uniform_index_word_13;
    }

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0u;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_7) >> 0x2u);

    uniform_index_word_6 = uint32_t(int32_t(uniform_address_word_10) >> 0x2u);

    uniform_block_index_16 = uniform_condition_4 ? uniform_block_index_16 : 0u;

    if (uniform_condition_0) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_12)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_2) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_14)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_2 = max(int32_t(0u), int32_t(uniform_block_index_9));
    if (!condition_0) {
        matrix_fragment_6 = uniform_block_index_16;

        matrix_fragment_9 = uniform_index_word_5;

        matrix_fragment_4 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

        matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_5));

        matrix_fragment_9 = matrix_fragment_9 * 0x2u;
        do {
            condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_6);
            if (!condition_0) {
                matrix_fragment_8 = matrix_fragment_9 * packed_half_0;

                matrix_fragment_5 = matrix_fragment_4;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_10];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_11];

                    matrix_fragment_7 = matrix_fragment_8 + matrix_fragment_5;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_7 = pipeline_read<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_6);

                    condition_0 = int32_t(matrix_fragment_7) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_7 = pipeline_read<CheckBounds>(
                                combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                memory_ranges, error_counts);

                            condition_0 = int32_t(matrix_fragment_7) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            packed_half_0 = packed_half_0 + 0x1u;

            condition_0 = uint32_t(packed_half_0) != uint32_t(uniform_index_word_8);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    matrix_accumulator_16 = parameters[parameter_tensor_height_6];
    matrix_accumulator_17 = parameters[parameter_tensor_width_7];

    matrix_accumulator_3 = uint32_t(uint32_t(matrix_fragment_15) >> 0x2u);

    __syncthreads();

    condition_0 = true;

    matrix_fragment_6 = matrix_accumulator_3 + uniform_index_word_7;

    matrix_accumulator_2 = permute_signed(0u, matrix_accumulator_2, 0x7610u);

    uniform_index_word_17 = 0u;

    packed_half_0 = matrix_accumulator_16 & 0xfffffffcu;

    matrix_fragment_5 = uint32_t(int32_t(matrix_accumulator_16) >> 0x1fu);

    matrix_fragment_4 = uint32_t(int32_t(matrix_accumulator_17) >> 0x1fu);

    condition_1 = uint32_t(packed_half_0) != uint32_t(0x4u);

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_accumulator_16;

    packed_half_0 = uint32_t(uint32_t(matrix_fragment_15) >> 0x1u);

    matrix_accumulator_13 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    packed_half_0 = packed_half_0 & 0x1u;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x2u >> 32) + matrix_accumulator_17;

    matrix_fragment_4 = 0u;

    matrix_fragment_7 = packed_half_0 + uniform_index_word_6;

    if (condition_1) {
        condition_2 = int32_t(matrix_fragment_6) >= int32_t(matrix_accumulator_13);
    }

    matrix_accumulator_12 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    packed_half_0 = matrix_fragment_15 * 0x80u;

    matrix_fragment_5 = matrix_fragment_7;

    if (condition_1) {
        condition_0 = int32_t(matrix_fragment_6) > -1 && !condition_2;

        matrix_fragment_4 = matrix_fragment_6 * matrix_accumulator_12;

        matrix_fragment_8 = condition_0 ? 0u : 0x1u;

        matrix_accumulator_2 = permute_signed(matrix_fragment_8, matrix_accumulator_2, 0x7610u);
    }
    if (condition_0) {
        matrix_fragment_5 = matrix_accumulator_17 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_fragment_5) != uint32_t(0x4u);

        matrix_fragment_5 = 0u;

        if (condition_2) {
            condition_3 = int32_t(matrix_fragment_7) >= int32_t(matrix_accumulator_12);
            condition_0 = int32_t(matrix_fragment_7) > -1;

            matrix_fragment_5 = matrix_fragment_7;

            matrix_fragment_8 = !condition_3 ? matrix_accumulator_2 : 0x1u;

            matrix_fragment_8 = condition_0 ? matrix_fragment_8 : 0x1u;

            matrix_accumulator_2 = permute_signed(matrix_fragment_8, matrix_accumulator_2, 0x7610u);
        }
    }
    matrix_fragment_8 = matrix_accumulator_2 & 0xffffu;

    uniform_index_word_20 = 0x400u;

    matrix_fragment_4 = matrix_fragment_5 + matrix_fragment_4;

    matrix_fragment_5 = packed_half_0 & 0x80u;

    uniform_index_word_21 = (uniform_index_word_17 << 0x18u) + uniform_index_word_20;

    packed_half_0 = packed_half_0 & 0x100u;

    condition_2 = uint32_t(matrix_fragment_8) != 0u;

    matrix_accumulator_2 = matrix_fragment_4 * 0x800u + matrix_fragment_5;

    condition_0 = uint32_t(matrix_fragment_8) != 0u;

    matrix_fragment_4 = matrix_accumulator_3 * 0x200u + packed_half_0;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);

    matrix_fragment_4 = matrix_fragment_4 + matrix_fragment_5;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        !condition_0 ? combine_words(matrix_accumulator_2, matrix_accumulator_3)
                                     : combine_words(0u, 0u));

    matrix_fragment_4 = (matrix_fragment_4 << 0x2u) + uniform_index_word_21;
    if (condition_2) {
        matrix_accumulator_3 = threadIdx.x & 31u;

        uniform_block_index_9 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = uniform_block_index_9;

        matrix_fragment_9 = uniform_block_index_9;

        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = matrix_fragment_8;

        matrix_fragment_10 = matrix_fragment_8;

        matrix_fragment_11 = matrix_fragment_8;

        matrix_fragment_4 = matrix_accumulator_3 * 0x10u + matrix_fragment_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_4, matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_block_index_9 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            matrix_fragment_8 = parameters[parameter_input_low_0];
            matrix_fragment_9 = parameters[parameter_input_high_1];

            uniform_index_word_12 = broadcast_active_lane(matrix_fragment_4);

            uniform_block_index_9 = uniform_index_word_20 + 0x2000u;

            uniform_index_word_13 = (uniform_index_word_17 << 0x18u) + uniform_block_index_9;

            uniform_block_index_9 = 0x20u;

            const uint64_t byte_address_0 =
                scaled_word_address(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
            matrix_fragment_8 = uint32_t(byte_address_0);

            matrix_accumulator_2 = uint32_t(byte_address_0 >> 32);

            uniform_address_word_10 = broadcast_active_lane(matrix_fragment_8);

            uniform_address_word_11 = broadcast_active_lane(matrix_accumulator_2);

            matrix_accumulator_2 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_index_word_12,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_block_index_9 * 16u, uniform_index_word_13, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_index_word_12,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_block_index_9 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_2 = matrix_fragment_15 + 0x4u;

    condition_0 = true;

    matrix_fragment_8 = 0u;

    matrix_fragment_9 = matrix_fragment_7;

    matrix_accumulator_3 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x2u);

    matrix_accumulator_2 = permute_signed(0u, matrix_accumulator_2, 0x7610u);

    matrix_fragment_4 = matrix_accumulator_3 + uniform_index_word_7;

    if (condition_1) {
        condition_2 = int32_t(matrix_fragment_4) >= int32_t(matrix_accumulator_13);

        matrix_fragment_8 = matrix_fragment_4 * matrix_accumulator_12;

        condition_0 = int32_t(matrix_fragment_4) > -1 && !condition_2;

        matrix_fragment_10 = condition_0 ? 0u : 0x1u;

        matrix_accumulator_2 = permute_signed(matrix_fragment_10, matrix_accumulator_2, 0x7610u);
    }
    if (condition_0) {
        matrix_accumulator_17 = matrix_accumulator_17 & 0xfffffffcu;

        matrix_fragment_9 = 0u;

        condition_0 = uint32_t(matrix_accumulator_17) != uint32_t(0x4u);

        if (condition_0) {
            condition_2 = int32_t(matrix_fragment_7) >= int32_t(matrix_accumulator_12);

            condition_1 = int32_t(matrix_fragment_7) > -1;

            matrix_fragment_9 = matrix_fragment_7;

            matrix_fragment_10 = !condition_2 ? matrix_accumulator_2 : 0x1u;

            matrix_fragment_10 = condition_1 ? matrix_fragment_10 : 0x1u;

            matrix_accumulator_2 =
                permute_signed(matrix_fragment_10, matrix_accumulator_2, 0x7610u);
        }
    }
    matrix_fragment_7 = matrix_accumulator_2 & 0xffffu;

    packed_half_0 = matrix_accumulator_3 * 0x200u + packed_half_0;

    matrix_fragment_8 = matrix_fragment_9 + matrix_fragment_8;

    condition_1 = uint32_t(matrix_fragment_7) != 0u;

    condition_0 = uint32_t(matrix_fragment_7) != 0u;

    matrix_accumulator_2 = matrix_fragment_8 * 0x800u + matrix_fragment_5;

    packed_half_0 = matrix_fragment_5 + packed_half_0;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);

    packed_half_0 = (packed_half_0 << 0x2u) + uniform_index_word_21;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        !condition_0 ? combine_words(matrix_accumulator_2, matrix_accumulator_3)
                                     : combine_words(0u, 0u));
    if (condition_1) {
        matrix_accumulator_3 = threadIdx.x & 31u;

        uniform_block_index_9 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = uniform_block_index_9;

        matrix_fragment_5 = uniform_block_index_9;

        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_5, 0x5410u);

        matrix_fragment_9 = matrix_fragment_8;

        matrix_fragment_10 = matrix_fragment_8;

        matrix_fragment_11 = matrix_fragment_8;

        packed_half_0 = matrix_accumulator_3 * 0x10u + packed_half_0;

        store_four_shared_words<CheckBounds>(
            shared_tile, packed_half_0, matrix_fragment_8, matrix_fragment_9, matrix_fragment_10,
            matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_block_index_9 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            matrix_fragment_8 = parameters[parameter_input_low_0];
            matrix_fragment_9 = parameters[parameter_input_high_1];

            uniform_index_word_12 = broadcast_active_lane(packed_half_0);

            uniform_block_index_9 = uniform_index_word_20 + 0x2000u;

            packed_half_0 = 0x200u;

            uniform_index_word_13 = (uniform_index_word_17 << 0x18u) + uniform_block_index_9;

            uniform_block_index_9 = 0x20u;

            const uint64_t byte_address_1 =
                scaled_word_address(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
            matrix_fragment_5 = uint32_t(byte_address_1);

            matrix_accumulator_2 = uint32_t(byte_address_1 >> 32);

            uniform_address_word_10 = broadcast_active_lane(matrix_fragment_5);

            uniform_address_word_11 = broadcast_active_lane(matrix_accumulator_2);

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_index_word_12,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_block_index_9 * 16u, uniform_index_word_13, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_index_word_12,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_block_index_9 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_fragment_5 = 0u;

    uniform_block_index_9 = uniform_index_word_20 + 0x2000u;

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(matrix_accumulator_13);

    condition_1 = int32_t(matrix_fragment_4) >= int32_t(matrix_accumulator_13);

    packed_half_0 = uniform_block_index_9;

    condition_0 = int32_t(matrix_fragment_6) < int32_t(0u) || condition_0;

    condition_1 = int32_t(matrix_fragment_4) < int32_t(0u) || condition_1;

    matrix_fragment_5 = (matrix_fragment_5 << 0x18u) + packed_half_0;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_fragment_5;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_2 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_2 = wait_copy_barrier(shared_tile, matrix_fragment_5,
                                    async_tokens[(((matrix_fragment_5) >> 3) & 1)]);
#else
    __syncthreads();
    condition_2 = true;
#endif
    if (!condition_2) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_2 = wait_copy_barrier(shared_tile, matrix_fragment_5,
                                            async_tokens[(((matrix_fragment_5) >> 3) & 1)]);
#else
            __syncthreads();
            condition_2 = true;
#endif
        } while (!condition_2);
    }
    matrix_accumulator_2 = 0x1u;

    packed_half_0 = 0u;

    matrix_accumulator_162 = 0;
    matrix_accumulator_163 = 0;

    matrix_accumulator_160 = 0;
    matrix_accumulator_161 = 0;

    packed_half_158 = 0;
    matrix_accumulator_159 = 0;

    thread_scratch[12] = matrix_accumulator_2;

    matrix_accumulator_156 = 0;
    matrix_accumulator_157 = 0;

    matrix_accumulator_154 = 0;
    matrix_accumulator_155 = 0;

    matrix_accumulator_152 = 0;
    matrix_accumulator_153 = 0;

    thread_scratch[11] = 0u;

    matrix_accumulator_150 = 0;
    matrix_accumulator_151 = 0;

    matrix_accumulator_148 = 0;
    matrix_accumulator_149 = 0;

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    thread_scratch[10] = 0u;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_146 = 0;
    matrix_accumulator_147 = 0;

    thread_scratch[9] = 0u;

    matrix_accumulator_144 = 0;
    matrix_accumulator_145 = 0;

    matrix_accumulator_142 = 0;
    matrix_accumulator_143 = 0;

    matrix_accumulator_140 = 0;
    matrix_accumulator_141 = 0;

    thread_scratch[8] = 0u;

    matrix_accumulator_132 = 0;
    matrix_accumulator_133 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    thread_scratch[7] = 0u;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

    thread_scratch[6] = 0u;

    matrix_accumulator_38 = 0;
    matrix_accumulator_39 = 0;

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    thread_scratch[5] = 0u;

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    thread_scratch[4] = 0u;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_accumulator_12 = 0;
    matrix_accumulator_13 = 0;

    thread_scratch[3] = 0u;

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    matrix_accumulator_18 = 0;
    matrix_accumulator_19 = 0;

    thread_scratch[2] = 0u;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    thread_scratch[1] = 0u;

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_accumulator_30 = 0;
    matrix_accumulator_31 = 0;

    thread_scratch[0] = 0u;

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_accumulator_34 = 0;
    matrix_accumulator_35 = 0;

    matrix_accumulator_164 = 0;
    matrix_accumulator_165 = 0;

#pragma unroll 1
    do {
        matrix_fragment_90 = thread_scratch[12];

        matrix_fragment_88 = packed_half_0 + 0x80u;

        matrix_fragment_89 = matrix_fragment_90 * 0x100u;

        condition_2 = uint32_t(matrix_fragment_89) >= uint32_t(matrix_fragment_88);

        condition_2 = uint32_t(matrix_fragment_90) > uint32_t(0x1u) || condition_2;
        if (!condition_2) {
            uniform_block_index_9 = blockIdx.y;

            matrix_fragment_88 = threadIdx.x;

            uniform_address_word_10 = parameters[parameter_geometry_or_control_9];

            matrix_fragment_89 = threadIdx.y;

            matrix_fragment_90 = threadIdx.z;

            uniform_block_index_9 = (uniform_block_index_9 << 0x3u) + uniform_address_word_10;

            matrix_fragment_91 = max(int32_t(0u), int32_t(uniform_block_index_9));

            condition_2 = (matrix_fragment_88 | matrix_fragment_89 | matrix_fragment_90) != 0;

            matrix_fragment_90 = uint32_t(uint32_t(matrix_fragment_91) >> 0x3u);

            condition_2 =
                int32_t(matrix_fragment_90) >= int32_t(uniform_index_word_8) || condition_2;
            if (!condition_2) {
                do {
                    uniform_block_index_9 = blockIdx.x;

                    uniform_address_word_10 = parameters[parameter_geometry_or_control_8];

                    uniform_block_index_9 =
                        (uniform_block_index_9 << 0x3u) + uniform_address_word_10;

                    uniform_address_word_10 = uniform_block_index_9 + 0x7u;

                    uniform_condition_0 = int32_t(uniform_block_index_9) >= int32_t(0u - 0x7u);

                    uniform_block_index_9 = max(int32_t(0u), int32_t(uniform_block_index_9));

                    uniform_address_word_10 =
                        uint32_t(combine_words(uniform_address_word_10, 0u) << 0x1du >> 32) + 0x1u;

                    uniform_block_index_9 = uint32_t(uint32_t(uniform_block_index_9) >> 0x3u);

                    uniform_address_word_10 = uniform_condition_0 ? uniform_address_word_10 : 0u;

                    uniform_address_word_10 =
                        min(int32_t(uniform_address_word_10), int32_t(uniform_index_word_5));

                    uniform_condition_0 =
                        int32_t(uniform_block_index_9) >= int32_t(uniform_address_word_10);
                    if (!uniform_condition_0) {
                        matrix_fragment_88 = matrix_fragment_90 * uniform_index_word_5;

                        matrix_fragment_91 = uniform_block_index_9;

                        matrix_fragment_92 = (matrix_fragment_88 << 0x1u) + uniform_index_word_5;
                        do {
                            matrix_fragment_88 = parameters[parameter_wait_optional_low_10];
                            matrix_fragment_89 = parameters[parameter_wait_optional_high_11];

                            matrix_fragment_93 = matrix_fragment_92 + matrix_fragment_91;

                            split_address_words(
                                matrix_fragment_88, matrix_fragment_89,
                                int64_t(int32_t(matrix_fragment_93)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_88, matrix_fragment_89));

                            matrix_fragment_93 = pipeline_read<CheckBounds>(
                                combine_words(matrix_fragment_88, matrix_fragment_89),
                                memory_ranges, error_counts);

                            condition_2 = int32_t(matrix_fragment_93) > -1;
                            if (!condition_2) {
                                do {
                                    __nanosleep(0x40u);

                                    matrix_fragment_93 = pipeline_read<CheckBounds>(
                                        combine_words(matrix_fragment_88, matrix_fragment_89),
                                        memory_ranges, error_counts);

                                    condition_2 = int32_t(matrix_fragment_93) > -1;
                                } while (!condition_2);
                            }
                            uniform_block_index_9 = blockIdx.x;

                            uniform_address_word_10 = parameters[parameter_geometry_or_control_8];

                            matrix_fragment_91 = matrix_fragment_91 + 0x1u;

                            uniform_block_index_9 =
                                (uniform_block_index_9 << 0x3u) + uniform_address_word_10;

                            uniform_address_word_10 = uniform_block_index_9 + 0x7u;

                            uniform_condition_0 =
                                int32_t(uniform_block_index_9) >= int32_t(0u - 0x7u);

                            uniform_address_word_10 =
                                uint32_t(combine_words(uniform_address_word_10, 0u) << 0x1du >>
                                         32) +
                                0x1u;

                            uniform_address_word_10 =
                                uniform_condition_0 ? uniform_address_word_10 : 0u;

                            uniform_address_word_10 = min(int32_t(uniform_address_word_10),
                                                          int32_t(uniform_index_word_5));

                            condition_2 =
                                uint32_t(matrix_fragment_91) != uint32_t(uniform_address_word_10);
                        } while (condition_2);
                    }
                    matrix_fragment_90 = matrix_fragment_90 + 0x1u;

                    condition_2 = uint32_t(matrix_fragment_90) != uint32_t(uniform_index_word_8);
                } while (condition_2);
            }
            matrix_fragment_88 = 0x2u;

            __syncwarp(__activemask());

            __syncthreads();

            thread_scratch[12] = matrix_fragment_88;
        }
        matrix_fragment_92 = threadIdx.y;

        matrix_fragment_91 = parameters[parameter_tensor_width_7];

        matrix_fragment_88 = parameters[parameter_tensor_height_6];
        matrix_fragment_89 = parameters[parameter_tensor_width_7];

        matrix_fragment_94 = uint32_t(int32_t(matrix_fragment_91) >> 0x1fu);

        matrix_fragment_95 =
            uint32_t(combine_words(matrix_fragment_94, 0u) << 0x2u >> 32) + matrix_fragment_91;

        matrix_fragment_90 = uint32_t(uint32_t(matrix_fragment_92) >> 0x1u);

        matrix_fragment_93 = matrix_fragment_88 & 0xfffffffcu;

        matrix_fragment_90 = matrix_fragment_90 & 0x1u;

        matrix_fragment_95 = uint32_t(int32_t(matrix_fragment_95) >> 0x2u);

        matrix_fragment_96 = matrix_fragment_89 & 0xfffffffcu;

        matrix_fragment_94 = matrix_fragment_90 + uniform_index_word_6;

        condition_2 = uint32_t(matrix_fragment_93) != uint32_t(0x4u) && condition_0;

        condition_3 = uint32_t(matrix_fragment_93) == uint32_t(0x4u) || !condition_0;

        split_address_words(matrix_fragment_90, matrix_fragment_91, 0ull);

        condition_5 = int32_t(matrix_fragment_94) >= int32_t(matrix_fragment_95);

        condition_4 = uint32_t(matrix_fragment_96) == uint32_t(0x4u) || condition_2;

        condition_2 = int32_t(matrix_fragment_94) > -1 && !condition_5;

        condition_3 = condition_3 & (condition_4 | condition_2);
        if (condition_3) {
            condition_4 = uint32_t(matrix_fragment_93) != uint32_t(0x4u);

            matrix_fragment_88 = condition_0 ? matrix_fragment_94 : 0u;

            condition_5 = uint32_t(matrix_fragment_96) != uint32_t(0x4u);

            matrix_fragment_91 = matrix_fragment_92 & 0x1u;

            matrix_fragment_89 = condition_4 ? matrix_fragment_88 : 0u;

            matrix_fragment_88 = uint32_t(combine_words(matrix_fragment_92, 0u) << 0x1eu >> 32) +
                                 uniform_index_word_7;

            matrix_fragment_89 = !condition_5 ? matrix_fragment_89 : matrix_fragment_94;

            if (condition_4) {
                matrix_fragment_89 = matrix_fragment_88 * matrix_fragment_95 + matrix_fragment_89;
            }

            matrix_fragment_88 = packed_half_0 + 0x40u;

            matrix_fragment_88 =
                uint32_t(combine_words(matrix_fragment_88, 0u) << 0x1bu >> 32) + matrix_fragment_91;

            matrix_fragment_88 = matrix_fragment_89 * 0x10u + matrix_fragment_88;

            matrix_fragment_90 = matrix_fragment_88 * 0x80u;

            matrix_fragment_91 = uint32_t(int32_t(matrix_fragment_90) >> 0x1fu);
        }
        if (!condition_3) {
            uniform_address_word_10 = 0u;

            matrix_fragment_97 = threadIdx.x & 31u;

            uniform_block_index_9 = 0x400u;

            matrix_fragment_88 = packed_half_0 * 0x40u;

            matrix_fragment_88 = ~matrix_fragment_88 & 0x1000u;

            uniform_block_index_9 = (uniform_address_word_10 << 0x18u) + uniform_block_index_9;

            uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                    select_half_operand<2, false, false>(0u));

            matrix_fragment_89 = (matrix_fragment_92 << 0x9u) + uniform_block_index_9;

            matrix_fragment_90 = matrix_fragment_88 + matrix_fragment_89;

            matrix_fragment_88 = uniform_address_word_10;

            matrix_fragment_89 = uniform_address_word_10;

            matrix_fragment_97 = matrix_fragment_97 * 0x10u + matrix_fragment_90;

            matrix_fragment_88 = permute_signed(matrix_fragment_88, matrix_fragment_89, 0x5410u);

            matrix_fragment_89 = matrix_fragment_88;

            matrix_fragment_90 = matrix_fragment_88;

            matrix_fragment_91 = matrix_fragment_88;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_97, matrix_fragment_88, matrix_fragment_89,
                matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_block_index_9 = 0xffffffffu;

            condition_3 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_3) {
                matrix_fragment_88 = parameters[parameter_input_low_0];
                matrix_fragment_89 = parameters[parameter_input_high_1];

                uniform_block_index_9 = broadcast_active_lane(packed_half_0);

                uniform_address_word_10 = 0x400u;

                uniform_index_word_14 = broadcast_active_lane(matrix_fragment_92);

                uniform_index_word_13 = uniform_address_word_10 + 0x2000u;

                uniform_address_word_11 = 0u;

                uniform_index_word_12 = uint32_t(uint32_t(uniform_block_index_9) >> 0x6u);

                uniform_block_index_9 = uniform_block_index_9 << (0x6u & 31u);

                {
                    uint32_t result_word;
                    result_word = uniform_index_word_12 & 0x1u;
                    uniform_condition_0 = result_word != 0;
                }

                uniform_block_index_9 = ~uniform_block_index_9 & 0x1000u;

                const uint64_t byte_address_2 =
                    scaled_word_address(combine_words(matrix_fragment_88, matrix_fragment_89),
                                        combine_words(matrix_fragment_90, matrix_fragment_91));
                matrix_fragment_88 = uint32_t(byte_address_2);

                matrix_fragment_90 = uint32_t(byte_address_2 >> 32);

                uniform_address_word_22 = broadcast_active_lane(matrix_fragment_88);

                uniform_index_word_12 =
                    (uniform_address_word_11 << 0x18u) + uniform_address_word_10;

                uniform_address_word_23 = broadcast_active_lane(matrix_fragment_90);

                uniform_address_word_11 =
                    (uniform_address_word_11 << 0x18u) + uniform_index_word_13;

                matrix_fragment_88 = 0x200u;

                uniform_address_word_10 = (uniform_index_word_14 << 0x9u) + uniform_index_word_12;

                if (!uniform_condition_0) {
                    uniform_address_word_11 = uniform_address_word_11 + 0x8u;
                }

                uniform_address_word_10 = uniform_block_index_9 + uniform_address_word_10;

                uniform_block_index_9 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_22, uniform_address_word_23),
                    uniform_block_index_9 * 16u, uniform_address_word_11, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_22, uniform_address_word_23),
                    uniform_block_index_9 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        condition_3 = uint32_t(matrix_fragment_93) != uint32_t(0x4u) && condition_1;

        condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x4u) || !condition_1;

        split_address_words(matrix_fragment_90, matrix_fragment_91, 0ull);

        condition_3 = uint32_t(matrix_fragment_96) == uint32_t(0x4u) || condition_3;

        condition_3 = condition_4 & (condition_3 | condition_2);
        if (condition_3) {
            condition_4 = uint32_t(matrix_fragment_96) != uint32_t(0x4u);

            condition_2 = uint32_t(matrix_fragment_93) != uint32_t(0x4u);

            matrix_fragment_89 = condition_1 ? matrix_fragment_94 : 0u;

            matrix_fragment_88 = matrix_fragment_92 + 0x4u;

            matrix_fragment_90 = matrix_fragment_92 & 0x1u;

            matrix_fragment_88 = uint32_t(combine_words(matrix_fragment_88, 0u) << 0x1eu >> 32) +
                                 uniform_index_word_7;

            if (!condition_4) {
                matrix_fragment_94 = condition_2 ? matrix_fragment_89 : 0u;
            }

            matrix_fragment_89 = packed_half_0 + 0x40u;

            if (condition_2) {
                matrix_fragment_94 = matrix_fragment_88 * matrix_fragment_95 + matrix_fragment_94;
            }

            matrix_fragment_89 =
                uint32_t(combine_words(matrix_fragment_89, 0u) << 0x1bu >> 32) + matrix_fragment_90;

            matrix_fragment_90 = matrix_fragment_94 * 0x10u + matrix_fragment_89;

            matrix_fragment_90 = matrix_fragment_90 * 0x80u;

            matrix_fragment_91 = uint32_t(int32_t(matrix_fragment_90) >> 0x1fu);
        }
        if (!condition_3) {
            uniform_address_word_10 = 0u;

            matrix_fragment_91 = threadIdx.x & 31u;

            matrix_fragment_92 = matrix_fragment_92 * 0x200u;

            uniform_address_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                    select_half_operand<2, false, false>(0u));

            uniform_block_index_9 = 0x400u;

            matrix_fragment_88 = matrix_fragment_92 + 0x800u;

            matrix_fragment_90 = uniform_address_word_11;

            matrix_fragment_93 = uniform_address_word_11;

            matrix_fragment_89 = matrix_fragment_88 & 0xff800u;

            matrix_fragment_88 = packed_half_0 * 0x40u;

            matrix_fragment_89 = matrix_fragment_89 | 0x400u & matrix_fragment_92;

            matrix_fragment_88 = ~matrix_fragment_88 & 0x1000u;

            matrix_fragment_89 = matrix_fragment_89 | 0x200u & matrix_fragment_92;

            uniform_block_index_9 = (uniform_address_word_10 << 0x18u) + uniform_block_index_9;

            matrix_fragment_92 = matrix_fragment_88 + uniform_block_index_9 + matrix_fragment_89;

            matrix_fragment_88 = permute_signed(matrix_fragment_90, matrix_fragment_93, 0x5410u);

            matrix_fragment_92 = matrix_fragment_91 * 0x10u + matrix_fragment_92;

            matrix_fragment_89 = matrix_fragment_88;

            matrix_fragment_90 = matrix_fragment_88;

            matrix_fragment_91 = matrix_fragment_88;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_92, matrix_fragment_88, matrix_fragment_89,
                matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_block_index_9 = 0xffffffffu;

            condition_2 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_2) {
                matrix_fragment_88 = parameters[parameter_input_low_0];
                matrix_fragment_89 = parameters[parameter_input_high_1];

                uniform_block_index_9 = broadcast_active_lane(matrix_fragment_92);

                uniform_address_word_11 = broadcast_active_lane(packed_half_0);

                uniform_index_word_13 = 0u;

                uniform_block_index_9 = uniform_block_index_9 << (0x9u & 31u);

                uniform_index_word_12 = uint32_t(uint32_t(uniform_address_word_11) >> 0x6u);

                uniform_address_word_11 = uniform_address_word_11 << (0x6u & 31u);

                uniform_address_word_10 = uniform_block_index_9 + 0x800u;

                {
                    uint32_t result_word;
                    result_word = uniform_index_word_12 & 0x1u;
                    uniform_condition_0 = result_word != 0;
                }

                uniform_address_word_10 = uniform_address_word_10 & 0xff800u;

                uniform_index_word_12 = 0x400u;

                const uint64_t byte_address_3 =
                    scaled_word_address(combine_words(matrix_fragment_88, matrix_fragment_89),
                                        combine_words(matrix_fragment_90, matrix_fragment_91));
                matrix_fragment_88 = uint32_t(byte_address_3);

                uniform_index_word_15 = uniform_index_word_12 + 0x2000u;

                uniform_address_word_10 = uniform_address_word_10 | 0x400u & uniform_block_index_9;

                matrix_fragment_90 = uint32_t(byte_address_3 >> 32);

                uniform_address_word_22 = broadcast_active_lane(matrix_fragment_88);

                uniform_address_word_10 = uniform_address_word_10 | 0x200u & uniform_block_index_9;

                uniform_index_word_14 = (uniform_index_word_13 << 0x18u) + uniform_index_word_12;

                uniform_index_word_12 = ~uniform_address_word_11 & 0x1000u;

                uniform_address_word_23 = broadcast_active_lane(matrix_fragment_90);

                uniform_address_word_11 = (uniform_index_word_13 << 0x18u) + uniform_index_word_15;

                matrix_fragment_88 = 0x200u;

                uniform_address_word_10 =
                    uniform_index_word_12 + uniform_index_word_14 + uniform_address_word_10;

                uniform_block_index_9 = 0x20u;

                if (!uniform_condition_0) {
                    uniform_address_word_11 = uniform_address_word_11 + 0x8u;
                }

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_22, uniform_address_word_23),
                    uniform_block_index_9 * 16u, uniform_address_word_11, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_22, uniform_address_word_23),
                    uniform_block_index_9 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        thread_scratch[14] = matrix_accumulator_2;

        uniform_address_word_10 = 0u;

        matrix_fragment_88 = packed_half_0 * 0x40u;

        uniform_block_index_9 = 0x400u;

        matrix_fragment_89 = threadIdx.x & 31u;

        matrix_fragment_88 = matrix_fragment_88 & 0x1000u;

        matrix_accumulator_2 = uint32_t(uint32_t(packed_half_0) >> 0x6u);

        thread_scratch[13] = matrix_accumulator_2;

        matrix_accumulator_2 = thread_scratch[14];

        uniform_block_index_9 = (uniform_address_word_10 << 0x18u) + uniform_block_index_9;

        matrix_fragment_88 = matrix_fragment_88 + uniform_block_index_9;

        __syncwarp(__activemask());

        matrix_fragment_116 = matrix_fragment_89 * 0x10u + matrix_fragment_88;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0x200u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0x400u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0x600u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0x800u, matrix_fragment_104, matrix_fragment_105,
            matrix_fragment_106, matrix_fragment_107, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0xa00u, matrix_fragment_108, matrix_fragment_109,
            matrix_fragment_110, matrix_fragment_111, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0xc00u, matrix_fragment_112, matrix_fragment_113,
            matrix_fragment_114, matrix_fragment_115, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_116 + 0xe00u, matrix_fragment_116, matrix_fragment_117,
            matrix_fragment_118, matrix_fragment_119, shared_bytes, memory_ranges, error_counts);

        thread_scratch[14] = matrix_fragment_118;
        thread_scratch[15] = matrix_fragment_119;

        multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_164,
                                 matrix_accumulator_165);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_34,
                                 matrix_accumulator_35);

        matrix_fragment_118 = thread_scratch[0];
        matrix_fragment_119 = thread_scratch[1];

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_22, matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_20, matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_38, matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133,
                                 matrix_fragment_104, matrix_fragment_105, matrix_fragment_106,
                                 matrix_fragment_107, matrix_fragment_78, matrix_fragment_79,
                                 matrix_accumulator_132, matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135,
                                 matrix_fragment_104, matrix_fragment_105, matrix_fragment_106,
                                 matrix_fragment_107, matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137,
                                 matrix_fragment_104, matrix_fragment_105, matrix_fragment_106,
                                 matrix_fragment_107, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139,
                                 matrix_fragment_104, matrix_fragment_105, matrix_fragment_106,
                                 matrix_fragment_107, matrix_fragment_44, matrix_fragment_45,
                                 matrix_accumulator_138, matrix_accumulator_139);

        uniform_block_index_16 = blockIdx.z;

        packed_half_0 = packed_half_0 + 0x40u;

        uniform_address_word_10 = parameters[parameter_trained_weights_low_4];
        uniform_address_word_11 = parameters[parameter_trained_weights_high_5];

        multiply_accumulate_half(matrix_fragment_118, matrix_fragment_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_118,
                                 matrix_fragment_119);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_140,
                                 matrix_accumulator_141);

        thread_scratch[0] = matrix_fragment_118;
        thread_scratch[1] = matrix_fragment_119;

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_142,
                                 matrix_accumulator_143);

        matrix_fragment_118 = thread_scratch[14];
        matrix_fragment_119 = thread_scratch[15];

        matrix_accumulator_140 = thread_scratch[0];

        matrix_accumulator_141 = thread_scratch[1];

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_144,
                                 matrix_accumulator_145);

        matrix_accumulator_142 = thread_scratch[10];
        matrix_accumulator_143 = thread_scratch[11];

        matrix_accumulator_144 = thread_scratch[8];
        matrix_accumulator_145 = thread_scratch[9];

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_146,
                                 matrix_accumulator_147);

        matrix_accumulator_146 = thread_scratch[6];
        matrix_accumulator_147 = thread_scratch[7];

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_148,
                                 matrix_accumulator_149);

        matrix_fragment_106 = thread_scratch[2];
        matrix_fragment_107 = thread_scratch[3];

        matrix_accumulator_148 = thread_scratch[4];
        matrix_accumulator_149 = thread_scratch[5];

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_162,
                                 matrix_accumulator_163);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_124, matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_48, matrix_fragment_49,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_48, matrix_fragment_49,
                                 matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_40, matrix_fragment_41,
                                 matrix_accumulator_138, matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_150,
                                 matrix_accumulator_151);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_152,
                                 matrix_accumulator_153);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_162, matrix_accumulator_163,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_fragment_60, matrix_fragment_61);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_146, matrix_accumulator_147);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_106,
                                 matrix_fragment_107);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_accumulator_52, matrix_accumulator_53,
                                 matrix_accumulator_148, matrix_accumulator_149);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_144,
                                 matrix_accumulator_145);

        multiply_accumulate_half(matrix_fragment_106, matrix_fragment_107, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_fragment_96, matrix_fragment_97);

        matrix_fragment_96 = uniform_block_index_16;

        thread_scratch[0] = matrix_fragment_106;
        thread_scratch[1] = matrix_fragment_107;

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_62,
                                 matrix_fragment_63);

        matrix_fragment_106 = threadIdx.y;

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_52,
                                 matrix_accumulator_53);

        thread_scratch[2] = matrix_accumulator_56;
        thread_scratch[3] = matrix_accumulator_57;

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_44,
                                 matrix_fragment_45);

        thread_scratch[4] = matrix_accumulator_52;
        thread_scratch[5] = matrix_accumulator_53;

        matrix_fragment_96 = matrix_fragment_96 * 0x4u + matrix_fragment_106;

        matrix_fragment_51 = matrix_fragment_96 * 0x60u;

        thread_scratch[6] = matrix_fragment_48;
        thread_scratch[7] = matrix_fragment_49;

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_142,
                                 matrix_accumulator_143);

        thread_scratch[8] = matrix_fragment_40;
        thread_scratch[9] = matrix_fragment_41;

        matrix_fragment_107 = thread_scratch[13];

        multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_104, matrix_fragment_105);

        matrix_accumulator_52 = threadIdx.x & 31u;

        matrix_fragment_40 = uint32_t(uint32_t(packed_half_0) >> 0x5u);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_46,
                                 matrix_fragment_47);

        matrix_fragment_40 = matrix_fragment_40 * 0x60u;

        matrix_fragment_97 = 0u;

        matrix_fragment_45 = matrix_fragment_51 + 0x20u;

        matrix_fragment_49 = matrix_fragment_51 + 0x40u;

        matrix_fragment_41 =
            uint32_t(combine_words(matrix_fragment_51, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        matrix_fragment_47 = matrix_fragment_51 + 0x30u;

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_154,
                                 matrix_accumulator_155);

        matrix_fragment_45 =
            uint32_t(combine_words(matrix_fragment_45, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        matrix_fragment_47 =
            uint32_t(combine_words(matrix_fragment_47, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        matrix_fragment_49 =
            uint32_t(combine_words(matrix_fragment_49, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        thread_scratch[10] = matrix_fragment_42;
        thread_scratch[11] = matrix_fragment_43;

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_156,
                                 matrix_accumulator_157);

        matrix_fragment_43 = matrix_fragment_51 + 0x10u;

        matrix_fragment_51 = matrix_fragment_51 + 0x50u;

        matrix_fragment_41 = matrix_fragment_41 * 0x80u;

        matrix_fragment_43 =
            uint32_t(combine_words(matrix_fragment_43, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        matrix_fragment_51 =
            uint32_t(combine_words(matrix_fragment_51, 0u) << 0x1cu >> 32) + matrix_fragment_40;

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_68, matrix_fragment_69, packed_half_158,
                                 matrix_accumulator_159);

        matrix_fragment_40 = 0x4u;

        matrix_fragment_42 = 0x4u;

        matrix_fragment_43 = matrix_fragment_43 * 0x80u;

        split_address_words(matrix_fragment_40, matrix_fragment_41,
                            uint64_t(uint32_t(matrix_fragment_41)) *
                                    uint64_t(uint32_t(matrix_fragment_40)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        split_address_words(matrix_fragment_42, matrix_fragment_43,
                            uint64_t(uint32_t(matrix_fragment_43)) *
                                    uint64_t(uint32_t(matrix_fragment_42)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_160,
                                 matrix_accumulator_161);

        matrix_fragment_44 = 0x4u;

        matrix_fragment_46 = 0x4u;

        matrix_fragment_48 = 0x4u;

        matrix_fragment_50 = 0x4u;

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_64, matrix_fragment_65,
                                 matrix_fragment_88, matrix_fragment_89);

        split_address_words(matrix_fragment_88, matrix_fragment_89,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_40, matrix_fragment_41));

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_14,
                                 matrix_fragment_15);

        matrix_fragment_45 = matrix_fragment_45 * 0x80u;

        matrix_fragment_47 = matrix_fragment_47 * 0x80u;

        matrix_fragment_49 = matrix_fragment_49 * 0x80u;

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_12,
                                 matrix_accumulator_13);

        matrix_fragment_51 = matrix_fragment_51 * 0x80u;

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_10,
                                 matrix_fragment_11);

        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            uint64_t(uint32_t(matrix_fragment_45)) *
                                    uint64_t(uint32_t(matrix_fragment_44)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_74, matrix_fragment_75,
                                 matrix_accumulator_130, matrix_accumulator_131);

        split_address_words(matrix_fragment_46, matrix_fragment_47,
                            uint64_t(uint32_t(matrix_fragment_47)) *
                                    uint64_t(uint32_t(matrix_fragment_46)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_64, matrix_fragment_65,
                                 matrix_accumulator_128, matrix_accumulator_129);

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            uint64_t(uint32_t(matrix_fragment_49)) *
                                    uint64_t(uint32_t(matrix_fragment_48)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_66, matrix_fragment_67,
                                 matrix_accumulator_126, matrix_accumulator_127);

        split_address_words(matrix_fragment_50, matrix_fragment_51,
                            uint64_t(uint32_t(matrix_fragment_51)) *
                                    uint64_t(uint32_t(matrix_fragment_50)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_58, matrix_accumulator_59,
                                 matrix_accumulator_122, matrix_accumulator_123);

        matrix_fragment_102 = 0x400u;

        multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_164,
                                 matrix_accumulator_165);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_90, matrix_fragment_91);

        split_address_words(matrix_fragment_90, matrix_fragment_91,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_42, matrix_fragment_43));

        multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_fragment_80, matrix_fragment_81,
                                 matrix_fragment_84, matrix_fragment_85);

        multiply_accumulate_half(matrix_accumulator_152, matrix_accumulator_153,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_86, matrix_fragment_87);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_88, matrix_fragment_89),
                                            matrix_fragment_84, matrix_fragment_85,
                                            matrix_fragment_86, matrix_fragment_87, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_88, matrix_fragment_89) + 0xc000ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        matrix_fragment_88 = matrix_fragment_102 + 0x2000u;

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_34,
                                 matrix_accumulator_35);

        condition_2 = (matrix_fragment_107 & 0x1u) != 0;

        split_address_words(matrix_fragment_100, matrix_fragment_101,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_50, matrix_fragment_51));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_100, matrix_fragment_101) + 0xc000ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_fragment_74, matrix_fragment_75,
                                 matrix_accumulator_132, matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_154, matrix_accumulator_155,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_fragment_72, matrix_fragment_73,
                                 matrix_fragment_76, matrix_fragment_77);

        multiply_accumulate_half(matrix_accumulator_156, matrix_accumulator_157,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_78, matrix_fragment_79);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_90, matrix_fragment_91),
                                            matrix_fragment_76, matrix_fragment_77,
                                            matrix_fragment_78, matrix_fragment_79, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_90, matrix_fragment_91) + 0xc000ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        matrix_fragment_91 = (matrix_fragment_97 << 0x18u) + matrix_fragment_88;

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_30,
                                 matrix_accumulator_31);

        matrix_fragment_90 = matrix_fragment_91 + 0x8u;

        matrix_fragment_88 = !condition_2 ? matrix_fragment_90 : matrix_fragment_91;

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                                 matrix_fragment_25);

        split_address_words(matrix_fragment_92, matrix_fragment_93,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_44, matrix_fragment_45));

        split_address_words(matrix_fragment_94, matrix_fragment_95,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_46, matrix_fragment_47));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_100, matrix_fragment_101),
                                            matrix_fragment_44, matrix_fragment_45,
                                            matrix_fragment_46, matrix_fragment_47, memory_ranges,
                                            error_counts);

        multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147,
                                 matrix_fragment_108, matrix_fragment_109, matrix_fragment_110,
                                 matrix_fragment_111, matrix_accumulator_58, matrix_accumulator_59,
                                 matrix_fragment_98, matrix_fragment_99);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_94, matrix_fragment_95),
                                            matrix_fragment_60, matrix_fragment_61,
                                            matrix_fragment_62, matrix_fragment_63, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_94, matrix_fragment_95) + 0xc000ull,
            matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58,
            matrix_accumulator_59, memory_ranges, error_counts);

        split_address_words(matrix_fragment_98, matrix_fragment_99,
                            int64_t(int32_t(matrix_accumulator_52)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_98, matrix_fragment_99),
                                            matrix_accumulator_52, matrix_accumulator_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            memory_ranges, error_counts);

        multiply_accumulate_half(packed_half_158, matrix_accumulator_159, matrix_fragment_116,
                                 matrix_fragment_117, matrix_fragment_118, matrix_fragment_119,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_68,
                                 matrix_fragment_69);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_98, matrix_fragment_99) + 0xc000ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_160, matrix_accumulator_161,
                                 matrix_fragment_116, matrix_fragment_117, matrix_fragment_118,
                                 matrix_fragment_119, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_70, matrix_fragment_71);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_92, matrix_fragment_93),
                                            matrix_fragment_68, matrix_fragment_69,
                                            matrix_fragment_70, matrix_fragment_71, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_92, matrix_fragment_93) + 0xc000ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = matrix_fragment_88;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_88, matrix_fragment_89,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_88 = 0u;
#endif

        matrix_fragment_91 = condition_2 ? matrix_fragment_91 : matrix_fragment_90;

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_2 = wait_copy_barrier(shared_tile, matrix_fragment_91,
                                        async_tokens[(((matrix_fragment_91) >> 3) & 1)]);
#else
        __syncthreads();
        condition_2 = true;
#endif
        if (!condition_2) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_2 = wait_copy_barrier(shared_tile, matrix_fragment_91,
                                                async_tokens[(((matrix_fragment_91) >> 3) & 1)]);
#else
                __syncthreads();
                condition_2 = true;
#endif
            } while (!condition_2);
        }
        condition_2 = uint32_t(packed_half_0) >= uint32_t(0x1c0u);
    } while (!condition_2);
    matrix_fragment_98 = thread_scratch[0];
    matrix_fragment_99 = thread_scratch[1];

    matrix_fragment_90 = threadIdx.x & 31u;

    packed_half_0 = (matrix_fragment_97 << 0x18u) + matrix_fragment_102;

    uniform_index_word_5 = blockIdx.y;

    uniform_index_word_8 = parameters[parameter_geometry_or_control_8];
    uniform_block_index_9 = parameters[parameter_geometry_or_control_9];

    matrix_fragment_118 = thread_scratch[4];
    matrix_fragment_119 = thread_scratch[5];

    matrix_fragment_114 = thread_scratch[6];
    matrix_fragment_115 = thread_scratch[7];

    matrix_fragment_106 = thread_scratch[8];
    matrix_fragment_107 = thread_scratch[9];

    matrix_fragment_102 = thread_scratch[10];
    matrix_fragment_103 = thread_scratch[11];

    packed_half_0 = matrix_fragment_90 * 0x10u + packed_half_0;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1000u, matrix_fragment_88,
                                        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1400u, matrix_fragment_92,
                                        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_164,
                             matrix_accumulator_165);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_18,
                             matrix_accumulator_19);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_16,
                             matrix_accumulator_17);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_14,
                             matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_12,
                             matrix_accumulator_13);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0x1800u, matrix_accumulator_12, matrix_accumulator_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_108, matrix_fragment_109, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = thread_scratch[2];
    matrix_accumulator_3 = thread_scratch[3];

    multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_10,
                             matrix_fragment_11);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1c00u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_112, matrix_fragment_113, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_6,
                             matrix_fragment_7);

    multiply_accumulate_half(matrix_fragment_110, matrix_fragment_111, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                             matrix_fragment_5);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1200u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_146,
                             matrix_accumulator_147);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_3 = 0x4u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_fragment_96)) *
                                uint64_t(uint32_t(matrix_accumulator_3)) +
                            combine_words(uniform_address_word_10, uniform_address_word_11));

    matrix_fragment_94 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0xe0000ull, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_140,
                             matrix_accumulator_141);

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_142,
                             matrix_accumulator_143);

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_144,
                             matrix_accumulator_145);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_134,
                             matrix_accumulator_135);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_accumulator_12,
                             matrix_accumulator_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_148,
                             matrix_accumulator_149);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                      select_half_operand<2, false, false>(matrix_accumulator_30),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_34,
                             matrix_accumulator_35);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_accumulator_150,
                             matrix_accumulator_151);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_152,
                             matrix_accumulator_153);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_154,
                             matrix_accumulator_155);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_156,
                             matrix_accumulator_157);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_68, matrix_fragment_69, packed_half_158,
                             matrix_accumulator_159);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_160,
                             matrix_accumulator_161);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_162,
                             matrix_accumulator_163);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_118,
                             matrix_fragment_119);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_114,
                             matrix_fragment_115);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_106,
                             matrix_fragment_107);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_102,
                             matrix_fragment_103);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1600u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_164,
                             matrix_accumulator_165);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_18,
                             matrix_accumulator_19);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_16,
                             matrix_accumulator_17);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_164),
                                      select_half_operand<2, false, false>(matrix_accumulator_164),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                         select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                      select_half_operand<2, false, false>(matrix_fragment_99),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_165),
                                      select_half_operand<2, false, false>(matrix_accumulator_165),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_38,
                             matrix_accumulator_39);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_105), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_106 = result_word;
    }

    matrix_accumulator_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                           select_half_operand<2, false, false>(matrix_fragment_6));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_95), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_104 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_114 = result_word;
    }

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_36,
                             matrix_accumulator_37);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1a00u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_106 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                                         select_half_operand<2, false, false>(matrix_fragment_106));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_106), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_accumulator_39 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                       select_half_operand<2, false, false>(matrix_fragment_114));

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_100,
                             matrix_fragment_101);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_92,
                             matrix_fragment_93);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    matrix_accumulator_36 = permute_signed(matrix_fragment_95, matrix_fragment_95, 0x5432u);

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                       select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_37 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_9 = permute_signed(matrix_accumulator_38, matrix_accumulator_38, 0x5432u);

    matrix_fragment_10 = permute_signed(matrix_accumulator_37, matrix_accumulator_37, 0x5432u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x4100410u);

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_37 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_92 = half_to_float_bits((matrix_fragment_95 & 65535u) * 65537u);

    matrix_accumulator_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38), 0x4100410u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_95 = half_to_float_bits((matrix_fragment_95 >> 16) * 65537u);

    matrix_fragment_9 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    matrix_fragment_93 = half_to_float_bits((matrix_accumulator_38 & 65535u) * 65537u);

    matrix_fragment_92 = rsqrt_bits(matrix_fragment_92);

    matrix_accumulator_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                           select_half_operand<2, false, false>(matrix_fragment_9));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1e00u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37), 0x4100410u);

    matrix_accumulator_38 = half_to_float_bits((matrix_accumulator_38 >> 16) * 65537u);

    matrix_fragment_115 = rsqrt_bits(matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_144,
                             matrix_accumulator_145);

    matrix_fragment_114 = half_to_float_bits((matrix_accumulator_37 >> 16) * 65537u);

    matrix_accumulator_163 = threadIdx.x & 31u;

    matrix_fragment_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36), 0x4100410u);

    matrix_fragment_93 = rsqrt_bits(matrix_fragment_93);

    matrix_fragment_95 = half_to_float_bits((matrix_accumulator_37 & 65535u) * 65537u);

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_118 = rsqrt_bits(matrix_accumulator_38);

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_146,
                             matrix_accumulator_147);

    matrix_fragment_116 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_154 = half_to_float_bits((matrix_fragment_117 & 65535u) * 65537u);

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                      select_half_operand<2, false, false>(matrix_accumulator_127),
                                      select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_fragment_95 = rsqrt_bits(matrix_fragment_95);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_119 = half_to_float_bits((matrix_fragment_117 >> 16) * 65537u);

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_114 = rsqrt_bits(matrix_fragment_114);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_140,
                             matrix_accumulator_141);

    matrix_fragment_116 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                         select_half_operand<2, false, false>(matrix_fragment_116));

    matrix_fragment_92 = pack_half(matrix_fragment_115, matrix_fragment_92);

    matrix_accumulator_36 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_38 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_154 = rsqrt_bits(matrix_accumulator_154);

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_142,
                             matrix_accumulator_143);

    matrix_fragment_115 = rsqrt_bits(matrix_fragment_119);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_116), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_117 = result_word;
    }

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    packed_half_0 = pack_half(matrix_fragment_118, matrix_fragment_93);

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_148 = result_word;
    }

    matrix_accumulator_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_107 = 0x4u;

    matrix_fragment_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_95 = pack_half(matrix_fragment_114, matrix_fragment_95);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_106 = matrix_fragment_96 * 0x800u;

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                                        select_half_operand<2, false, false>(matrix_fragment_100));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    split_address_words(matrix_fragment_106, matrix_fragment_107,
                        uint64_t(uint32_t(matrix_fragment_106)) *
                                uint64_t(uint32_t(matrix_fragment_107)) +
                            combine_words(uniform_address_word_10, uniform_address_word_11));

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                      select_half_operand<2, false, false>(matrix_fragment_88),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_accumulator_144));

    matrix_accumulator_154 = pack_half(matrix_fragment_115, matrix_accumulator_154);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_164 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_164),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_accumulator_146));

    split_address_words(matrix_fragment_106, matrix_fragment_107,
                        int64_t(int32_t(matrix_accumulator_163)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_106, matrix_fragment_107));

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    multiply_accumulate_half(matrix_accumulator_152, matrix_accumulator_153, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0000ull, matrix_accumulator_36,
        matrix_accumulator_37, matrix_accumulator_38, matrix_accumulator_39, memory_ranges,
        error_counts);

    matrix_accumulator_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_158 = pack_half(0u, matrix_fragment_94);

    uniform_address_word_10 = parameters[parameter_tensor_height_6];
    uniform_address_word_11 = parameters[parameter_tensor_width_7];

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0200ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_114, matrix_fragment_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                                      select_half_operand<2, false, false>(matrix_accumulator_142),
                                      select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                            select_half_operand<2, false, false>(matrix_fragment_101));

    matrix_fragment_65 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_116),
                       select_half_operand<2, false, false>(matrix_fragment_117));

    multiply_accumulate_half(matrix_fragment_116, matrix_fragment_117, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                       select_half_operand<2, false, false>(matrix_accumulator_148));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_32), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_155 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_69 = result_word;
    }

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_accumulator_154));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_accumulator_154));

    matrix_accumulator_165 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_165),
                            select_half_operand<2, false, false>(matrix_accumulator_154));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_34), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    matrix_accumulator_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_154));

    matrix_fragment_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_118),
                      select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_155 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_155));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_33 = result_word;
    }

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_accumulator_145));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_60 = result_word;
    }

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_accumulator_147));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_32 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_35 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                                      select_half_operand<2, false, false>(matrix_accumulator_143),
                                      select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_57 = permute_signed(matrix_fragment_64, matrix_fragment_64, 0x5432u);

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_56), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_57 = result_word;
    }

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x4100410u);

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_93 = half_to_float_bits((matrix_fragment_64 & 65535u) * 65537u);

    matrix_fragment_66 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_58 = half_to_float_bits((matrix_fragment_64 >> 16) * 65537u);

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_33 = permute_signed(matrix_accumulator_34, matrix_accumulator_34, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_59 = result_word;
    }

    matrix_fragment_93 = rsqrt_bits(matrix_fragment_93);

    matrix_fragment_63 = permute_signed(matrix_fragment_60, matrix_fragment_60, 0x5432u);

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_62 = result_word;
    }

    matrix_accumulator_58 = rsqrt_bits(matrix_accumulator_58);

    matrix_fragment_61 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_148 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                      select_half_operand<2, false, false>(matrix_accumulator_31));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0600ull, matrix_accumulator_32,
        matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, memory_ranges,
        error_counts);

    matrix_fragment_119 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_119),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0400ull, matrix_accumulator_28,
        matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31, memory_ranges,
        error_counts);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x4100410u);

    matrix_accumulator_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x4100410u);

    matrix_accumulator_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_152),
                            select_half_operand<2, false, false>(matrix_accumulator_152));

    matrix_accumulator_159 = half_to_float_bits((matrix_fragment_61 & 65535u) * 65537u);

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_114),
                                      select_half_operand<2, false, false>(matrix_fragment_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_157 = half_to_float_bits((matrix_fragment_60 & 65535u) * 65537u);

    matrix_accumulator_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_116),
                                      select_half_operand<2, false, false>(matrix_fragment_116),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_71 = half_to_float_bits((matrix_fragment_60 >> 16) * 65537u);

    matrix_fragment_93 = pack_half(matrix_accumulator_58, matrix_fragment_93);

    matrix_fragment_61 = half_to_float_bits((matrix_fragment_61 >> 16) * 65537u);

    matrix_accumulator_149 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_149),
                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_accumulator_159 = rsqrt_bits(matrix_accumulator_159);

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_68 = rsqrt_bits(matrix_fragment_61);

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_156 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_59));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_accumulator_157 = rsqrt_bits(matrix_accumulator_157);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0800ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    matrix_accumulator_155 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_155),
                       select_half_operand<2, false, false>(matrix_fragment_62));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0a00ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_71 = rsqrt_bits(matrix_fragment_71);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_159 = pack_half(matrix_fragment_68, matrix_accumulator_159);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0c00ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_accumulator_164 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_164),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_115),
                                      select_half_operand<2, false, false>(matrix_fragment_115),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_157 = pack_half(matrix_fragment_71, matrix_accumulator_157);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_153),
                            select_half_operand<2, false, false>(matrix_accumulator_153));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_117),
                                      select_half_operand<2, false, false>(matrix_fragment_117),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_69 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5432u);

    matrix_accumulator_160 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_164),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_161 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc0e00ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_accumulator_165 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_165),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_161 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_161), 0x4100410u);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_165),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_99 = half_to_float_bits((matrix_accumulator_161 & 65535u) * 65537u);

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_161 = half_to_float_bits((matrix_accumulator_161 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_160), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    matrix_accumulator_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_154),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_99 = rsqrt_bits(matrix_fragment_99);

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_154 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_154),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_161 = rsqrt_bits(matrix_accumulator_161);

    matrix_fragment_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                      select_half_operand<2, false, false>(matrix_accumulator_154));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_159));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_159));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_159));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_160),
                                   select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_99 = pack_half(matrix_accumulator_161, matrix_fragment_99);

    matrix_accumulator_159 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_159));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_157));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_157));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_159),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_124 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_98 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_128 =
        permute_signed(matrix_accumulator_155, matrix_accumulator_155, 0x5432u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_155 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_155),
                       select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_accumulator_155 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_155), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_24 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_140 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x4100410u);

    matrix_fragment_97 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    packed_half_0 = half_to_float_bits((matrix_accumulator_140 & 65535u) * 65537u);

    matrix_accumulator_140 = half_to_float_bits((matrix_accumulator_140 >> 16) * 65537u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_accumulator_142 = half_to_float_bits((matrix_accumulator_155 & 65535u) * 65537u);

    matrix_fragment_25 = half_to_float_bits((matrix_accumulator_155 >> 16) * 65537u);

    matrix_accumulator_155 =
        permute_signed(matrix_accumulator_156, matrix_accumulator_156, 0x5432u);

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x4100410u);

    matrix_accumulator_156 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_156),
                       select_half_operand<2, false, false>(matrix_accumulator_155));

    matrix_fragment_97 = rsqrt_bits(matrix_accumulator_140);

    matrix_accumulator_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_accumulator_157));

    matrix_accumulator_156 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_156), 0x4100410u);

    matrix_accumulator_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_accumulator_157));

    matrix_fragment_24 = rsqrt_bits(matrix_accumulator_142);

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_157),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = half_to_float_bits((matrix_fragment_27 & 65535u) * 65537u);

    matrix_accumulator_142 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = half_to_float_bits((matrix_fragment_27 >> 16) * 65537u);

    matrix_accumulator_145 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                      select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_accumulator_142 = half_to_float_bits((matrix_accumulator_156 & 65535u) * 65537u);

    matrix_accumulator_156 = half_to_float_bits((matrix_accumulator_156 >> 16) * 65537u);

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_144),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_25 = rsqrt_bits(matrix_fragment_25);

    matrix_accumulator_144 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = pack_half(matrix_fragment_97, packed_half_0);

    packed_half_0 = rsqrt_bits(matrix_accumulator_142);

    matrix_accumulator_143 = rsqrt_bits(matrix_accumulator_156);

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    matrix_fragment_27 = rsqrt_bits(matrix_fragment_27);

    matrix_fragment_25 = pack_half(matrix_fragment_25, matrix_fragment_24);

    packed_half_0 = pack_half(matrix_accumulator_143, packed_half_0);

    matrix_accumulator_144 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(matrix_accumulator_144));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_26 = pack_half(matrix_fragment_27, matrix_fragment_26);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_144, matrix_accumulator_145, matrix_accumulator_30,
                             matrix_accumulator_31);

    matrix_fragment_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_116),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_152),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_153 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_153),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_114),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_115),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_117),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_153),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_accumulator_152 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_152),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_116),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(packed_half_158));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                            select_half_operand<0, false, false>(packed_half_158));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_117 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_117),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_115 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_115),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_36,
                             matrix_accumulator_37);

    matrix_accumulator_30 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(matrix_accumulator_152));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_100),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_114 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_114),
                                        select_half_operand<2, false, false>(matrix_fragment_117));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_fragment_88 = 0x29c0u;

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_114, matrix_fragment_115, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_60,
                             matrix_fragment_61);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_36 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3c203c20u);

    matrix_accumulator_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_38 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3c203c20u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_122 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3c203c20u);

    matrix_accumulator_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_accumulator_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3c203c20u);

    matrix_accumulator_140 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_37 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3c203c20u);

    matrix_accumulator_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3e473e47u);

    matrix_accumulator_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_39 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3c203c20u);

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_38 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3e473e47u);

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_122 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3e473e47u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_123 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3c203c20u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3c203c20u);

    matrix_accumulator_141 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_28 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28), 0x3c203c20u);

    matrix_accumulator_140 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140), 0x3c203c20u);

    matrix_accumulator_128 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_32 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3c203c20u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_37 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3e473e47u);

    matrix_accumulator_29 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3c203c20u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_89 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3e473e47u);

    matrix_accumulator_123 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3e473e47u);

    matrix_accumulator_129 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_114, matrix_fragment_115, matrix_fragment_70,
                             matrix_fragment_71);

    packed_half_0 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       0x3e473e47u);

    matrix_accumulator_140 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140), 0x3e473e47u);

    matrix_accumulator_34 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34), 0x3c203c20u);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_accumulator_39 = (matrix_accumulator_36 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_28 = (matrix_accumulator_38 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = (matrix_accumulator_122 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_accumulator_128 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_accumulator_141 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141), 0x3c203c20u);

    matrix_accumulator_32 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_32), 0x3e473e47u);

    matrix_accumulator_29 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3e473e47u);

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_accumulator_36 = (matrix_accumulator_37 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_34 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34), 0x3e473e47u);

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                       select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_89 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_38 = (matrix_accumulator_123 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 = (matrix_accumulator_129 << 0x5u) + 0x7ff88000u;

    packed_half_0 = (packed_half_0 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_37 = (matrix_accumulator_140 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_accumulator_141 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141), 0x3e473e47u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_26 = (matrix_accumulator_32 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = (matrix_accumulator_29 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_32 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_27 = (matrix_accumulator_34 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_29 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_57 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_accumulator_34 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_56 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_59 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = (matrix_accumulator_141 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_100));

    matrix_accumulator_58 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_61 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                       select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_35),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_33 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3c203c20u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_accumulator_33 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x3e473e47u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_65 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_accumulator_33 = (matrix_accumulator_33 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_67 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_92 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_69 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_100 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                         select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_70 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_fragment_100));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_94 = matrix_accumulator_163 * 0x4u;

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    condition_0 = (matrix_accumulator_163 & 0x1u) != 0;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    condition_1 = (matrix_accumulator_163 & 0x2u) != 0;

    matrix_accumulator_35 = matrix_fragment_94 & 0x1cu;

    matrix_fragment_94 = condition_0 ? matrix_fragment_63 : matrix_fragment_62;

    matrix_fragment_97 = condition_0 ? matrix_fragment_64 : matrix_fragment_95;

    matrix_fragment_63 = condition_0 ? matrix_fragment_62 : matrix_fragment_63;

    matrix_fragment_64 = condition_0 ? matrix_fragment_95 : matrix_fragment_64;

    matrix_accumulator_35 =
        uint32_t(int64_t(int32_t(matrix_accumulator_163)) * (1ll << 0x1du) >> 32) +
        matrix_accumulator_35;

    matrix_fragment_100 = !condition_1 ? matrix_fragment_94 : matrix_fragment_97;

    matrix_fragment_94 = !condition_1 ? matrix_fragment_97 : matrix_fragment_94;

    matrix_fragment_95 = !condition_1 ? matrix_fragment_63 : matrix_fragment_64;

    matrix_fragment_97 = !condition_1 ? matrix_fragment_64 : matrix_fragment_63;

    matrix_fragment_62 = 0x1u ^ matrix_accumulator_35;

    matrix_fragment_63 = 0x2u ^ matrix_accumulator_35;

    matrix_fragment_64 = 0x3u ^ matrix_accumulator_35;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_100), "r"(matrix_accumulator_35), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_100 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_95), "r"(matrix_fragment_62), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(matrix_fragment_63), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_94 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(matrix_fragment_64), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    condition_2 = (matrix_accumulator_163 & 0x8u) != 0;

    multiply_accumulate_half(matrix_fragment_112, matrix_fragment_113, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_112,
                             matrix_fragment_113);

    condition_3 = (matrix_accumulator_163 & 0x10u) != 0;

    multiply_accumulate_half(matrix_fragment_110, matrix_fragment_111, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_110,
                             matrix_fragment_111);

    multiply_accumulate_half(matrix_fragment_108, matrix_fragment_109, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_108,
                             matrix_fragment_109);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_134,
                             matrix_accumulator_135);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_12,
                             matrix_accumulator_13);

    matrix_fragment_4 = !condition_2 ? matrix_fragment_100 : matrix_fragment_95;

    matrix_fragment_5 = !condition_2 ? matrix_fragment_95 : matrix_fragment_100;

    matrix_fragment_7 = !condition_2 ? matrix_fragment_94 : matrix_fragment_97;

    matrix_fragment_6 = !condition_2 ? matrix_fragment_97 : matrix_fragment_94;

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_80 = !condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_81 = !condition_3 ? matrix_fragment_5 : matrix_fragment_6;

    matrix_fragment_4 = !condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_5 = !condition_3 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_80 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_80),
                                        select_half_operand<1, false, false>(matrix_fragment_80));

    matrix_fragment_80 = permute_signed(matrix_fragment_80, matrix_fragment_80, 0x5410u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_80 & 65535u) * 65537u);

    matrix_fragment_80 = half_to_float_bits((matrix_fragment_80 >> 16) * 65537u);

    matrix_fragment_72 = matrix_accumulator_163;

    matrix_fragment_4 = reciprocal_bits(matrix_fragment_4);

    matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_72) >> 0x2u);

    matrix_fragment_5 = reciprocal_bits(matrix_fragment_80);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_7 = matrix_fragment_73 + 0x8u;

    matrix_fragment_48 = uint32_t(int32_t(matrix_fragment_72) >> 0x1fu);

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_fragment_48, 0u) << 0x5u >> 32) + matrix_fragment_73;

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_6 = matrix_fragment_6 & 0xffffffe0u;

    matrix_fragment_74 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    matrix_fragment_74 =
        uint32_t(combine_words(matrix_fragment_74, 0u) << 0x5u >> 32) + matrix_fragment_7;

    matrix_fragment_6 = matrix_fragment_73 + (0u - matrix_fragment_6);

    matrix_fragment_49 = pack_half(matrix_fragment_5, matrix_fragment_4);

    matrix_fragment_74 = matrix_fragment_74 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    matrix_fragment_74 = matrix_fragment_7 + (0u - matrix_fragment_74);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_74), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_40 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5410u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_41 = permute_signed(matrix_fragment_74, matrix_fragment_74, 0x5410u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_38 =
        uint32_t(int64_t(int32_t(matrix_fragment_72)) * (1ll << 0x1eu) >> 32) + 0x18u;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_accumulator_39 = uint32_t(int32_t(matrix_accumulator_38) >> 0x1fu);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_39 =
        uint32_t(combine_words(matrix_accumulator_39, 0u) << 0x5u >> 32) + matrix_accumulator_38;

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        uint32_t(int64_t(int32_t(matrix_fragment_72)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = uint32_t(int32_t(matrix_accumulator_28) >> 0x1fu);

    matrix_accumulator_36 =
        uint32_t(combine_words(matrix_fragment_9, 0u) << 0x5u >> 32) + matrix_accumulator_28;

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = matrix_accumulator_36 & 0xffffffe0u;

    matrix_accumulator_39 = matrix_accumulator_39 & 0xffffffe0u;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 = matrix_accumulator_28 + (0u - matrix_fragment_15);

    matrix_fragment_42 = matrix_accumulator_38 + (0u - matrix_accumulator_39);

    matrix_accumulator_23 = transpose_half_tile(matrix_accumulator_23);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = transpose_half_tile(matrix_accumulator_19);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_121 = transpose_half_tile(matrix_accumulator_121);

    matrix_fragment_51 = transpose_half_tile(matrix_accumulator_21);

    matrix_fragment_103 = transpose_half_tile(matrix_fragment_103);

    matrix_accumulator_17 = transpose_half_tile(matrix_accumulator_17);

    matrix_fragment_105 = transpose_half_tile(matrix_fragment_105);

    matrix_accumulator_22 = transpose_half_tile(matrix_accumulator_22);

    matrix_fragment_10 = transpose_half_tile(matrix_accumulator_18);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_accumulator_28), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_42), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_accumulator_120 = transpose_half_tile(matrix_accumulator_120);

    matrix_fragment_50 = transpose_half_tile(matrix_accumulator_20);

    matrix_fragment_102 = transpose_half_tile(matrix_fragment_102);

    matrix_fragment_14 = transpose_half_tile(matrix_accumulator_16);

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_104);

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_121 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_103),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_36 = permute_signed(matrix_accumulator_28, matrix_accumulator_28, 0x5410u);

    matrix_fragment_74 = permute_signed(matrix_fragment_42, matrix_fragment_42, 0x5410u);

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                      select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                       select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 = transpose_half_tile(matrix_accumulator_135);

    matrix_accumulator_53 = transpose_half_tile(matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_28 = transpose_half_tile(matrix_accumulator_134);

    matrix_accumulator_29 = transpose_half_tile(matrix_accumulator_52);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 = transpose_half_tile(matrix_accumulator_137);

    matrix_accumulator_55 = transpose_half_tile(matrix_accumulator_55);

    matrix_accumulator_139 = transpose_half_tile(matrix_accumulator_139);

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_accumulator_13 = transpose_half_tile(matrix_accumulator_13);

    matrix_fragment_47 = transpose_half_tile(matrix_fragment_47);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_136 = transpose_half_tile(matrix_accumulator_136);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_accumulator_54 = transpose_half_tile(matrix_accumulator_54);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_44 = transpose_half_tile(matrix_fragment_44);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_12 = transpose_half_tile(matrix_accumulator_12);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_46 = transpose_half_tile(matrix_fragment_46);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_66 = transpose_half_tile(matrix_accumulator_138);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    uniform_index_word_13 = blockIdx.x;

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_66 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                                       select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_11) >> 0x1fu);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_42,
                             matrix_fragment_43);

    uniform_index_word_12 =
        uint32_t(combine_words(uniform_index_word_12, 0u) << 0x2u >> 32) + uniform_address_word_11;

    uniform_address_word_11 = uint32_t(int32_t(uniform_address_word_10) >> 0x1fu);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_37 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 = parameters[parameter_output_low_2];
    matrix_accumulator_13 = parameters[parameter_output_high_3];

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    uniform_address_word_11 = uint32_t(combine_words(uniform_address_word_11, 0u) << 0x2u >> 32) +
                              uniform_address_word_10;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_56,
                             matrix_accumulator_57);

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_12) >> 0x2u);

    uniform_address_word_11 = uint32_t(int32_t(uniform_address_word_11) >> 0x2u);

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_block_index_9;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    uniform_address_word_10 = uniform_index_word_7 * uniform_index_word_12 + uniform_index_word_6;

    uniform_condition_2 = int32_t(uniform_index_word_7) >= int32_t(uniform_address_word_11);

    uniform_index_word_8 = (uniform_index_word_13 << 0x3u) + uniform_index_word_8;

    uniform_condition_1 = int32_t(uniform_index_word_6) >= int32_t(uniform_index_word_12);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_20,
                             matrix_accumulator_21);

    uniform_condition_2 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x4u) && !uniform_condition_2;

    uniform_condition_1 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x4u) && !uniform_condition_1;

    matrix_fragment_25 = uniform_address_word_10;

    uniform_condition_0 = uniform_condition_2 & uniform_condition_1;

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_18,
                             matrix_accumulator_19);

    matrix_fragment_25 = matrix_fragment_25 * 0x10u + matrix_fragment_96;

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_38,
                             matrix_accumulator_39);

    matrix_fragment_25 = matrix_fragment_25 * 0x80u;

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_16,
                             matrix_accumulator_17);

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));
    if (uniform_condition_0) {
        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                                      select_half_operand<2, false, false>(0u));

        matrix_accumulator_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_19 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                          select_half_operand<2, false, false>(matrix_accumulator_17));

        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            int64_t(int32_t(matrix_accumulator_163)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                          select_half_operand<2, false, false>(matrix_accumulator_19));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_16, matrix_accumulator_17), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    uniform_index_word_5 = uniform_index_word_6 + 0x1u;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                      select_half_operand<2, false, false>(0u));

    uniform_condition_0 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_12);

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                  select_half_operand<2, false, false>(0u));

    uniform_condition_0 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x8u) && !uniform_condition_0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                      select_half_operand<2, false, false>(0u));

    uniform_condition_2 = uniform_condition_2 & uniform_condition_0;

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_2) {
        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_accumulator_16, 0x5410u);

        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_163)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_accumulator_19, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_accumulator_17, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x2000ull,
            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
            memory_ranges, error_counts);
    }
    packed_half_0 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0xe0000ull, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1200ull, matrix_accumulator_16,
        matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1000ull, matrix_accumulator_36,
        matrix_accumulator_37, matrix_accumulator_38, matrix_accumulator_39, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1800ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1a00ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1400ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1c00ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1600ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_106, matrix_fragment_107) + 0xc1e00ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108),
                            select_half_operand<2, false, false>(matrix_fragment_108));

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_fragment_48 =
        uint32_t(combine_words(matrix_fragment_48, 0u) << 0x5u >> 32) + matrix_fragment_73;

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    uniform_block_index_9 = blockIdx.y;

    matrix_accumulator_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_110),
                                      select_half_operand<2, false, false>(matrix_fragment_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_12));

    uniform_index_word_8 = parameters[parameter_geometry_or_control_9];

    matrix_accumulator_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_13 = result_word;
    }

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                                      select_half_operand<2, false, false>(matrix_fragment_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_accumulator_2));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    matrix_accumulator_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                       select_half_operand<2, false, false>(matrix_accumulator_13));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_13), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_20 = result_word;
    }

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_32 = result_word;
    }

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_13 = permute_signed(matrix_accumulator_20, matrix_accumulator_20, 0x5432u);

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20), 0x4100410u);

    matrix_accumulator_3 = permute_signed(matrix_accumulator_32, matrix_accumulator_32, 0x5432u);

    matrix_accumulator_21 = half_to_float_bits((matrix_accumulator_20 & 65535u) * 65537u);

    matrix_accumulator_20 = half_to_float_bits((matrix_accumulator_20 >> 16) * 65537u);

    matrix_accumulator_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_21 = rsqrt_bits(matrix_accumulator_21);

    matrix_accumulator_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32), 0x4100410u);

    matrix_accumulator_20 = rsqrt_bits(matrix_accumulator_20);

    matrix_accumulator_33 = half_to_float_bits((matrix_accumulator_32 & 65535u) * 65537u);

    matrix_accumulator_32 = half_to_float_bits((matrix_accumulator_32 >> 16) * 65537u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_33 = rsqrt_bits(matrix_accumulator_33);

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_accumulator_12));

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_32 = rsqrt_bits(matrix_accumulator_32);

    matrix_accumulator_20 = pack_half(matrix_accumulator_20, matrix_accumulator_21);

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_13 = result_word;
    }

    matrix_fragment_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_110),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_20 = result_word;
    }

    matrix_accumulator_32 = pack_half(matrix_accumulator_32, matrix_accumulator_33);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    packed_half_0 = pack_half(0u, packed_half_0);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_110),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_110),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_113),
                                       select_half_operand<2, false, false>(matrix_accumulator_12));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_112),
                                       select_half_operand<2, false, false>(matrix_accumulator_21));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_13), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_12 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_20), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_21 = result_word;
    }

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_108),
                                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_109),
                                       select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_16,
                             matrix_accumulator_17);

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                       select_half_operand<2, false, false>(matrix_accumulator_12));

    matrix_accumulator_17 = permute_signed(matrix_accumulator_16, matrix_accumulator_16, 0x5432u);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_18,
                             matrix_accumulator_19);

    matrix_accumulator_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_18 = permute_signed(matrix_accumulator_21, matrix_accumulator_21, 0x5432u);

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                       select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_accumulator_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16), 0x4100410u);

    matrix_accumulator_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21), 0x4100410u);

    matrix_accumulator_17 = half_to_float_bits((matrix_accumulator_16 & 65535u) * 65537u);

    matrix_accumulator_16 = half_to_float_bits((matrix_accumulator_16 >> 16) * 65537u);

    matrix_accumulator_18 = half_to_float_bits((matrix_accumulator_21 & 65535u) * 65537u);

    matrix_accumulator_21 = half_to_float_bits((matrix_accumulator_21 >> 16) * 65537u);

    matrix_accumulator_17 = rsqrt_bits(matrix_accumulator_17);

    matrix_accumulator_16 = rsqrt_bits(matrix_accumulator_16);

    matrix_accumulator_18 = rsqrt_bits(matrix_accumulator_18);

    matrix_accumulator_21 = rsqrt_bits(matrix_accumulator_21);

    matrix_accumulator_17 = pack_half(matrix_accumulator_16, matrix_accumulator_17);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_19 = pack_half(matrix_accumulator_21, matrix_accumulator_18);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(packed_half_0));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(packed_half_0));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                      select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                      select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                          select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                      select_half_operand<2, false, false>(matrix_accumulator_20));

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_accumulator_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_2),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_12),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_accumulator_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_36 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3c203c20u);

    matrix_accumulator_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_3),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_accumulator_38 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3c203c20u);

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x3c203c20u);

    matrix_accumulator_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    packed_half_0 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                                       0x3c203c20u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_37 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3c203c20u);

    matrix_accumulator_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3e473e47u);

    matrix_accumulator_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_13),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_accumulator_38 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3e473e47u);

    matrix_accumulator_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x3c203c20u);

    matrix_accumulator_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x3e473e47u);

    matrix_accumulator_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x3e473e47u);

    matrix_accumulator_52 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3c203c20u);

    matrix_accumulator_54 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3c203c20u);

    matrix_accumulator_37 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3e473e47u);

    matrix_accumulator_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_114, matrix_fragment_115, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_accumulator_39 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3c203c20u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_13 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_13), 0x3c203c20u);

    matrix_accumulator_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3e473e47u);

    matrix_accumulator_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3e473e47u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3e473e47u);

    packed_half_0 = (matrix_accumulator_36 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_3 = (matrix_accumulator_38 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_12 = (matrix_accumulator_12 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_21 = (matrix_accumulator_21 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_114, matrix_fragment_115, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_39 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3e473e47u);

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_31 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_13), 0x3e473e47u);

    matrix_accumulator_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_2 = (matrix_accumulator_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_30 = (matrix_accumulator_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_33 = (matrix_accumulator_54 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_32 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_37 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_accumulator_53 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3c203c20u);

    matrix_accumulator_55 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3c203c20u);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3c203c20u);

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_13 = (matrix_accumulator_39 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_20 = (matrix_accumulator_20 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_31 = (matrix_accumulator_31 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_6),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3e473e47u);

    matrix_accumulator_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3e473e47u);

    matrix_accumulator_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3e473e47u);

    matrix_accumulator_39 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_fragment_40 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x3c203c20u);

    matrix_accumulator_34 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_5),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_18 = (matrix_accumulator_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_7),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_43 = (matrix_accumulator_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_accumulator_16 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3c203c20u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x3e473e47u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_6 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_19 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3e473e47u);

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3c203c20u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x3c203c20u);

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<0, false, false>(matrix_fragment_88), 0x3d343d34u);

    matrix_fragment_4 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_17 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                       select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3c203c20u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3c203c20u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3c203c20u);

    matrix_accumulator_16 = (matrix_accumulator_36 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_39 = (matrix_accumulator_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3e473e47u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x3e473e47u);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_accumulator_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3e473e47u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3e473e47u);

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3e473e47u);

    matrix_fragment_44 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_fragment_40 = (matrix_fragment_5 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_fragment_7 = (matrix_fragment_24 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_5 = (matrix_fragment_25 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_38 = (matrix_fragment_26 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_36 = (matrix_fragment_27 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_44 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                       select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_fragment_26 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_55 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_26 = condition_0 ? matrix_fragment_25 : matrix_fragment_24;

    matrix_fragment_24 = condition_0 ? matrix_fragment_24 : matrix_fragment_25;

    matrix_fragment_25 = condition_0 ? matrix_accumulator_54 : matrix_fragment_27;

    matrix_fragment_27 = condition_0 ? matrix_fragment_27 : matrix_accumulator_54;

    matrix_fragment_46 = !condition_1 ? matrix_fragment_26 : matrix_fragment_25;

    matrix_accumulator_53 = !condition_1 ? matrix_fragment_24 : matrix_fragment_27;

    matrix_fragment_26 = !condition_1 ? matrix_fragment_25 : matrix_fragment_26;

    matrix_fragment_27 = !condition_1 ? matrix_fragment_27 : matrix_fragment_24;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_accumulator_35), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_53), "r"(matrix_fragment_62), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_53 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_26), "r"(matrix_fragment_63), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_27), "r"(matrix_fragment_64), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_27 = result_word;
    }

    matrix_fragment_24 = !condition_2 ? matrix_fragment_46 : matrix_accumulator_53;

    matrix_fragment_25 = !condition_2 ? matrix_accumulator_53 : matrix_fragment_46;

    matrix_accumulator_55 = !condition_2 ? matrix_fragment_26 : matrix_fragment_27;

    matrix_accumulator_52 = !condition_2 ? matrix_fragment_27 : matrix_fragment_26;

    matrix_accumulator_54 = !condition_3 ? matrix_fragment_24 : matrix_accumulator_55;

    matrix_accumulator_57 = !condition_3 ? matrix_fragment_25 : matrix_accumulator_52;

    matrix_fragment_24 = !condition_3 ? matrix_accumulator_55 : matrix_fragment_24;

    matrix_fragment_25 = !condition_3 ? matrix_accumulator_52 : matrix_fragment_25;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_54),
                       select_half_operand<1, false, false>(matrix_accumulator_54));

    matrix_accumulator_54 = permute_signed(matrix_accumulator_54, matrix_accumulator_54, 0x5410u);

    matrix_accumulator_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54), 0x4100410u);

    matrix_fragment_24 = half_to_float_bits((matrix_accumulator_54 & 65535u) * 65537u);

    matrix_accumulator_54 = half_to_float_bits((matrix_accumulator_54 >> 16) * 65537u);

    matrix_fragment_26 =
        uint32_t(int64_t(int32_t(matrix_fragment_72)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_72 =
        uint32_t(int64_t(int32_t(matrix_fragment_72)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_24 = reciprocal_bits(matrix_fragment_24);

    matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_26) >> 0x1fu);

    matrix_accumulator_35 = uint32_t(int32_t(matrix_fragment_72) >> 0x1fu);

    matrix_fragment_25 = reciprocal_bits(matrix_accumulator_54);

    matrix_fragment_27 =
        uint32_t(combine_words(matrix_fragment_27, 0u) << 0x5u >> 32) + matrix_fragment_26;

    matrix_accumulator_35 =
        uint32_t(combine_words(matrix_accumulator_35, 0u) << 0x5u >> 32) + matrix_fragment_72;

    matrix_fragment_27 = matrix_fragment_27 & 0xffffffe0u;

    matrix_accumulator_35 = matrix_accumulator_35 & 0xffffffe0u;

    matrix_accumulator_55 = matrix_fragment_26 + (0u - matrix_fragment_27);

    matrix_fragment_27 = matrix_fragment_73 + 0x8u;

    matrix_accumulator_57 = matrix_fragment_72 + (0u - matrix_accumulator_35);

    matrix_fragment_24 = pack_half(matrix_fragment_25, matrix_fragment_24);

    matrix_fragment_46 = uint32_t(int32_t(matrix_fragment_27) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(matrix_accumulator_55), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_25 = result_word;
    }

    matrix_fragment_46 =
        uint32_t(combine_words(matrix_fragment_46, 0u) << 0x5u >> 32) + matrix_fragment_27;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(matrix_accumulator_57), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    matrix_fragment_48 = matrix_fragment_48 & 0xffffffe0u;

    matrix_fragment_46 = matrix_fragment_46 & 0xffffffe0u;

    matrix_accumulator_35 = matrix_fragment_73 + (0u - matrix_fragment_48);

    matrix_accumulator_53 = matrix_fragment_27 + (0u - matrix_fragment_46);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(matrix_accumulator_35), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(matrix_accumulator_53), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_48 = permute_signed(matrix_fragment_25, matrix_fragment_25, 0x5410u);

    matrix_accumulator_52 = permute_signed(matrix_fragment_26, matrix_fragment_26, 0x5410u);

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_37 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_35 = permute_signed(matrix_accumulator_35, matrix_accumulator_35, 0x5410u);

    matrix_fragment_46 = permute_signed(matrix_fragment_46, matrix_fragment_46, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                      select_half_operand<2, false, false>(matrix_accumulator_39));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_0 = threadIdx.x;

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_60 = threadIdx.y;

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_36 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_56,
                             matrix_accumulator_57);

    uniform_index_word_5 = uniform_index_word_7 * uniform_index_word_12 + uniform_index_word_12;

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_58,
                             matrix_accumulator_59);

    uniform_index_word_7 = uniform_index_word_7 + 0x1u;

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_7 = threadIdx.z;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_accumulator_36));

    uniform_index_word_5 = uniform_index_word_6 + uniform_index_word_5;

    uniform_index_word_8 = (uniform_block_index_9 << 0x3u) + uniform_index_word_8;

    uniform_condition_2 = int32_t(uniform_index_word_7) >= int32_t(uniform_address_word_11);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_22,
                             matrix_accumulator_23);

    uniform_condition_2 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x8u) && !uniform_condition_2;

    matrix_fragment_5 = uniform_index_word_5;

    uniform_condition_1 = uniform_condition_2 & uniform_condition_1;

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_12,
                             matrix_accumulator_13);

    matrix_fragment_5 = matrix_fragment_5 * 0x10u + matrix_fragment_96;

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_fragment_5 = matrix_fragment_5 * 0x80u;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_14,
                             matrix_fragment_15);

    condition_0 = (matrix_fragment_60 | matrix_fragment_7 | packed_half_0) != 0;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_1) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_accumulator_163)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_2 & uniform_condition_0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_4, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_163)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_11, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_accumulator_54, matrix_fragment_9, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x2000ull,
            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
            memory_ranges, error_counts);
    }
    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_index_word_6 = parameters[parameter_publication_low_12];
    uniform_index_word_7 = parameters[parameter_publication_high_13];

    uniform_index_word_5 = (uniform_block_index_9 << 0x2u) + uniform_block_index_16;

    uniform_index_word_5 = uniform_index_word_5 * uniform_index_word_4 + uniform_index_word_13;

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
extern "C" __global__ __maxnreg__(
    255) void phoenix_global_attention_split_heads_512_query_key_value(KernelArguments parameters) {
    run_global_attention_split_heads_512_query_key_value<false>(parameters.words, nullptr, nullptr);
}
