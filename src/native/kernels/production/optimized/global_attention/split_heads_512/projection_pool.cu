#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/projection_pool.cu
// 512-channel split-head stage. Separate projection/feed-forward modules use the same persistent feature layout.
// The filename identifies which stage is executed; wait/view variants preserve producer-consumer synchronization.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 9232
#include "../../../common/async_memory.cuh"
#include "../../../common/cached_vector_access.cuh"

// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_input_skip_low_2 = 2,
    parameter_input_skip_high_3 = 3,
    parameter_output_encoder_skip_low_4 = 4,
    parameter_output_encoder_skip_high_5 = 5,
    parameter_output_pool_low_6 = 6,
    parameter_output_pool_high_7 = 7,
    parameter_trained_weights_low_8 = 8,
    parameter_trained_weights_high_9 = 9,
    parameter_wait_low_10 = 10,
    parameter_wait_high_11 = 11,
    parameter_publication_low_12 = 12,
    parameter_publication_high_13 = 13,
    parameter_optional_control_14 = 14,
    parameter_scalar_control_15 = 15,
    parameter_tensor_height_16 = 16,
    parameter_tensor_width_17 = 17,
    parameter_LH_18 = 18,
    parameter_LW_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[20];
};
static_assert(sizeof(KernelArguments) == 80, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_projection_pool(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_accumulator_4 = 0;
    uint32_t matrix_accumulator_5 = 0;
    uint32_t matrix_accumulator_6 = 0;
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
    uint32_t matrix_accumulator_40 = 0;
    uint32_t matrix_accumulator_41 = 0;
    uint32_t matrix_accumulator_42 = 0;
    uint32_t matrix_accumulator_43 = 0;
    uint32_t matrix_accumulator_44 = 0;
    uint32_t matrix_accumulator_45 = 0;
    uint32_t matrix_accumulator_46 = 0;
    uint32_t matrix_accumulator_47 = 0;
    uint32_t matrix_accumulator_48 = 0;
    uint32_t matrix_accumulator_49 = 0;
    uint32_t matrix_accumulator_50 = 0;
    uint32_t matrix_accumulator_51 = 0;
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
    uint32_t reused_word_130 = 0;
    uint32_t index_word_131 = 0;
    uint32_t block_index_132 = 0;
    uint32_t index_word_133 = 0;
    uint32_t index_word_134 = 0;
    uint32_t index_word_135 = 0;
    uint32_t index_word_136 = 0;
    uint32_t thread_row_137 = 0;
    uint32_t block_index_138 = 0;
    uint32_t index_word_139 = 0;
    uint32_t index_word_140 = 0;
    uint32_t index_word_141 = 0;
    uint32_t index_word_142 = 0;
    uint32_t index_word_143 = 0;
    uint32_t tile_word_144 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_block_index_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_index_word_17 = 0;
    uint32_t uniform_index_word_18 = 0;
    uint32_t uniform_index_word_79 = 0;
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

    matrix_fragment_11 = parameters[parameter_tensor_width_17];

    block_index_132 = blockIdx.x;

    uniform_index_word_13 = parameters[parameter_scalar_control_15];

    thread_row_137 = threadIdx.y;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    matrix_fragment_15 = parameters[parameter_tensor_height_16];

    block_index_138 = blockIdx.z;

    matrix_fragment_9 = threadIdx.x & 31u;

    uniform_block_index_9 = blockIdx.y;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_index_word_13));

    index_word_0 = matrix_fragment_11 + 0xffffffffu;

    matrix_accumulator_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x3u >> 32) + index_word_0;

    matrix_accumulator_5 =
        uint32_t(int64_t(int32_t(matrix_accumulator_3)) * (1ll << 0x1du) >> 32) + 0x1u;

    matrix_fragment_13 =
        signed_word_magnitude(matrix_accumulator_5);

    matrix_accumulator_6 =
        signed_word_magnitude(matrix_accumulator_5);

    index_word_0 = __float_as_uint(__int2float_ru(matrix_fragment_13));

    index_word_131 = ~matrix_accumulator_5;

    index_word_0 = reciprocal_bits(index_word_0);

    matrix_accumulator_2 = index_word_0 + 0xffffffeu;

    index_word_0 = signed_word_magnitude(block_index_132);

    matrix_accumulator_3 = __float2uint_rz(__uint_as_float(matrix_accumulator_2));

    matrix_accumulator_2 = 0u;

    matrix_accumulator_4 = 0u - matrix_accumulator_3;

    matrix_accumulator_7 = matrix_accumulator_4 * matrix_fragment_13;

    matrix_accumulator_4 = 0u - matrix_accumulator_6;

    matrix_accumulator_6 = threadIdx.x;

    matrix_accumulator_3 = uint32_t(uint64_t(matrix_accumulator_3) * matrix_accumulator_7 +
                                        combine_words(matrix_accumulator_2, matrix_accumulator_3) >>
                                    32);

    matrix_accumulator_2 = block_index_132 ^ matrix_accumulator_5;

    matrix_accumulator_3 = uint32_t(uint64_t(matrix_accumulator_3) * index_word_0 >> 32);

    index_word_0 = matrix_accumulator_3 * matrix_accumulator_4 + index_word_0;

    condition_1 = uint32_t(matrix_fragment_13) > uint32_t(index_word_0);

    if (!condition_1) {
        index_word_0 = index_word_0 + (0u - matrix_fragment_13);

        matrix_accumulator_3 = matrix_accumulator_3 + 0x1u;
    }

    condition_1 = int32_t(matrix_accumulator_2) >= int32_t(0u);

    condition_0 = uint32_t(index_word_0) >= uint32_t(matrix_fragment_13);

    if (condition_0) {
        matrix_accumulator_3 = matrix_accumulator_3 + 0x1u;
    }

    condition_0 = int32_t(matrix_accumulator_5) != int32_t(0u);

    matrix_accumulator_4 = matrix_accumulator_3;

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_8];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_9];

    if (!condition_1) {
        matrix_accumulator_4 = 0u - matrix_accumulator_4;
    }

    matrix_accumulator_7 = matrix_accumulator_6 | thread_row_137;
    condition_1 = matrix_accumulator_7 != 0;

    matrix_accumulator_6 = matrix_fragment_15 + (0u - uniform_index_word_4);

    matrix_accumulator_4 = !condition_0 ? index_word_131 : matrix_accumulator_4;

    matrix_accumulator_4 = matrix_accumulator_4 * 0x4u + thread_row_137;

    reused_word_130 = matrix_accumulator_4 * 0x40u;

    matrix_accumulator_5 = reused_word_130 * 0x8u;

    matrix_accumulator_5 = block_index_138 * 0x10000u + matrix_accumulator_5;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));
    if (!condition_1) {
        uniform_address_word_6 = 0u;

        uniform_address_word_7 = blockDim.x;

        uniform_index_word_8 = blockDim.y;

        uniform_index_word_4 = 0x400u;

        uniform_index_word_5 = uniform_index_word_4 + 0x2000u;

        uniform_index_word_4 = uniform_address_word_7 * uniform_index_word_8;

        uniform_address_word_6 = (uniform_address_word_6 << 0x18u) + uniform_index_word_5;

        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);

        __threadfence_block();

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }
    matrix_fragment_10 = parameters[parameter_optional_control_14];

    matrix_fragment_8 = threadIdx.z;

    phoenix_architecture::finish_compatibility_copies();

    condition_2 = int32_t(block_index_132) >= int32_t(0u);

    condition_4 = uint32_t(matrix_accumulator_6) != 0u;

    condition_1 = uint32_t(matrix_fragment_13) > uint32_t(index_word_0);

    matrix_accumulator_5 = index_word_0 + (0u - matrix_fragment_13);

    matrix_accumulator_4 = min(int32_t(0u), int32_t(matrix_fragment_10));

    matrix_accumulator_4 = 0u - matrix_accumulator_4 + matrix_fragment_11;

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                        matrix_fragment_80, matrix_fragment_81, matrix_fragment_82,
                                        matrix_fragment_83, memory_ranges, error_counts);

    condition_3 = uint32_t(matrix_accumulator_4) != 0u;

    index_word_0 = !condition_1 ? matrix_accumulator_5 : index_word_0;

    condition_1 = (matrix_accumulator_7 | matrix_fragment_8) != 0;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    if (condition_4) {
        matrix_accumulator_6 = matrix_accumulator_6 + 0xffffffffu;
    }

    if (!condition_2) {
        index_word_0 = 0u - index_word_0;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x400ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x600ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    if (condition_3) {
        matrix_accumulator_4 = matrix_accumulator_4 + 0xffffffffu;
    }

    index_word_131 = !condition_0 ? index_word_131 : index_word_0;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4000ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    index_word_0 = uint32_t(int32_t(matrix_fragment_15) >> 0x1fu);

    if (condition_4) {
        matrix_accumulator_5 = uint32_t(int32_t(matrix_accumulator_6) >> 0x1fu);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4200ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4400ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4600ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    index_word_0 = uint32_t(combine_words(index_word_0, 0u) << 0x2u >> 32) + matrix_fragment_15;

    uniform_index_word_12 = uniform_block_index_9 << (0x1u & 31u);

    if (condition_4) {
        matrix_accumulator_5 =
            uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x3u >> 32) + matrix_accumulator_6;
    }

    matrix_fragment_16 = index_word_131 * 0x2u;

    index_word_141 = 0u;

    index_word_135 = 0u;

    matrix_fragment_12 = uint32_t(int32_t(index_word_0) >> 0x2u);

    if (condition_4) {
        index_word_135 =
            uint32_t(int64_t(int32_t(matrix_accumulator_5)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_2 = uint32_t(int32_t(matrix_fragment_11) >> 0x1fu);

    if (condition_3) {
        matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_4) >> 0x1fu);
    }

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x2u >> 32) + matrix_fragment_11;

    if (condition_3) {
        matrix_accumulator_3 =
            uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x3u >> 32) + matrix_accumulator_4;
    }

    matrix_fragment_13 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    if (condition_3) {
        index_word_141 =
            uint32_t(int64_t(int32_t(matrix_accumulator_3)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    index_word_0 = reused_word_130 + 0x20u;
    if (!condition_1) {
        uniform_index_word_4 = (uniform_block_index_9 << 0x3u) + (0u - uniform_index_word_13);

        matrix_accumulator_2 = index_word_131 * 0x8u + (0u - matrix_fragment_10);

        uniform_index_word_5 = uniform_index_word_4 + 0x7u;

        uniform_address_word_6 = uniform_index_word_4 + 0x8u;

        uniform_index_word_4 = max(int32_t(0u), int32_t(uniform_index_word_4));

        matrix_accumulator_4 = matrix_accumulator_2 + 0x8u;

        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x1du >> 32) + 0x1u;

        uniform_condition_0 = int32_t(uniform_address_word_6) > int32_t(0u);

        uniform_index_word_4 = uint32_t(uint32_t(uniform_index_word_4) >> 0x3u);

        matrix_accumulator_3 = matrix_accumulator_2 + 0x7u;

        condition_0 = int32_t(matrix_accumulator_4) > int32_t(0u);

        uniform_index_word_5 = uniform_condition_0 ? uniform_index_word_5 : 0u;

        matrix_accumulator_2 = max(int32_t(0u), int32_t(matrix_accumulator_2));

        matrix_accumulator_3 =
            uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x1du >> 32) + 0x1u;

        matrix_accumulator_6 = min(int32_t(index_word_135), int32_t(uniform_index_word_5));

        matrix_accumulator_4 = condition_0 ? matrix_accumulator_3 : 0u;

        condition_1 = int32_t(matrix_accumulator_6) <= int32_t(uniform_index_word_4);
        if (!condition_1) {
            matrix_fragment_8 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

            matrix_fragment_11 = index_word_141 * 0x4u;

            matrix_fragment_9 = min(int32_t(matrix_accumulator_4), int32_t(index_word_141));

            matrix_accumulator_5 = uniform_index_word_4;
            do {
                condition_0 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_9);
                if (!condition_0) {
                    matrix_accumulator_2 = block_index_138 + matrix_accumulator_5;

                    matrix_accumulator_4 = matrix_fragment_8;

                    matrix_fragment_15 = matrix_fragment_11 * matrix_accumulator_2;
                    do {
                        matrix_accumulator_2 = parameters[parameter_wait_low_10];
                        matrix_accumulator_3 = parameters[parameter_wait_high_11];

                        matrix_accumulator_7 = matrix_fragment_15 + matrix_accumulator_4;

                        split_address_words(
                            matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

                        matrix_accumulator_7 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_2, matrix_accumulator_3),
                            memory_ranges, error_counts);

                        matrix_accumulator_4 = matrix_accumulator_4 + 0x1u;

                        condition_1 = uint32_t(matrix_accumulator_4) != uint32_t(matrix_fragment_9);

                        condition_0 = int32_t(matrix_accumulator_7) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                matrix_accumulator_7 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(matrix_accumulator_7) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                matrix_accumulator_5 = matrix_accumulator_5 + 0x1u;

                condition_0 = uint32_t(matrix_accumulator_5) != uint32_t(matrix_accumulator_6);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    matrix_accumulator_6 = parameters[parameter_tensor_height_16];

    index_word_140 = uint32_t(uint32_t(thread_row_137) >> 0x2u);

    matrix_accumulator_3 = thread_row_137 * 0x80u;

    matrix_accumulator_2 = uint32_t(uint32_t(thread_row_137) >> 0x1u);

    phoenix_architecture::finish_compatibility_copies();

    matrix_accumulator_4 = index_word_140 + uniform_index_word_12;

    matrix_accumulator_2 = matrix_accumulator_2 & 0x1u;

    condition_2 = false;

    condition_1 = int32_t(matrix_accumulator_4) >= int32_t(matrix_fragment_12);

    matrix_accumulator_4 = matrix_accumulator_4 * matrix_fragment_13;

    matrix_fragment_14 = matrix_fragment_16 + matrix_accumulator_2;

    uniform_index_word_17 = 0u;

    matrix_accumulator_5 = matrix_accumulator_3 & 0x100u;

    matrix_accumulator_2 = matrix_fragment_14;

    matrix_accumulator_6 = matrix_accumulator_6 & 0xfffffffcu;

    condition_3 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

    condition_0 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

    matrix_accumulator_7 = condition_0 ? matrix_accumulator_4 : 0u;
    if (condition_1) {
        if (!condition_3) {
            matrix_accumulator_2 = parameters[parameter_tensor_width_17];

            matrix_accumulator_2 = matrix_accumulator_2 & 0xfffffffcu;

            condition_1 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

            matrix_accumulator_2 = 0u;

            condition_2 = int32_t(matrix_fragment_14) < int32_t(matrix_fragment_13) || !condition_1;

            if (condition_1) {
                matrix_accumulator_2 = matrix_fragment_14;
            }
        }
    } else {
        matrix_accumulator_2 = parameters[parameter_tensor_width_17];

        matrix_accumulator_2 = matrix_accumulator_2 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

        matrix_accumulator_2 = 0u;

        condition_2 = int32_t(matrix_fragment_14) < int32_t(matrix_fragment_13) || !condition_1;

        if (condition_1) {
            matrix_accumulator_2 = matrix_fragment_14;
        }
    }
    if (condition_2) {
        matrix_fragment_8 = block_index_138 + matrix_accumulator_7 + matrix_accumulator_2;
    }

    matrix_accumulator_4 = index_word_140 * 0x200u + matrix_accumulator_5;

    matrix_accumulator_7 = matrix_accumulator_3 & 0x80u;

    uniform_index_word_16 = 0x400u;

    uniform_index_word_13 = (uniform_index_word_17 << 0x18u) + uniform_index_word_16;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3, 0ull);

    if (condition_2) {
        matrix_fragment_8 = matrix_fragment_8 * 0x800u + matrix_accumulator_7;
    }

    matrix_accumulator_4 = matrix_accumulator_4 + matrix_accumulator_7;

    if (condition_2) {
        matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);
    }

    matrix_accumulator_4 = (matrix_accumulator_4 << 0x2u) + uniform_index_word_13;

    if (condition_2) {
        matrix_accumulator_2 = matrix_fragment_8;

        matrix_accumulator_3 = matrix_fragment_9;
    }
    if (!condition_2) {
        matrix_accumulator_3 = threadIdx.x & 31u;

        uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = uniform_index_word_4;

        matrix_fragment_9 = uniform_index_word_4;

        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = matrix_fragment_8;

        matrix_fragment_10 = matrix_fragment_8;

        matrix_fragment_11 = matrix_fragment_8;

        matrix_accumulator_4 = matrix_accumulator_3 * 0x10u + matrix_accumulator_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_4, matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_index_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            matrix_fragment_8 = parameters[parameter_input_low_0];
            matrix_fragment_9 = parameters[parameter_input_high_1];

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_4);

            uniform_index_word_4 = uniform_index_word_16 + 0x2000u;

            uniform_address_word_15 = (uniform_index_word_17 << 0x18u) + uniform_index_word_4;

            uniform_index_word_4 = 0x20u;

            const uint64_t byte_address_0 =
                scaled_word_address(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
            matrix_fragment_8 = uint32_t(byte_address_0);

            matrix_accumulator_2 = uint32_t(byte_address_0 >> 32);

            uniform_address_word_6 = broadcast_active_lane(matrix_fragment_8);

            uniform_address_word_7 = broadcast_active_lane(matrix_accumulator_2);

            matrix_accumulator_2 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_6, uniform_address_word_7),
                uniform_index_word_4 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_6, uniform_address_word_7),
                                   uniform_index_word_4 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_2 = thread_row_137 + 0x4u;

    condition_3 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

    condition_2 = false;

    matrix_accumulator_2 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x2u);

    matrix_accumulator_4 = matrix_fragment_14;

    matrix_accumulator_3 = matrix_accumulator_2 + uniform_index_word_12;

    condition_1 = int32_t(matrix_accumulator_3) >= int32_t(matrix_fragment_12);

    matrix_accumulator_3 = matrix_accumulator_3 * matrix_fragment_13;

    matrix_accumulator_3 = condition_0 ? matrix_accumulator_3 : 0u;
    if (condition_1) {
        if (!condition_3) {
            matrix_accumulator_4 = parameters[parameter_tensor_width_17];

            matrix_accumulator_4 = matrix_accumulator_4 & 0xfffffffcu;

            condition_1 = uint32_t(matrix_accumulator_4) != uint32_t(0x4u);

            matrix_accumulator_4 = 0u;

            condition_2 = int32_t(matrix_fragment_14) < int32_t(matrix_fragment_13) || !condition_1;

            if (condition_1) {
                matrix_accumulator_4 = matrix_fragment_14;
            }
        }
    } else {
        matrix_accumulator_4 = parameters[parameter_tensor_width_17];

        matrix_accumulator_4 = matrix_accumulator_4 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_4) != uint32_t(0x4u);

        matrix_accumulator_4 = 0u;

        condition_2 = int32_t(matrix_fragment_14) < int32_t(matrix_fragment_13) || !condition_1;

        if (condition_1) {
            matrix_accumulator_4 = matrix_fragment_14;
        }
    }
    if (condition_2) {
        matrix_fragment_8 = block_index_138 + matrix_accumulator_3 + matrix_accumulator_4;
    }

    matrix_accumulator_4 = matrix_accumulator_2 * 0x200u + matrix_accumulator_5;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3, 0ull);

    if (condition_2) {
        matrix_fragment_8 = matrix_fragment_8 * 0x800u + matrix_accumulator_7;
    }

    matrix_accumulator_4 = matrix_accumulator_7 + matrix_accumulator_4;

    if (condition_2) {
        matrix_accumulator_5 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);
    }

    matrix_accumulator_4 = (matrix_accumulator_4 << 0x2u) + uniform_index_word_13;

    if (condition_2) {
        matrix_accumulator_2 = matrix_fragment_8;

        matrix_accumulator_3 = matrix_accumulator_5;
    }
    if (!condition_2) {
        matrix_accumulator_3 = threadIdx.x & 31u;

        uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = uniform_index_word_4;

        matrix_accumulator_5 = uniform_index_word_4;

        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_accumulator_5, 0x5410u);

        matrix_fragment_9 = matrix_fragment_8;

        matrix_fragment_10 = matrix_fragment_8;

        matrix_fragment_11 = matrix_fragment_8;

        matrix_accumulator_4 = matrix_accumulator_3 * 0x10u + matrix_accumulator_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_4, matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_index_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            matrix_fragment_8 = parameters[parameter_input_low_0];
            matrix_fragment_9 = parameters[parameter_input_high_1];

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_4);

            uniform_index_word_4 = uniform_index_word_16 + 0x2000u;

            uniform_address_word_15 = (uniform_index_word_17 << 0x18u) + uniform_index_word_4;

            uniform_index_word_4 = 0x20u;

            // Later control flow also consumes this address-overflow predicate.
            condition_1 = (uint64_t(matrix_accumulator_2) << 2) + matrix_fragment_8 > 0xffffffffull;
            const uint64_t byte_address_1 =
                scaled_word_address(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
            matrix_accumulator_5 = uint32_t(byte_address_1);

            matrix_accumulator_2 = uint32_t(byte_address_1 >> 32);

            uniform_address_word_6 = broadcast_active_lane(matrix_accumulator_5);

            uniform_address_word_7 = broadcast_active_lane(matrix_accumulator_2);

            matrix_accumulator_2 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_6, uniform_address_word_7),
                uniform_index_word_4 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_6, uniform_address_word_7),
                                   uniform_index_word_4 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_5 = 0u;

    uniform_index_word_4 = uniform_index_word_16 + 0x2000u;

    matrix_accumulator_4 = parameters[parameter_tensor_width_17];

    condition_6 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

    matrix_accumulator_2 = uniform_index_word_4;

    condition_2 = int32_t(matrix_fragment_12) <= int32_t(uniform_index_word_12) && condition_6;

    matrix_accumulator_6 = matrix_accumulator_4 & 0xfffffffcu;

    matrix_accumulator_5 = (matrix_accumulator_5 << 0x18u) + matrix_accumulator_2;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_accumulator_5;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_2 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 = wait_copy_barrier(shared_tile, matrix_accumulator_5,
                                    async_tokens[(((matrix_accumulator_5) >> 3) & 1)]);
#else
    phoenix_architecture::finish_compatibility_copies();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_accumulator_5,
                                            async_tokens[(((matrix_accumulator_5) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    condition_4 = uint32_t(matrix_accumulator_6) == uint32_t(0x4u) || condition_2;

    matrix_fragment_10 = matrix_fragment_16 + 0x1u;

    uniform_index_word_4 = uniform_index_word_12 + 0x1u;

    matrix_accumulator_2 = matrix_fragment_13 * uniform_index_word_12;

    condition_5 = int32_t(matrix_fragment_12) > int32_t(uniform_index_word_12) || !condition_0;

    condition_3 = int32_t(matrix_fragment_16) < int32_t(matrix_fragment_13) || condition_4;

    condition_4 = int32_t(matrix_fragment_10) < int32_t(matrix_fragment_13) || condition_4;

    condition_2 = uint32_t(matrix_accumulator_6) == uint32_t(0x4u) && !condition_2;

    condition_1 = int32_t(matrix_fragment_12) <= int32_t(uniform_index_word_4) && condition_6;

    matrix_accumulator_4 = condition_0 ? matrix_accumulator_2 : 0u;

    condition_3 = condition_3 & condition_5;

    matrix_accumulator_28 = threadIdx.x & 31u;

    matrix_accumulator_2 = matrix_accumulator_2 + matrix_fragment_13;

    condition_5 = condition_4 & condition_5;

    condition_4 = uint32_t(matrix_accumulator_6) == uint32_t(0x4u) || condition_1;

    matrix_fragment_9 = !condition_2 ? matrix_fragment_16 : 0u;

    matrix_fragment_22 = condition_0 ? matrix_accumulator_2 : 0u;

    matrix_fragment_11 = !condition_2 ? matrix_fragment_10 : 0u;

    condition_0 = int32_t(matrix_fragment_12) > int32_t(uniform_index_word_4) || !condition_0;

    condition_2 = int32_t(matrix_fragment_16) < int32_t(matrix_fragment_13) || condition_4;

    condition_4 = int32_t(matrix_fragment_10) < int32_t(matrix_fragment_13) || condition_4;

    condition_1 = uint32_t(matrix_accumulator_6) == uint32_t(0x4u) && !condition_1;

    if (condition_3) {
        matrix_accumulator_2 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_3 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_5 = matrix_accumulator_4 + matrix_fragment_9;
    }

    condition_2 = condition_2 & condition_0;

    uniform_index_word_4 = parameters[parameter_optional_control_14];
    uniform_index_word_5 = parameters[parameter_scalar_control_15];

    condition_4 = condition_4 & condition_0;

    if (condition_5) {
        matrix_accumulator_7 = matrix_accumulator_4 + matrix_fragment_11;
    }

    if (condition_3) {
        matrix_accumulator_5 = matrix_accumulator_5 * 0x200u + reused_word_130;
    }

    if (condition_5) {
        matrix_fragment_14 = parameters[parameter_input_skip_low_2];
        matrix_fragment_15 = parameters[parameter_input_skip_high_3];
    }

    matrix_accumulator_27 = !condition_1 ? matrix_fragment_16 : 0u;

    if (condition_3) {
        matrix_accumulator_31 = matrix_accumulator_4 + matrix_fragment_9;
    }

    if (condition_5) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x200u + reused_word_130;
    }

    matrix_accumulator_29 = !condition_1 ? matrix_fragment_10 : 0u;

    if (condition_3) {
        matrix_accumulator_5 = matrix_accumulator_5 * 0x4u;
    }

    if (!condition_3) {
        matrix_accumulator_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (condition_5) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_18 = parameters[parameter_input_skip_low_2];
        matrix_fragment_19 = parameters[parameter_input_skip_high_3];

        matrix_fragment_9 = matrix_fragment_22 + matrix_accumulator_27;
    }

    if (condition_5) {
        matrix_accumulator_41 = matrix_accumulator_4 + matrix_fragment_11;
    }

    matrix_fragment_8 = matrix_accumulator_28;

    if (condition_2) {
        matrix_fragment_9 = matrix_fragment_9 * 0x200u + reused_word_130;
    }

    if (condition_4) {
        matrix_fragment_20 = parameters[parameter_input_skip_low_2];
        matrix_fragment_21 = parameters[parameter_input_skip_high_3];
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (condition_4) {
        matrix_fragment_17 = matrix_fragment_22 + matrix_accumulator_29;
    }

    if (!condition_2) {
        matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_fragment_9 = matrix_fragment_9 * 0x4u;
    }

    if (!condition_5) {
        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    matrix_accumulator_26 = reused_word_130 + 0x8u;

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_8];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_9];

    if (condition_5) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_3) {
        matrix_accumulator_7 = permute_signed(matrix_accumulator_6, matrix_accumulator_6, 0x5410u);
    }

    if (condition_4) {
        matrix_fragment_23 = matrix_fragment_17 * 0x200u + reused_word_130;
    }

    if (!condition_3) {
        matrix_accumulator_4 = matrix_accumulator_7;

        matrix_accumulator_5 = matrix_accumulator_7;

        matrix_accumulator_6 = matrix_accumulator_7;
    }

    if (condition_5) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_2) {
        matrix_fragment_15 = permute_signed(matrix_accumulator_24, matrix_accumulator_24, 0x5410u);
    }

    if (condition_2) {
        matrix_accumulator_48 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_49 = parameters[parameter_input_skip_high_3];
    }

    matrix_accumulator_39 = matrix_fragment_8 & 0x7ffffffcu;

    if (condition_2) {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), matrix_accumulator_4,
            matrix_accumulator_5, matrix_accumulator_6, matrix_accumulator_7, memory_ranges,
            error_counts);
    }

    if (!condition_2) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (!condition_5) {
        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5410u);
    }

    if (condition_4) {
        matrix_fragment_23 = matrix_fragment_23 * 0x4u;
    }

    matrix_accumulator_39 = 0u - matrix_accumulator_39 + matrix_accumulator_28;

    if (condition_2) {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    if (condition_3) {
        matrix_accumulator_24 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_25 = parameters[parameter_input_skip_high_3];
    }

    if (!condition_4) {
        matrix_accumulator_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_4) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (!condition_2) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_13 = matrix_fragment_15;
    }

    if (condition_2) {
        matrix_accumulator_43 = matrix_fragment_22 + matrix_accumulator_27;
    }

    matrix_fragment_23 = matrix_accumulator_39 * 0x2u + reused_word_130;

    if (!condition_5) {
        matrix_fragment_8 = matrix_fragment_11;

        matrix_fragment_9 = matrix_fragment_11;

        matrix_fragment_10 = matrix_fragment_11;
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (!condition_4) {
        matrix_accumulator_35 =
            permute_signed(matrix_accumulator_35, matrix_accumulator_35, 0x5410u);
    }

    if (condition_4) {
        matrix_accumulator_50 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_51 = parameters[parameter_input_skip_high_3];
    }

    matrix_fragment_23 = uint32_t(int32_t(matrix_fragment_23) >> 0x1u);

    if (condition_4) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    matrix_fragment_19 = matrix_accumulator_39 * 0x2u + matrix_accumulator_26;

    if (condition_5) {
        matrix_accumulator_26 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_27 = parameters[parameter_input_skip_high_3];
    }

    if (!condition_4) {
        matrix_accumulator_32 = matrix_accumulator_35;

        matrix_accumulator_33 = matrix_accumulator_35;

        matrix_accumulator_34 = matrix_accumulator_35;
    }

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_accumulator_30 = reused_word_130 + 0x18u;

    if (condition_3) {
        matrix_fragment_23 = matrix_accumulator_31 * 0x200u + index_word_0;
    }

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), matrix_accumulator_32,
            matrix_accumulator_33, matrix_accumulator_34, matrix_accumulator_35, memory_ranges,
            error_counts);
    }

    matrix_accumulator_30 = matrix_accumulator_39 * 0x2u + matrix_accumulator_30;

    if (condition_4) {
        matrix_accumulator_45 = matrix_fragment_22 + matrix_accumulator_29;
    }

    matrix_fragment_18 = reused_word_130 + 0x10u;

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_19) >> 0x1u);

    matrix_accumulator_37 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_16, matrix_fragment_17) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_3) {
        matrix_fragment_21 = matrix_fragment_23 * 0x4u;
    }

    if (condition_5) {
        matrix_fragment_20 = matrix_accumulator_41 * 0x200u + index_word_0;
    }

    matrix_accumulator_30 =
        uint32_t(combine_words(matrix_accumulator_30, 0u) << 0x1u >> 32) + matrix_accumulator_30;

    if (condition_3) {
        split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                            int64_t(int32_t(matrix_fragment_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    if (!condition_5) {
        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_accumulator_29 = matrix_accumulator_43 * 0x200u + index_word_0;
    }

    if (condition_5) {
        matrix_fragment_17 = matrix_fragment_20 * 0x4u;
    }

    matrix_accumulator_43 = uint32_t(int32_t(matrix_accumulator_30) >> 0x1u);

    matrix_fragment_22 = matrix_accumulator_39 * 0x2u + matrix_fragment_18;

    if (condition_4) {
        index_word_0 = matrix_accumulator_45 * 0x200u + index_word_0;
    }

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!condition_3) {
        matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_5) {
        matrix_fragment_23 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5410u);
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_30, matrix_accumulator_31,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    matrix_accumulator_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_18, matrix_fragment_19) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_5) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    if (condition_2) {
        matrix_accumulator_25 = matrix_accumulator_29 * 0x4u;
    }

    if (condition_4) {
        matrix_accumulator_29 = index_word_0 * 0x4u;
    }

    index_word_0 = reused_word_130 + 0x20u;

    matrix_accumulator_45 = uint32_t(int32_t(matrix_fragment_22) >> 0x1u);

    if (!condition_3) {
        matrix_fragment_19 = permute_signed(matrix_fragment_16, matrix_fragment_16, 0x5410u);
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_46, matrix_accumulator_47,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    if (!condition_5) {
        matrix_fragment_20 = matrix_fragment_23;

        matrix_fragment_21 = matrix_fragment_23;

        matrix_fragment_22 = matrix_fragment_23;
    }

    if (!condition_2) {
        matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    index_word_0 = matrix_accumulator_39 * 0x2u + index_word_0;

    matrix_accumulator_36 = reused_word_130 + 0x28u;

    if (!condition_3) {
        matrix_fragment_16 = matrix_fragment_19;

        matrix_fragment_17 = matrix_fragment_19;

        matrix_fragment_18 = matrix_fragment_19;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_48, matrix_accumulator_49));
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_46, matrix_accumulator_47), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    if (!condition_2) {
        matrix_accumulator_27 =
            permute_signed(matrix_accumulator_27, matrix_accumulator_27, 0x5410u);
    }

    if (condition_4) {
        split_address_words(matrix_accumulator_50, matrix_accumulator_51,
                            int64_t(int32_t(matrix_accumulator_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_50, matrix_accumulator_51));
    }

    if (!condition_4) {
        matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_30, matrix_accumulator_31), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    matrix_accumulator_36 = matrix_accumulator_39 * 0x2u + matrix_accumulator_36;

    matrix_accumulator_47 = uint32_t(int32_t(index_word_0) >> 0x1u);

    index_word_0 = reused_word_130 + 0x30u;

    if (!condition_2) {
        matrix_accumulator_24 = matrix_accumulator_27;

        matrix_accumulator_25 = matrix_accumulator_27;

        matrix_accumulator_26 = matrix_accumulator_27;
    }

    if (!condition_4) {
        matrix_accumulator_31 =
            permute_signed(matrix_accumulator_29, matrix_accumulator_29, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_48, matrix_accumulator_49));
    }

    matrix_accumulator_36 =
        uint32_t(combine_words(matrix_accumulator_36, 0u) << 0x1u >> 32) + matrix_accumulator_36;

    split_address_words(matrix_accumulator_44, matrix_accumulator_45,
                        int64_t(int32_t(matrix_accumulator_45)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!condition_4) {
        matrix_accumulator_29 = matrix_accumulator_31;
    }

    index_word_0 = matrix_accumulator_39 * 0x2u + index_word_0;

    split_address_words(matrix_accumulator_42, matrix_accumulator_43,
                        int64_t(int32_t(matrix_accumulator_43)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!condition_4) {
        matrix_accumulator_30 = matrix_accumulator_31;
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_48, matrix_accumulator_49), matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_accumulator_50, matrix_accumulator_51,
                            int64_t(int32_t(matrix_accumulator_28)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_50, matrix_accumulator_51));
    }

    if (!condition_4) {
        matrix_accumulator_28 = matrix_accumulator_31;
    }

    matrix_fragment_87 = uint32_t(int32_t(matrix_accumulator_36) >> 0x1u);

    matrix_accumulator_44 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_44, matrix_accumulator_45) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_46, matrix_accumulator_47,
                        int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_accumulator_42 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_42, matrix_accumulator_43) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_49 = uint32_t(int32_t(index_word_0) >> 0x1u);

    index_word_0 = reused_word_130 + 0x38u;

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_50, matrix_accumulator_51), matrix_accumulator_28,
            matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31, memory_ranges,
            error_counts);
    }

    split_address_words(matrix_fragment_86, matrix_fragment_87,
                        int64_t(int32_t(matrix_fragment_87)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    index_word_0 = matrix_accumulator_39 * 0x2u + index_word_0;

    matrix_accumulator_43 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_46, matrix_accumulator_47) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                        int64_t(int32_t(matrix_accumulator_49)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_fragment_86 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0x40000ull, memory_ranges,
        error_counts);

    index_word_0 = uint32_t(combine_words(index_word_0, 0u) << 0x1u >> 32) + index_word_0;

    matrix_fragment_84 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_48, matrix_accumulator_49) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_39 = uint32_t(int32_t(index_word_0) >> 0x1u);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_39)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    index_word_0 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_fragment_85 = (index_word_131 << 0x3u) + (0u - uniform_index_word_4);

    matrix_accumulator_36 = matrix_fragment_85 + 0x8u;

    condition_0 = int32_t(matrix_accumulator_36) > int32_t(0u);

    uniform_index_word_4 = (uniform_block_index_9 << 0x3u) + (0u - uniform_index_word_5);

    uniform_address_word_6 = uniform_index_word_4 + 0x8u;

    uniform_index_word_5 = uniform_index_word_4 + 0x7u;

    uniform_condition_0 = int32_t(uniform_address_word_6) > int32_t(0u);

    uniform_index_word_5 = uint32_t(combine_words(uniform_index_word_5, 0u) << 0x1du >> 32) + 0x1u;

    index_word_134 = max(int32_t(0u), int32_t(matrix_fragment_85));

    index_word_136 = max(int32_t(0u), int32_t(uniform_index_word_4));

    uniform_index_word_5 = uniform_condition_0 ? uniform_index_word_5 : 0u;

    index_word_139 = 0u;

    index_word_134 = uint32_t(uint32_t(index_word_134) >> 0x3u);

    index_word_136 = uint32_t(uint32_t(index_word_136) >> 0x3u);

    index_word_135 = min(int32_t(index_word_135), int32_t(uniform_index_word_5));

    matrix_accumulator_128 = unpack_fp8_bits(matrix_accumulator_4);

    matrix_accumulator_126 = unpack_fp8_bits(matrix_accumulator_4 >> 16);

    matrix_accumulator_129 = unpack_fp8_bits(matrix_accumulator_5);

    matrix_accumulator_127 = unpack_fp8_bits(matrix_accumulator_5 >> 16);

    matrix_accumulator_124 = unpack_fp8_bits(matrix_accumulator_6);

    matrix_accumulator_122 = unpack_fp8_bits(matrix_accumulator_6 >> 16);

    matrix_accumulator_125 = unpack_fp8_bits(matrix_accumulator_7);

    matrix_accumulator_123 = unpack_fp8_bits(matrix_accumulator_7 >> 16);

    matrix_accumulator_4 = unpack_fp8_bits(matrix_fragment_12);

    matrix_accumulator_5 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_38 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_8);

    matrix_accumulator_50 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_9);

    matrix_accumulator_51 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_accumulator_48 = unpack_fp8_bits(matrix_fragment_10);

    matrix_accumulator_46 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_accumulator_49 = unpack_fp8_bits(matrix_fragment_11);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_accumulator_32);

    matrix_fragment_8 = unpack_fp8_bits(matrix_accumulator_33);

    matrix_accumulator_47 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_accumulator_39 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_accumulator_40 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_accumulator_34);

    matrix_accumulator_6 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_accumulator_7 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_14 = unpack_fp8_bits(matrix_accumulator_35);

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_15 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_4),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_accumulator_45 = unpack_fp8_bits(matrix_fragment_21);

    matrix_accumulator_120 = unpack_fp8_bits(matrix_fragment_16);

    matrix_accumulator_118 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_accumulator_121 = unpack_fp8_bits(matrix_fragment_17);

    matrix_accumulator_119 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_accumulator_116 = unpack_fp8_bits(matrix_fragment_18);

    matrix_accumulator_4 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_accumulator_117 = unpack_fp8_bits(matrix_fragment_19);

    matrix_accumulator_5 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_fragment_22);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_accumulator_24 >> 16);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_16 = unpack_fp8_bits(matrix_accumulator_28);

    matrix_accumulator_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_17 = unpack_fp8_bits(matrix_accumulator_29);

    matrix_accumulator_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_6),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_44 = unpack_fp8_bits(matrix_fragment_20);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_fragment_14 = unpack_fp8_bits(matrix_accumulator_24);

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_fragment_15 = unpack_fp8_bits(matrix_accumulator_25);

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_accumulator_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_accumulator_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_42));

    matrix_accumulator_42 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_20 = unpack_fp8_bits(matrix_accumulator_25 >> 16);

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_fragment_21 = unpack_fp8_bits(matrix_accumulator_28 >> 16);

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_26);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_fragment_89 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_accumulator_25 = unpack_fp8_bits(matrix_accumulator_27);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_fragment_90 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_fragment_22 = unpack_fp8_bits(matrix_accumulator_29 >> 16);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_fragment_87 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_43));

    matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_30);

    matrix_accumulator_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_31);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_28 = matrix_fragment_85 + 0x7u;

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_88 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_84 = uint32_t(combine_words(matrix_accumulator_28, 0u) << 0x1du >> 32) + 0x1u;

    matrix_fragment_91 = unpack_fp8_bits(matrix_accumulator_30 >> 16);

    matrix_fragment_92 = unpack_fp8_bits(matrix_accumulator_31 >> 16);

    matrix_fragment_84 = condition_0 ? matrix_fragment_84 : 0u;

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    index_word_133 = min(int32_t(matrix_fragment_84), int32_t(index_word_141));

    matrix_accumulator_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_4),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(index_word_0));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(index_word_0));
    do {
        uniform_address_word_6 = 0u;

        matrix_fragment_85 = threadIdx.x & 31u;

        index_word_0 = index_word_139 * 0x40u;

        uniform_index_word_4 = 0x400u;

        index_word_142 = uint32_t(uint32_t(index_word_139) >> 0x6u);

        index_word_143 = threadIdx.z;

        index_word_0 = index_word_0 & 0x1000u;

        index_word_142 = ~index_word_142 & 0x1u;

        uniform_index_word_4 = (uniform_address_word_6 << 0x18u) + uniform_index_word_4;

        index_word_0 = index_word_0 + uniform_index_word_4;

        __syncwarp(__activemask());

        tile_word_144 = matrix_fragment_85 * 0x10u + index_word_0;

        index_word_0 = threadIdx.x;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144, matrix_fragment_84, matrix_fragment_85, matrix_fragment_86,
            matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0x200u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0x400u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0x600u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0x800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0xa00u, matrix_fragment_104, matrix_fragment_105,
            matrix_fragment_106, matrix_fragment_107, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0xc00u, matrix_fragment_108, matrix_fragment_109,
            matrix_fragment_110, matrix_fragment_111, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_144 + 0xe00u, matrix_fragment_112, matrix_fragment_113,
            matrix_fragment_114, matrix_fragment_115, shared_bytes, memory_ranges, error_counts);

        index_word_0 = index_word_0 | thread_row_137;

        index_word_0 = index_word_0 | index_word_143;

        index_word_143 = index_word_139 + 0x40u;

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_128,
                                 matrix_accumulator_129);

        condition_0 = (index_word_139 & 0x40u) != 0;

        condition_2 = uint32_t(index_word_139) >= uint32_t(0x180u);

        index_word_139 = index_word_143;

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_32,
                                 matrix_accumulator_33);
        if (condition_0) {
            condition_0 = int32_t(index_word_136) >= int32_t(index_word_135);

            condition_0 = uint32_t(index_word_0) != 0u || condition_0;
            if (!condition_0) {
                matrix_fragment_52 = block_index_138 * 0x200u + index_word_143;

                matrix_fragment_54 = index_word_136;

                matrix_fragment_57 = uint32_t(uint32_t(matrix_fragment_52) >> 0x7u);
                do {
                    condition_0 = int32_t(index_word_134) >= int32_t(index_word_133);
                    if (!condition_0) {
                        matrix_fragment_52 = matrix_fragment_54 * 0x4u + matrix_fragment_57;

                        matrix_fragment_56 = index_word_134;

                        matrix_fragment_59 = matrix_fragment_52 * index_word_141;
                        do {
                            matrix_fragment_52 = parameters[parameter_wait_low_10];
                            matrix_fragment_53 = parameters[parameter_wait_high_11];

                            matrix_fragment_55 = matrix_fragment_59 + matrix_fragment_56;

                            split_address_words(
                                matrix_fragment_52, matrix_fragment_53,
                                int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_52, matrix_fragment_53));

                            matrix_fragment_55 = pipeline_read<CheckBounds>(
                                combine_words(matrix_fragment_52, matrix_fragment_53),
                                memory_ranges, error_counts);

                            matrix_fragment_56 = matrix_fragment_56 + 0x1u;

                            condition_1 = uint32_t(matrix_fragment_56) != uint32_t(index_word_133);

                            condition_0 = int32_t(matrix_fragment_55) > -1;
                            if (!condition_0) {
                                do {
                                    __nanosleep(0x40u);

                                    matrix_fragment_55 = pipeline_read<CheckBounds>(
                                        combine_words(matrix_fragment_52, matrix_fragment_53),
                                        memory_ranges, error_counts);

                                    condition_0 = int32_t(matrix_fragment_55) > -1;
                                } while (!condition_0);
                            }
                        } while (condition_1);
                    }
                    matrix_fragment_54 = matrix_fragment_54 + 0x1u;

                    condition_0 = uint32_t(matrix_fragment_54) != uint32_t(index_word_135);
                } while (condition_0);
            }
            __syncwarp(__activemask());

            phoenix_architecture::finish_compatibility_copies();
        }
        matrix_fragment_52 = parameters[parameter_tensor_height_16];

        uniform_index_word_18 = uniform_block_index_9 << (0x1u & 31u);

        matrix_fragment_102 = index_word_131 * 0x2u;

        uniform_index_word_8 = broadcast_active_lane(index_word_142);

        matrix_fragment_61 = block_index_138 * 0x200u + index_word_143;

        uniform_index_word_4 = 0x400u;

        matrix_fragment_58 = thread_row_137 * 0x200u;

        matrix_fragment_54 = index_word_140 + uniform_index_word_18;

        matrix_fragment_100 = parameters[parameter_tensor_width_17];

        uniform_address_word_6 = uniform_index_word_4 + 0x2000u;

        matrix_fragment_56 = parameters[parameter_tensor_width_17];

        matrix_fragment_55 = parameters[parameter_tensor_height_16];

        matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_52) >> 0x1fu);

        matrix_fragment_53 =
            uint32_t(combine_words(matrix_fragment_53, 0u) << 0x2u >> 32) + matrix_fragment_52;

        matrix_fragment_52 = uint32_t(uint32_t(thread_row_137) >> 0x1u);

        uniform_index_word_5 = 0u;

        matrix_fragment_101 = uint32_t(int32_t(matrix_fragment_53) >> 0x2u);

        matrix_fragment_57 = uint32_t(int32_t(matrix_fragment_100) >> 0x1fu);

        matrix_fragment_53 = matrix_fragment_52 & 0x1u;

        condition_4 = int32_t(matrix_fragment_54) >= int32_t(matrix_fragment_101) && condition_6;

        matrix_fragment_63 = matrix_fragment_56 & 0xfffffffcu;

        matrix_fragment_57 =
            uint32_t(combine_words(matrix_fragment_57, 0u) << 0x2u >> 32) + matrix_fragment_100;

        matrix_fragment_60 = matrix_fragment_102 | matrix_fragment_53;

        matrix_fragment_103 = uint32_t(int32_t(matrix_fragment_57) >> 0x2u);

        matrix_fragment_62 = matrix_fragment_55 & 0xfffffffcu;

        condition_0 = int32_t(matrix_fragment_54) >= int32_t(matrix_fragment_101);

        condition_1 = uint32_t(matrix_fragment_63) == uint32_t(0x4u) || condition_4;

        condition_3 = uint32_t(matrix_fragment_63) != uint32_t(0x4u);

        matrix_fragment_52 = condition_4 ? matrix_fragment_60 : 0u;

        matrix_fragment_56 = thread_row_137 & 0x1u;

        condition_0 = uint32_t(matrix_fragment_62) == uint32_t(0x4u) || !condition_0;

        condition_1 = int32_t(matrix_fragment_60) < int32_t(matrix_fragment_103) || condition_1;

        matrix_fragment_52 = !condition_3 ? matrix_fragment_52 : matrix_fragment_60;

        matrix_fragment_57 = matrix_fragment_56 * 0x200u;

        uniform_address_word_6 = (uniform_index_word_5 << 0x18u) + uniform_address_word_6;

        uniform_address_word_7 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

        condition_0 = condition_1 & condition_0;

        uniform_index_word_5 = (uniform_index_word_8 << 0x3u) + uniform_address_word_6;

        condition_1 = uint32_t(matrix_fragment_62) != uint32_t(0x4u) && condition_0;

        if (condition_0) {
            matrix_fragment_53 =
                uint32_t(combine_words(matrix_fragment_61, 0u) << 0x1bu >> 32) + matrix_fragment_56;
        }

        if (condition_1) {
            matrix_fragment_52 = matrix_fragment_54 * matrix_fragment_103 + matrix_fragment_52;
        }

        matrix_fragment_54 = matrix_fragment_57 | 0x7fc00u & matrix_fragment_58;

        if (condition_0) {
            matrix_fragment_55 = matrix_fragment_52 * 0x10u + matrix_fragment_53;
        }

        split_address_words(matrix_fragment_52, matrix_fragment_53, 0ull);

        matrix_fragment_59 = matrix_fragment_54 + uniform_address_word_7;

        if (condition_0) {
            matrix_fragment_55 = matrix_fragment_55 * 0x80u;

            matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_55) >> 0x1fu);

            matrix_fragment_52 = matrix_fragment_55;
        }
        if (!condition_0) {
            matrix_fragment_52 = threadIdx.x & 31u;

            uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                 select_half_operand<2, false, false>(0u));

            matrix_fragment_59 = index_word_142 * 0x1000u + matrix_fragment_59;

            matrix_fragment_53 = uniform_index_word_4;

            matrix_fragment_54 = uniform_index_word_4;

            matrix_fragment_59 = matrix_fragment_52 * 0x10u + matrix_fragment_59;

            matrix_fragment_52 = permute_signed(matrix_fragment_53, matrix_fragment_54, 0x5410u);

            matrix_fragment_53 = matrix_fragment_52;

            matrix_fragment_54 = matrix_fragment_52;

            matrix_fragment_55 = matrix_fragment_52;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_59, matrix_fragment_52, matrix_fragment_53,
                matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_index_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                matrix_fragment_54 = parameters[parameter_input_low_0];
                matrix_fragment_55 = parameters[parameter_input_high_1];

                uniform_address_word_6 = broadcast_active_lane(matrix_fragment_59);

                uniform_index_word_4 = broadcast_active_lane(index_word_142);

                uniform_index_word_4 = (uniform_index_word_4 << 0xcu) + uniform_address_word_6;

                uniform_address_word_6 = 0x20u;

                const uint64_t byte_address_2 =
                    scaled_word_address(combine_words(matrix_fragment_54, matrix_fragment_55),
                                        combine_words(matrix_fragment_52, matrix_fragment_53));
                matrix_fragment_54 = uint32_t(byte_address_2);

                matrix_fragment_52 = uint32_t(byte_address_2 >> 32);

                uniform_address_word_14 = broadcast_active_lane(matrix_fragment_54);

                uniform_address_word_15 = broadcast_active_lane(matrix_fragment_52);

                matrix_fragment_52 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_index_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_6 * 16u, uniform_index_word_5, shared_bytes, memory_ranges,
                    error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_index_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        matrix_fragment_52 = thread_row_137 + 0x4u;

        matrix_fragment_53 = matrix_fragment_58 + 0x800u;

        matrix_fragment_52 = uint32_t(uint32_t(matrix_fragment_52) >> 0x2u);

        matrix_fragment_53 = matrix_fragment_53 & 0xff800u;

        matrix_fragment_54 = matrix_fragment_52 + uniform_index_word_18;

        matrix_fragment_58 = matrix_fragment_53 | 0x400u & matrix_fragment_58;

        condition_4 = int32_t(matrix_fragment_54) >= int32_t(matrix_fragment_101) && condition_6;

        condition_0 = int32_t(matrix_fragment_54) >= int32_t(matrix_fragment_101);

        matrix_fragment_58 = matrix_fragment_57 | matrix_fragment_58;

        condition_1 = uint32_t(matrix_fragment_63) == uint32_t(0x4u) || condition_4;

        condition_0 = uint32_t(matrix_fragment_62) == uint32_t(0x4u) || !condition_0;

        matrix_fragment_52 = condition_4 ? matrix_fragment_60 : 0u;

        matrix_fragment_57 = matrix_fragment_58 + uniform_address_word_7;

        condition_1 = int32_t(matrix_fragment_60) < int32_t(matrix_fragment_103) || condition_1;

        matrix_fragment_52 = !condition_3 ? matrix_fragment_52 : matrix_fragment_60;

        condition_0 = condition_1 & condition_0;

        condition_1 = uint32_t(matrix_fragment_62) != uint32_t(0x4u) && condition_0;

        if (condition_0) {
            matrix_fragment_55 =
                uint32_t(combine_words(matrix_fragment_61, 0u) << 0x1bu >> 32) + matrix_fragment_56;
        }

        if (condition_1) {
            matrix_fragment_52 = matrix_fragment_54 * matrix_fragment_103 + matrix_fragment_52;
        }

        if (condition_0) {
            matrix_fragment_55 = matrix_fragment_52 * 0x10u + matrix_fragment_55;
        }

        split_address_words(matrix_fragment_52, matrix_fragment_53, 0ull);

        if (condition_0) {
            matrix_fragment_55 = matrix_fragment_55 * 0x80u;

            matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_55) >> 0x1fu);

            matrix_fragment_52 = matrix_fragment_55;
        }
        if (!condition_0) {
            matrix_fragment_56 = threadIdx.x & 31u;

            uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                 select_half_operand<2, false, false>(0u));

            matrix_fragment_57 = index_word_142 * 0x1000u + matrix_fragment_57;

            matrix_fragment_52 = uniform_index_word_4;

            matrix_fragment_53 = uniform_index_word_4;

            matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_53, 0x5410u);

            matrix_fragment_53 = matrix_fragment_52;

            matrix_fragment_54 = matrix_fragment_52;

            matrix_fragment_55 = matrix_fragment_52;

            matrix_fragment_56 = matrix_fragment_56 * 0x10u + matrix_fragment_57;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_56, matrix_fragment_52, matrix_fragment_53,
                matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_index_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                matrix_fragment_54 = parameters[parameter_input_low_0];
                matrix_fragment_55 = parameters[parameter_input_high_1];

                uniform_index_word_4 = broadcast_active_lane(index_word_142);

                uniform_address_word_6 = broadcast_active_lane(matrix_fragment_57);

                uniform_index_word_4 = (uniform_index_word_4 << 0xcu) + uniform_address_word_6;

                uniform_address_word_6 = 0x20u;

                // Later control flow also consumes this address-overflow predicate.
                condition_0 =
                    (uint64_t(matrix_fragment_52) << 2) + matrix_fragment_54 > 0xffffffffull;
                const uint64_t byte_address_3 =
                    scaled_word_address(combine_words(matrix_fragment_54, matrix_fragment_55),
                                        combine_words(matrix_fragment_52, matrix_fragment_53));
                matrix_fragment_54 = uint32_t(byte_address_3);

                matrix_fragment_52 = uint32_t(byte_address_3 >> 32);

                uniform_address_word_14 = broadcast_active_lane(matrix_fragment_54);

                uniform_address_word_15 = broadcast_active_lane(matrix_fragment_52);

                matrix_fragment_52 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_index_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_6 * 16u, uniform_index_word_5, shared_bytes, memory_ranges,
                    error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_index_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        matrix_fragment_104 = threadIdx.x & 31u;

        matrix_fragment_52 = parameters[parameter_trained_weights_low_8];
        matrix_fragment_53 = parameters[parameter_trained_weights_high_9];

        matrix_fragment_54 = reused_word_130 * 0x8u;

        matrix_fragment_55 = matrix_fragment_61 * 0x80u + matrix_fragment_54;

        split_address_words(matrix_fragment_52, matrix_fragment_53,
                            int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_52, matrix_fragment_53));

        split_address_words(matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_52, matrix_fragment_53));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_84, matrix_fragment_85),
                                            matrix_fragment_80, matrix_fragment_81,
                                            matrix_fragment_82, matrix_fragment_83, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x200ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x400ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x600ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x4000ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x4200ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x4400ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x4600ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = uniform_index_word_5;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_86 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_0 = wait_copy_barrier(shared_tile, uniform_index_word_5,
                                        async_tokens[(((uniform_index_word_5) >> 3) & 1)]);
#else
        phoenix_architecture::finish_compatibility_copies();
        condition_0 = true;
#endif
        if (!condition_0) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_0 = wait_copy_barrier(shared_tile, uniform_index_word_5,
                                                async_tokens[(((uniform_index_word_5) >> 3) & 1)]);
#else
                phoenix_architecture::finish_compatibility_copies();
                condition_0 = true;
#endif
            } while (!condition_0);
        }
    } while (!condition_2);
    matrix_fragment_105 = (matrix_fragment_104 << 0x4u) + uniform_address_word_7;

    __syncwarp(__activemask());

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1000u, matrix_fragment_84, matrix_fragment_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    condition_0 = int32_t(matrix_fragment_102) >= int32_t(matrix_fragment_103);

    reused_word_130 = reused_word_130 * 0x4u;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1400u, matrix_fragment_88, matrix_fragment_89,
        matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

    condition_2 = int32_t(matrix_fragment_101) <= int32_t(uniform_index_word_18) || condition_0;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1800u, matrix_fragment_92, matrix_fragment_93,
        matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1c00u, matrix_fragment_96, matrix_fragment_97,
        matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_4,
                             matrix_accumulator_5);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_50,
                             matrix_accumulator_51);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_22,
                             matrix_fragment_23);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_14,
                             matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_18,
                             matrix_fragment_19);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_16,
                             matrix_fragment_17);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1a00u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_6,
                             matrix_accumulator_7);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_10,
                             matrix_fragment_11);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1200u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1600u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_20,
                             matrix_fragment_21);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_105 + 0x1e00u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_40,
                             matrix_accumulator_41);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_6,
                             matrix_accumulator_7);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_56, matrix_fragment_57, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_30,
                             matrix_accumulator_31);

    matrix_fragment_16 = parameters[parameter_output_encoder_skip_low_4];
    matrix_fragment_17 = parameters[parameter_output_encoder_skip_high_5];

    matrix_fragment_18 = parameters[parameter_LW_19];

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                             matrix_accumulator_129);

    matrix_fragment_19 = parameters[parameter_LH_18];

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_4,
                             matrix_accumulator_5);

    matrix_fragment_8 = matrix_fragment_102 + 0x1u;

    matrix_fragment_9 = matrix_fragment_103 * uniform_index_word_18 + matrix_fragment_102;

    condition_1 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_103);

    matrix_fragment_9 = matrix_fragment_9 * 0x800u + reused_word_130;

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_2,
                             matrix_accumulator_3);

    condition_3 = int32_t(matrix_fragment_101) <= int32_t(uniform_index_word_18) || condition_1;

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_50,
                             matrix_accumulator_51);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_40,
                             matrix_accumulator_41);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_56, matrix_fragment_57, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_32,
                             matrix_accumulator_33);

    split_address_words(matrix_fragment_22, matrix_fragment_23,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_16, matrix_fragment_17));
    if (!condition_2) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_5),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_4),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    if (!condition_3) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x2000ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x2200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    uniform_index_word_4 = uniform_index_word_18 + 0x1u;

    matrix_fragment_103 = matrix_fragment_103 * uniform_index_word_18 + matrix_fragment_103;

    matrix_fragment_8 = uint32_t(uint32_t(matrix_fragment_104) >> 0x2u);

    matrix_fragment_22 = matrix_fragment_104 * 0x2u;

    matrix_fragment_9 = uint32_t(uint32_t(matrix_fragment_104) >> 0x4u);

    condition_0 = int32_t(matrix_fragment_101) <= int32_t(uniform_index_word_4) || condition_0;

    matrix_fragment_103 = matrix_fragment_102 + matrix_fragment_103;

    matrix_fragment_8 = matrix_fragment_8 & 0x2u;

    matrix_fragment_9 = matrix_fragment_9 & 0x1u;

    matrix_fragment_21 = matrix_fragment_103 * 0x800u + reused_word_130;

    condition_2 = uint32_t(index_word_0) != 0u;

    condition_3 = (matrix_fragment_8 | matrix_fragment_9) != 0;

    index_word_0 = matrix_fragment_8 | matrix_fragment_9;

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        int64_t(int32_t(matrix_fragment_21)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_16, matrix_fragment_17));
    if (!condition_0) {
        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_6),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    condition_1 = int32_t(matrix_fragment_101) <= int32_t(uniform_index_word_4) || condition_1;

    matrix_fragment_9 = matrix_fragment_22 & 0x8u;

    condition_0 = uint32_t(index_word_0) != 0u;

    matrix_fragment_16 = matrix_fragment_9 | 0x3u & matrix_fragment_104;
    if (!condition_1) {
        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                           select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                           select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2000ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_128), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_10 = 0x4u | matrix_fragment_16;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_129), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_12 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_11 = matrix_fragment_8;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_11 = !condition_4 ? matrix_fragment_9 : matrix_fragment_12;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_128), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_12 = result_word;
    }

    matrix_fragment_9 = 0x10u | matrix_fragment_16;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_129), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_12 = matrix_fragment_8;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_12 = !condition_4 ? matrix_fragment_13 : matrix_fragment_14;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_128), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_8 = 0x14u | matrix_fragment_16;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_129), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_13 = matrix_fragment_14;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_13 = !condition_4 ? matrix_fragment_15 : matrix_fragment_20;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_128), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_128 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_129), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_129 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_128 = matrix_accumulator_129;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_128 = !condition_4 ? matrix_accumulator_2 : matrix_accumulator_3;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_34), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_35), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_36), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_37), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_3 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_3 = !condition_4 ? matrix_fragment_14 : matrix_fragment_15;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_34), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_35), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_36), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_37), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_14 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_14 = !condition_4 ? matrix_fragment_15 : matrix_fragment_20;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_34), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_35), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_36), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_37), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_15 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_15 = !condition_4 ? matrix_fragment_17 : matrix_fragment_20;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_34), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_35), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_36), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_37), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_34 = matrix_accumulator_35;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_34 = !condition_4 ? matrix_accumulator_36 : matrix_accumulator_37;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_126), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_127), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_50), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_51), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_17 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_17 = !condition_4 ? matrix_fragment_20 : matrix_fragment_21;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_126), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_127), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_50), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_51), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_20 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_20 = !condition_4 ? matrix_fragment_21 : matrix_fragment_22;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_126), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_127), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_50), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_51), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_21 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_21 = !condition_4 ? matrix_fragment_22 : matrix_fragment_23;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_126), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_126 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_127), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_127 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_50), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_51), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_51 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_126 = matrix_accumulator_127;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_126 = !condition_4 ? matrix_accumulator_50 : matrix_accumulator_51;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_38), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_40), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_41), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_22 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_22 = !condition_4 ? matrix_fragment_23 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_38), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_40), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_41), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_23 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_23 = !condition_4 ? matrix_accumulator_35 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_38), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_40), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_41), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_35 = matrix_accumulator_2;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_35 = !condition_4 ? matrix_accumulator_36 : matrix_accumulator_37;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_38), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_39), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_40), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_41), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_38 = matrix_accumulator_39;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_38 = !condition_4 ? matrix_accumulator_40 : matrix_accumulator_41;
        }
    }
    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                       select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                       select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_49), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_accumulator_38));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_15));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_124), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_48), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_13 = matrix_fragment_14;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_13 = !condition_4 ? matrix_fragment_15 : matrix_fragment_20;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_124), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_48), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_49), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_14 = matrix_fragment_15;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_14 = !condition_4 ? matrix_fragment_17 : matrix_fragment_20;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_124), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_48), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_49), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_15 = matrix_fragment_17;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_15 = !condition_4 ? matrix_fragment_20 : matrix_fragment_21;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_124), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_124 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_125 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_48), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_49), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_49 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_124 = matrix_accumulator_125;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_124 = !condition_4 ? matrix_accumulator_48 : matrix_accumulator_49;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_6), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_17 = matrix_fragment_20;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_17 = !condition_4 ? matrix_fragment_21 : matrix_fragment_22;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_6), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_20 = matrix_fragment_21;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_20 = !condition_4 ? matrix_fragment_22 : matrix_fragment_23;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_6), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_21 = matrix_fragment_22;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_21 = !condition_4 ? matrix_fragment_23 : matrix_accumulator_34;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_6), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_22 = matrix_accumulator_7;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_22 = !condition_4 ? matrix_fragment_76 : matrix_fragment_77;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_46), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_47), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_23 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_23 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_34;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_46), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_47), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_34 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_34 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_46), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_47), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_35 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_35 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_122 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_123 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_46), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_47), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_47 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_122 = matrix_accumulator_123;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_122 = !condition_4 ? matrix_accumulator_46 : matrix_accumulator_47;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_36 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_36 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_38;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_37 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_37 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_38;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_38 = matrix_accumulator_6;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_38 = !condition_4 ? matrix_accumulator_7 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_39 = matrix_fragment_87;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_39 = !condition_4 ? matrix_fragment_78 : matrix_fragment_79;
        }
    }
    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_accumulator_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                       select_half_operand<2, false, false>(matrix_accumulator_34));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_121), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_14 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_44), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_6 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_6),
                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_7 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                       select_half_operand<2, false, false>(matrix_fragment_21));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_14 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                       select_half_operand<2, false, false>(matrix_accumulator_38));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_45), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_15 = matrix_fragment_17;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_15 = !condition_4 ? matrix_fragment_20 : matrix_fragment_21;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_121), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_44), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_45), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_20 = matrix_fragment_17;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_20 = !condition_4 ? matrix_fragment_21 : matrix_fragment_22;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_121), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_44), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_45), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_17 = matrix_fragment_21;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_17 = !condition_4 ? matrix_fragment_22 : matrix_fragment_23;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_120 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_121), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_121 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_44), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_44 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_45), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_45 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_120 = matrix_accumulator_121;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_120 = !condition_4 ? matrix_accumulator_44 : matrix_accumulator_45;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_89), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_21 = matrix_fragment_22;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_21 = !condition_4 ? matrix_fragment_23 : matrix_accumulator_34;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_89), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_22 = matrix_fragment_23;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_22 = !condition_4 ? matrix_accumulator_34 : matrix_accumulator_35;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_89), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_23 = matrix_accumulator_34;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_23 = !condition_4 ? matrix_accumulator_35 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_89), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_88 = matrix_fragment_89;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_88 = !condition_4 ? matrix_fragment_72 : matrix_fragment_73;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_42), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_43), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_34 = matrix_accumulator_35;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_34 = !condition_4 ? matrix_accumulator_36 : matrix_accumulator_37;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_42), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_43), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_35 = matrix_accumulator_36;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_35 = !condition_4 ? matrix_accumulator_37 : matrix_accumulator_38;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_42), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_43), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_36 = matrix_accumulator_37;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_36 = !condition_4 ? matrix_accumulator_38 : matrix_accumulator_39;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_119 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_42), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_42 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_43), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_43 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_37 = matrix_accumulator_119;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_37 = !condition_4 ? matrix_accumulator_42 : matrix_accumulator_43;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_90), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_91), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_38 = matrix_accumulator_39;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_38 = !condition_4 ? matrix_accumulator_40 : matrix_accumulator_41;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_90), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_91), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_42 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_39 = matrix_accumulator_40;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_39 = !condition_4 ? matrix_accumulator_41 : matrix_accumulator_42;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_90), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_91), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_42 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_43 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_40 = matrix_accumulator_41;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_40 = !condition_4 ? matrix_accumulator_42 : matrix_accumulator_43;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_90), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_91), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_41 = matrix_fragment_91;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_41 = !condition_4 ? matrix_fragment_74 : matrix_fragment_75;
        }
    }
    matrix_fragment_17 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                       select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_fragment_23 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_22));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_accumulator_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_accumulator_40 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                       select_half_operand<2, false, false>(matrix_accumulator_41));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_16), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_accumulator_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_40));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_17 = matrix_fragment_20;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_17 = !condition_4 ? matrix_fragment_22 : matrix_fragment_23;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_10), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_20 = matrix_fragment_22;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_20 = !condition_4 ? matrix_fragment_23 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_22 = matrix_fragment_23;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_22 = !condition_4 ? matrix_accumulator_35 : matrix_accumulator_36;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_23 = matrix_accumulator_117;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_23 = !condition_4 ? matrix_fragment_84 : matrix_fragment_85;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_24), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_35 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_25), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_26), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_27), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_35 = matrix_accumulator_36;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_35 = !condition_4 ? matrix_accumulator_37 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_24), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_25), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_26), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_27), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_36 = matrix_accumulator_37;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_36 = !condition_4 ? matrix_accumulator_39 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_24), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_25), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_26), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_27), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_37 = matrix_accumulator_39;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_37 = !condition_4 ? matrix_accumulator_40 : matrix_accumulator_41;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_24), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_24 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_25), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_25 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_26), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_27), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_27 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_24 = matrix_accumulator_25;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_24 = !condition_4 ? matrix_accumulator_26 : matrix_accumulator_27;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_4), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_25 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_5), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_28), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_27 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_29), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_25 = matrix_accumulator_26;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_25 = !condition_4 ? matrix_accumulator_27 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_4), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_5), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_27 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_28), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_29), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_26 = matrix_accumulator_27;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_26 = !condition_4 ? matrix_accumulator_39 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_4), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_27 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_5), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_28), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_29), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_41 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_27 = matrix_accumulator_39;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_27 = !condition_4 ? matrix_accumulator_40 : matrix_accumulator_41;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_4), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_4 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_5), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_28), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_29), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_4 = matrix_accumulator_5;

        if (condition_1) {
            condition_4 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_4 = !condition_4 ? matrix_accumulator_28 : matrix_accumulator_29;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_33), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_30), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_31), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_32), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }
    if (condition_3) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_accumulator_5 = matrix_accumulator_28;

        if (condition_1) {
            condition_3 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_accumulator_5 = !condition_3 ? matrix_accumulator_29 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_30), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_31), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_32), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_33), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_16 = matrix_accumulator_28;

        if (condition_1) {
            condition_3 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_16 = !condition_3 ? matrix_accumulator_29 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_30), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_31), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_32), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_33), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_40 = result_word;
    }
    if (condition_0) {
        condition_1 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_10 = matrix_accumulator_28;

        if (condition_1) {
            condition_3 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_10 = !condition_3 ? matrix_accumulator_29 : matrix_accumulator_40;
        }
    }
    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_30), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_accumulator_29 = uint32_t(int32_t(matrix_fragment_18) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_31), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_31 = result_word;
    }

    matrix_accumulator_28 = uint32_t(int32_t(matrix_fragment_19) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_32), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_32 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_33), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_33 = result_word;
    }
    if (condition_0) {
        condition_0 = uint32_t(index_word_0) != uint32_t(0x1u);

        matrix_fragment_9 = matrix_accumulator_31;

        if (condition_0) {
            condition_1 = uint32_t(index_word_0) != uint32_t(0x2u);

            matrix_fragment_9 = !condition_1 ? matrix_accumulator_32 : matrix_accumulator_33;
        }
    }
    matrix_accumulator_29 =
        uint32_t(combine_words(matrix_accumulator_29, 0u) << 0x2u >> 32) + matrix_fragment_18;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_28 =
        uint32_t(combine_words(matrix_accumulator_28, 0u) << 0x2u >> 32) + matrix_fragment_19;

    matrix_fragment_22 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    index_word_0 = uint32_t(int32_t(matrix_accumulator_29) >> 0x2u);

    matrix_accumulator_35 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_28 = uint32_t(int32_t(matrix_accumulator_28) >> 0x2u);

    matrix_accumulator_24 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                       select_half_operand<2, false, false>(matrix_accumulator_24));

    condition_0 = int32_t(index_word_131) >= int32_t(index_word_0);

    matrix_accumulator_25 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                       select_half_operand<2, false, false>(matrix_accumulator_26));

    matrix_accumulator_4 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                       select_half_operand<2, false, false>(matrix_accumulator_4));

    condition_0 = int32_t(matrix_accumulator_28) <= int32_t(uniform_block_index_9) || condition_0;

    matrix_accumulator_5 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_accumulator_24 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                       select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_accumulator_4 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                       select_half_operand<2, false, false>(matrix_accumulator_4));

    matrix_accumulator_5 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_2), 0x34003400u);

    matrix_fragment_10 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_3), 0x34003400u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x34003400u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x34003400u);

    matrix_accumulator_6 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_6), 0x34003400u);

    matrix_accumulator_7 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_7), 0x34003400u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x34003400u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x34003400u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x34003400u);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x34003400u);

    matrix_accumulator_34 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_34), 0x34003400u);

    matrix_accumulator_38 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x34003400u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x34003400u);

    matrix_accumulator_24 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_24), 0x34003400u);

    matrix_accumulator_4 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_4), 0x34003400u);

    matrix_accumulator_5 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_5), 0x34003400u);
    if (!condition_0) {
        matrix_accumulator_2 = parameters[parameter_output_pool_low_6];
        matrix_accumulator_3 = parameters[parameter_output_pool_high_7];

        index_word_131 = index_word_0 * uniform_block_index_9 + index_word_131;

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_5),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                          select_half_operand<2, false, false>(0u));

        index_word_131 = index_word_131 * 0x800u + reused_word_130;

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                          select_half_operand<2, false, false>(matrix_accumulator_5));

        matrix_accumulator_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_4),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(index_word_131)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                          select_half_operand<2, false, false>(matrix_accumulator_38));

        matrix_accumulator_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_accumulator_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_6),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_104)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                          select_half_operand<2, false, false>(matrix_accumulator_34));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_accumulator_4,
            matrix_accumulator_5, matrix_accumulator_6, matrix_accumulator_7, memory_ranges,
            error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    phoenix_architecture::finish_compatibility_copies();

    if (condition_2) {
        return;
    }

    matrix_accumulator_2 = parameters[parameter_publication_low_12];
    matrix_accumulator_3 = parameters[parameter_publication_high_13];

    matrix_fragment_100 = matrix_fragment_100 + 0xffffffffu;

    matrix_accumulator_5 = uint32_t(int32_t(matrix_fragment_100) >> 0x1fu);

    matrix_accumulator_5 =
        uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x3u >> 32) + matrix_fragment_100;

    matrix_accumulator_5 =
        uint32_t(int64_t(int32_t(matrix_accumulator_5)) * (1ll << 0x1du) >> 32) + 0x1u;

    matrix_accumulator_5 = matrix_accumulator_5 * uniform_index_word_18 + block_index_132;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__ __maxnreg__(144) void phoenix_global_attention_split_heads_512_projection_pool(KernelArguments parameters) {
    run_global_attention_split_heads_512_projection_pool<false>(parameters.words, nullptr, nullptr);
}
