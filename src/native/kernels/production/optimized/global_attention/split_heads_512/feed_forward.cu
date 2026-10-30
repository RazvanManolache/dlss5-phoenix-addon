#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/feed_forward.cu
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
    parameter_output_low_2 = 2,
    parameter_output_high_3 = 3,
    parameter_trained_weights_low_4 = 4,
    parameter_trained_weights_high_5 = 5,
    parameter_tensor_height_6 = 6,
    parameter_tensor_width_7 = 7,
    parameter_wait_optional_low_8 = 8,
    parameter_wait_optional_high_9 = 9,
    parameter_optional_control_10 = 10,
    parameter_optional_control_11 = 11,
    parameter_publication_low_12 = 12,
    parameter_publication_high_13 = 13,
};

struct alignas(8) KernelArguments {
    uint32_t words[14];
};
static_assert(sizeof(KernelArguments) == 56, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_feed_forward(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
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
    uint32_t reused_word_40 = 0;
    uint32_t block_index_41 = 0;
    uint32_t index_word_42 = 0;
    uint32_t block_index_43 = 0;
    uint32_t warp_lane_44 = 0;
    uint32_t packed_half_45 = 0;
    uint32_t matrix_accumulator_46 = 0;
    uint32_t matrix_accumulator_47 = 0;
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
    uint32_t matrix_accumulator_66 = 0;
    uint32_t matrix_accumulator_67 = 0;
    uint32_t matrix_accumulator_68 = 0;
    uint32_t matrix_accumulator_69 = 0;
    uint32_t matrix_accumulator_70 = 0;
    uint32_t matrix_accumulator_71 = 0;
    uint32_t matrix_accumulator_72 = 0;
    uint32_t matrix_accumulator_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
    uint32_t matrix_accumulator_82 = 0;
    uint32_t matrix_accumulator_83 = 0;
    uint32_t matrix_accumulator_84 = 0;
    uint32_t matrix_accumulator_85 = 0;
    uint32_t matrix_accumulator_86 = 0;
    uint32_t matrix_accumulator_87 = 0;
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
    uint32_t matrix_accumulator_98 = 0;
    uint32_t matrix_accumulator_99 = 0;
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t matrix_accumulator_103 = 0;
    uint32_t matrix_accumulator_104 = 0;
    uint32_t matrix_accumulator_105 = 0;
    uint32_t address_word_106 = 0;
    uint32_t address_word_107 = 0;
    uint32_t address_word_108 = 0;
    uint32_t address_word_109 = 0;
    uint32_t address_word_110 = 0;
    uint32_t address_word_111 = 0;
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
    uint32_t packed_half_130 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_address_word_12 = 0;
    uint32_t uniform_address_word_13 = 0;
    uint32_t uniform_reused_word_14 = 0;
    uint32_t uniform_reused_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_index_word_17 = 0;
    uint32_t uniform_index_word_79 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    matrix_fragment_49 = threadIdx.x;

    address_word_2 = parameters[parameter_trained_weights_low_4];
    address_word_3 = parameters[parameter_trained_weights_high_5];

    uniform_reused_word_14 = 0u;
    uniform_reused_word_15 = 0u;

    matrix_fragment_53 = threadIdx.y;

    uniform_address_word_13 = parameters[parameter_optional_control_11];

    block_index_43 = blockIdx.z;

    warp_lane_44 = parameters[parameter_optional_control_10];

    uniform_address_word_12 = parameters[parameter_tensor_width_7];

    matrix_accumulator_47 = threadIdx.x & 31u;

    uniform_index_word_11 = parameters[parameter_tensor_height_6];

    index_word_42 = matrix_fragment_49 | matrix_fragment_53;
    condition_0 = index_word_42 != 0;

    index_word_0 = matrix_fragment_53 * 0x40u;

    index_word_0 = index_word_0 & 0xc0u;

    reused_word_40 = block_index_43 * 0x100u + index_word_0;

    if (!condition_0) {
        matrix_fragment_4 = blockDim.x;

        matrix_fragment_7 = 0u;

        index_word_0 = 0x400u;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    uniform_condition_1 = uniform_condition_0;

    packed_half_45 = reused_word_40 * 0x8u;

    if (!condition_0) {
        index_word_0 = index_word_0 + 0x2000u;

        matrix_fragment_5 = blockDim.y;

        matrix_fragment_4 = matrix_fragment_4 * matrix_fragment_5;

        index_word_0 = (matrix_fragment_7 << 0x18u) + index_word_0;
    }

    matrix_fragment_5 = packed_half_45 + 0x80u;

    uniform_index_word_4 =
        broadcast_warp_word(matrix_fragment_4, !condition_0, uniform_index_word_4);

    uniform_address_word_6 =
        broadcast_warp_word(index_word_0, !condition_0, uniform_address_word_6);

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    split_address_words(address_word_2, address_word_3,
                        uint64_t(uint32_t(packed_half_45)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x200u);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x400u);

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    if (!uniform_condition_0) {
        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_2, address_word_3));

    split_address_words(matrix_fragment_38, matrix_fragment_39,
                        int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    split_address_words(matrix_accumulator_46, matrix_accumulator_47,
                        int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    phoenix_architecture::finish_compatibility_copies();

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_36, matrix_fragment_37),
                                        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30,
                                        matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x4000ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_46, matrix_accumulator_47),
                                        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22,
                                        matrix_fragment_23, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_46, matrix_accumulator_47) + 0x4000ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    index_word_0 = min(int32_t(0u), int32_t(warp_lane_44));

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_address_word_13));

    block_index_41 = blockIdx.x;

    index_word_0 = 0u - index_word_0 + uniform_address_word_12;

    uniform_address_word_6 = 0u - uniform_index_word_4 + uniform_index_word_11;

    uniform_condition_1 = uint32_t(uniform_address_word_12) != 0u;

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3),
                                        matrix_fragment_32, matrix_fragment_33, matrix_fragment_34,
                                        matrix_fragment_35, memory_ranges, error_counts);

    condition_1 = uint32_t(index_word_0) != 0u;

    uniform_condition_0 = uint32_t(uniform_address_word_6) != 0u;

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x4000ull,
                                        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18,
                                        matrix_fragment_19, memory_ranges, error_counts);

    matrix_accumulator_76 = matrix_fragment_53 & 0x3u;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_38, matrix_fragment_39),
                                        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26,
                                        matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_38, matrix_fragment_39) + 0x4000ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    address_word_2 = blockIdx.y;

    if (uniform_condition_1) {
        uniform_address_word_8 = uniform_address_word_12 + 0xffffffffu;
    }

    if (uniform_condition_0) {
        uniform_address_word_6 = uniform_address_word_6 + 0xffffffffu;
    }

    if (condition_1) {
        index_word_0 = index_word_0 + 0xffffffffu;
    }

    matrix_fragment_38 = matrix_accumulator_76 | matrix_fragment_49;
    condition_0 = matrix_fragment_38 != 0;

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x2u >> 32) + uniform_index_word_11;

    uniform_index_word_10 = uint32_t(int32_t(uniform_address_word_12) >> 0x1fu);

    if (uniform_condition_1) {
        uniform_address_word_9 = uint32_t(int32_t(uniform_address_word_8) >> 0x1fu);
    }

    if (uniform_condition_0) {
        uniform_index_word_7 = uint32_t(int32_t(uniform_address_word_6) >> 0x1fu);
    }

    if (condition_1) {
        address_word_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);
    }

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_address_word_12;

    if (uniform_condition_1) {
        uniform_address_word_9 = uint32_t(combine_words(uniform_address_word_9, 0u) << 0x3u >> 32) +
                                 uniform_address_word_8;
    }

    if (uniform_condition_0) {
        uniform_index_word_7 = uint32_t(combine_words(uniform_index_word_7, 0u) << 0x3u >> 32) +
                               uniform_address_word_6;
    }

    if (condition_1) {
        address_word_3 = uint32_t(combine_words(address_word_3, 0u) << 0x3u >> 32) + index_word_0;
    }

    uniform_index_word_4 = 0u;

    uniform_index_word_5 = 0u;

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_10) >> 0x2u);

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_address_word_9)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_60 = 0u;

    if (condition_1) {
        matrix_accumulator_60 =
            uint32_t(int64_t(int32_t(address_word_3)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    address_word_3 = block_index_41 * 0x2u;
    if (!condition_0) {
        index_word_0 = address_word_2 * 0x8u + matrix_fragment_53;

        index_word_0 = index_word_0 + (0u - uniform_address_word_13);

        matrix_fragment_37 = index_word_0 + 0x4u;

        matrix_fragment_36 = index_word_0 + 0x3u;

        index_word_0 = max(int32_t(0u), int32_t(index_word_0));

        condition_0 = int32_t(matrix_fragment_37) > int32_t(0u);

        matrix_fragment_36 = uint32_t(combine_words(matrix_fragment_36, 0u) << 0x1du >> 32) + 0x1u;

        matrix_fragment_37 = block_index_41 * 0x8u + (0u - warp_lane_44);

        index_word_0 = uint32_t(uint32_t(index_word_0) >> 0x3u);

        matrix_fragment_36 = condition_0 ? matrix_fragment_36 : 0u;

        matrix_fragment_39 = matrix_fragment_37 + 0x8u;

        matrix_fragment_49 = min(int32_t(matrix_fragment_36), int32_t(uniform_index_word_4));

        matrix_fragment_36 = matrix_fragment_37 + 0x7u;

        condition_0 = int32_t(matrix_fragment_39) > int32_t(0u);

        matrix_fragment_37 = max(int32_t(0u), int32_t(matrix_fragment_37));

        condition_1 = int32_t(index_word_0) >= int32_t(matrix_fragment_49);

        matrix_fragment_36 = uint32_t(combine_words(matrix_fragment_36, 0u) << 0x1du >> 32) + 0x1u;

        matrix_accumulator_47 = condition_0 ? matrix_fragment_36 : 0u;
        if (!condition_1) {
            matrix_fragment_39 = uint32_t(uint32_t(matrix_fragment_37) >> 0x3u);

            matrix_fragment_51 = matrix_accumulator_60 * 0x2u;

            matrix_accumulator_46 =
                min(int32_t(matrix_accumulator_47), int32_t(matrix_accumulator_60));
            do {
                condition_0 = int32_t(matrix_fragment_39) >= int32_t(matrix_accumulator_46);
                if (!condition_0) {
                    matrix_fragment_55 = matrix_fragment_51 * index_word_0;

                    warp_lane_44 = matrix_fragment_39;
                    do {
                        matrix_fragment_36 = parameters[parameter_wait_optional_low_8];
                        matrix_fragment_37 = parameters[parameter_wait_optional_high_9];

                        matrix_accumulator_47 = matrix_fragment_55 + warp_lane_44;

                        split_address_words(
                            matrix_fragment_36, matrix_fragment_37,
                            int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_36, matrix_fragment_37));

                        matrix_accumulator_47 = pipeline_read<CheckBounds>(
                            combine_words(matrix_fragment_36, matrix_fragment_37), memory_ranges,
                            error_counts);

                        warp_lane_44 = warp_lane_44 + 0x1u;

                        condition_1 = uint32_t(warp_lane_44) != uint32_t(matrix_accumulator_46);

                        condition_0 = int32_t(matrix_accumulator_47) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                matrix_accumulator_47 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_36, matrix_fragment_37),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(matrix_accumulator_47) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                index_word_0 = index_word_0 + 0x1u;

                condition_0 = uint32_t(index_word_0) != uint32_t(matrix_fragment_49);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    matrix_accumulator_77 = parameters[parameter_tensor_height_6];

    matrix_accumulator_61 = uint32_t(uint32_t(matrix_fragment_53) >> 0x2u);

    index_word_0 = uint32_t(uint32_t(matrix_fragment_53) >> 0x1u);

    matrix_fragment_36 = matrix_fragment_53 * 0x80u;

    phoenix_architecture::finish_compatibility_copies();

    matrix_accumulator_101 = address_word_2 * 0x2u + matrix_accumulator_61;

    matrix_accumulator_86 = index_word_0 & 0x1u;

    condition_1 = false;

    index_word_0 = matrix_accumulator_101 * uniform_index_word_10;

    condition_0 = int32_t(matrix_accumulator_101) >= int32_t(uniform_index_word_11);

    matrix_accumulator_86 = address_word_3 + matrix_accumulator_86;

    uniform_index_word_17 = 0u;

    matrix_accumulator_87 = matrix_fragment_53 & 0x1u;

    matrix_fragment_37 = matrix_accumulator_86;

    matrix_accumulator_77 = matrix_accumulator_77 & 0xfffffffcu;

    condition_3 = uint32_t(matrix_accumulator_77) != uint32_t(0x4u);

    condition_2 = uint32_t(matrix_accumulator_77) != uint32_t(0x4u);

    matrix_fragment_88 = condition_2 ? index_word_0 : 0u;
    if (condition_0) {
        if (!condition_3) {
            matrix_fragment_37 = parameters[parameter_tensor_width_7];

            matrix_fragment_37 = matrix_fragment_37 & 0xfffffffcu;

            condition_0 = uint32_t(matrix_fragment_37) != uint32_t(0x4u);

            matrix_fragment_37 = 0u;

            condition_1 =
                int32_t(matrix_accumulator_86) < int32_t(uniform_index_word_10) || !condition_0;

            if (condition_0) {
                matrix_fragment_37 = matrix_accumulator_86;
            }
        }
    } else {
        matrix_fragment_37 = parameters[parameter_tensor_width_7];

        matrix_fragment_37 = matrix_fragment_37 & 0xfffffffcu;

        condition_0 = uint32_t(matrix_fragment_37) != uint32_t(0x4u);

        matrix_fragment_37 = 0u;

        condition_1 =
            int32_t(matrix_accumulator_86) < int32_t(uniform_index_word_10) || !condition_0;

        if (condition_0) {
            matrix_fragment_37 = matrix_accumulator_86;
        }
    }
    warp_lane_44 = matrix_fragment_36 & 0x100u;

    matrix_fragment_39 = matrix_accumulator_87 * 0x80u;

    matrix_accumulator_46 = matrix_fragment_88 + matrix_fragment_37;

    uniform_index_word_16 = 0x400u;

    warp_lane_44 = matrix_accumulator_61 * 0x200u + warp_lane_44;

    uniform_address_word_6 = (uniform_index_word_17 << 0x18u) + uniform_index_word_16;

    matrix_fragment_36 = matrix_accumulator_46 * 0x800u + matrix_fragment_39;

    warp_lane_44 = warp_lane_44 + matrix_fragment_39;

    matrix_fragment_37 = uint32_t(int32_t(matrix_fragment_36) >> 0x1fu);

    matrix_fragment_89 = (warp_lane_44 << 0x2u) + uniform_address_word_6;

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        condition_1 ? combine_words(matrix_fragment_36, matrix_fragment_37)
                                    : combine_words(0u, 0u));
    if (!condition_1) {
        matrix_fragment_36 = threadIdx.x & 31u;

        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_fragment_48 = uniform_address_word_6;

        matrix_fragment_37 = uniform_address_word_6;

        matrix_fragment_48 = permute_signed(matrix_fragment_48, matrix_fragment_37, 0x5410u);

        matrix_fragment_49 = matrix_fragment_48;

        matrix_fragment_50 = matrix_fragment_48;

        matrix_fragment_51 = matrix_fragment_48;

        matrix_fragment_36 = matrix_fragment_36 * 0x10u + matrix_fragment_89;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_36, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_6 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            matrix_accumulator_46 = parameters[parameter_input_low_0];
            matrix_accumulator_47 = parameters[parameter_input_high_1];

            uniform_address_word_6 = uniform_index_word_16 + 0x2000u;

            uniform_address_word_12 = broadcast_active_lane(matrix_fragment_89);

            uniform_address_word_13 = (uniform_index_word_17 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = 0x20u;

            const uint64_t byte_address_0 =
                scaled_word_address(combine_words(matrix_accumulator_46, matrix_accumulator_47),
                                    combine_words(matrix_fragment_36, matrix_fragment_37));
            matrix_fragment_39 = uint32_t(byte_address_0);

            matrix_fragment_36 = uint32_t(byte_address_0 >> 32);

            uniform_address_word_8 = broadcast_active_lane(matrix_fragment_39);

            uniform_address_word_9 = broadcast_active_lane(matrix_fragment_36);

            matrix_fragment_36 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_12,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_6 * 16u, uniform_address_word_13, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_12,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_fragment_39 = 0u;

    uniform_address_word_6 = uniform_index_word_16 + 0x2000u;

    matrix_fragment_48 = parameters[parameter_optional_control_10];
    matrix_fragment_49 = parameters[parameter_optional_control_11];

    warp_lane_44 = address_word_2 * 0x8u + matrix_fragment_53;

    matrix_fragment_36 = uniform_address_word_6;

    warp_lane_44 = warp_lane_44 + (0u - matrix_fragment_49);

    matrix_fragment_48 = block_index_41 * 0x8u + (0u - matrix_fragment_48);

    matrix_accumulator_46 = warp_lane_44 + 0x4u;

    matrix_fragment_39 = (matrix_fragment_39 << 0x18u) + matrix_fragment_36;

    matrix_accumulator_47 = warp_lane_44 + 0x3u;

    matrix_fragment_90 = max(int32_t(0u), int32_t(warp_lane_44));

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_fragment_39;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_fragment_36, matrix_fragment_37,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_fragment_36 = 0u;
#endif

    condition_0 = (int32_t(matrix_accumulator_46) > int32_t(0u));

    warp_lane_44 = matrix_fragment_48 + 0x8u;

    matrix_accumulator_47 =
        uint32_t(combine_words(matrix_accumulator_47, 0u) << 0x1du >> 32) + 0x1u;

    matrix_fragment_90 = uint32_t(uint32_t(matrix_fragment_90) >> 0x3u);

    condition_2 = int32_t(warp_lane_44) > int32_t(0u);

    matrix_accumulator_47 = condition_0 ? matrix_accumulator_47 : 0u;

    matrix_fragment_91 = min(int32_t(matrix_accumulator_47), int32_t(uniform_index_word_4));

    condition_0 = int32_t(matrix_fragment_90) >= int32_t(matrix_fragment_91);

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 = wait_copy_barrier(shared_tile, matrix_fragment_39,
                                    async_tokens[(((matrix_fragment_39) >> 3) & 1)]);
#else
    phoenix_architecture::finish_compatibility_copies();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_fragment_39,
                                            async_tokens[(((matrix_fragment_39) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    matrix_accumulator_92 = parameters[parameter_tensor_width_7];

    matrix_fragment_36 = matrix_fragment_48 + 0x7u;

    matrix_accumulator_94 = max(int32_t(0u), int32_t(matrix_fragment_48));

    condition_0 = uint32_t(matrix_fragment_38) != 0u || condition_0;

    matrix_accumulator_104 = 0x1u;

    matrix_accumulator_95 = 0u;

    matrix_fragment_36 = uint32_t(combine_words(matrix_fragment_36, 0u) << 0x1du >> 32) + 0x1u;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_93 = condition_2 ? matrix_fragment_36 : 0u;

    matrix_accumulator_64 = 0;
    matrix_accumulator_65 = 0;

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    matrix_accumulator_66 = 0;
    matrix_accumulator_67 = 0;

    matrix_accumulator_68 = 0;
    matrix_accumulator_69 = 0;

    matrix_accumulator_70 = 0;
    matrix_accumulator_71 = 0;

    matrix_accumulator_72 = 0;
    matrix_accumulator_73 = 0;

    matrix_accumulator_74 = 0;
    matrix_accumulator_75 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    matrix_accumulator_92 = matrix_accumulator_92 & 0xfffffffcu;

    matrix_accumulator_94 = uint32_t(uint32_t(matrix_accumulator_94) >> 0x3u);

    matrix_accumulator_93 = min(int32_t(matrix_accumulator_93), int32_t(matrix_accumulator_60));
    do {
        matrix_fragment_36 = matrix_accumulator_95 + 0x80u;

        matrix_fragment_37 = matrix_accumulator_104 * 0x100u;

        condition_1 = uint32_t(matrix_fragment_37) >= uint32_t(matrix_fragment_36);

        matrix_accumulator_96 = matrix_accumulator_95 + 0x40u;

        condition_1 = uint32_t(matrix_accumulator_104) > uint32_t(0x1u) || condition_1;
        if (!condition_1) {
            if (!condition_0) {
                matrix_fragment_38 = matrix_fragment_90;
                do {
                    condition_1 = int32_t(matrix_accumulator_94) >= int32_t(matrix_accumulator_93);
                    if (!condition_1) {
                        matrix_fragment_37 = matrix_fragment_38 * matrix_accumulator_60;

                        warp_lane_44 = matrix_accumulator_94;

                        matrix_fragment_49 = matrix_fragment_37 * 0x2u + matrix_accumulator_60;
                        do {
                            matrix_fragment_36 = parameters[parameter_wait_optional_low_8];
                            matrix_fragment_37 = parameters[parameter_wait_optional_high_9];

                            matrix_fragment_39 = matrix_fragment_49 + warp_lane_44;

                            split_address_words(
                                matrix_fragment_36, matrix_fragment_37,
                                int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_36, matrix_fragment_37));

                            matrix_fragment_39 = pipeline_read<CheckBounds>(
                                combine_words(matrix_fragment_36, matrix_fragment_37),
                                memory_ranges, error_counts);

                            warp_lane_44 = warp_lane_44 + 0x1u;

                            condition_2 = uint32_t(warp_lane_44) != uint32_t(matrix_accumulator_93);

                            condition_1 = int32_t(matrix_fragment_39) > -1;
                            if (!condition_1) {
                                do {
                                    __nanosleep(0x40u);

                                    matrix_fragment_39 = pipeline_read<CheckBounds>(
                                        combine_words(matrix_fragment_36, matrix_fragment_37),
                                        memory_ranges, error_counts);

                                    condition_1 = int32_t(matrix_fragment_39) > -1;
                                } while (!condition_1);
                            }
                        } while (condition_2);
                    }
                    matrix_fragment_38 = matrix_fragment_38 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_38) != uint32_t(matrix_fragment_91);
                } while (condition_1);
            }
            __syncwarp(__activemask());

            phoenix_architecture::finish_compatibility_copies();

            matrix_accumulator_104 = 0x2u;
        }
        matrix_fragment_50 = 0u;

        condition_1 = int32_t(matrix_accumulator_101) >= int32_t(uniform_index_word_11);

        condition_2 = int32_t(matrix_accumulator_101) >= int32_t(uniform_index_word_11);

        matrix_fragment_36 = uint32_t(uint32_t(matrix_accumulator_95) >> 0x6u);

        matrix_fragment_49 = 0x400u;

        condition_4 = uint32_t(matrix_accumulator_77) != uint32_t(0x4u) && condition_1;

        condition_2 = uint32_t(matrix_accumulator_77) == uint32_t(0x4u) || !condition_2;

        condition_5 = (matrix_fragment_36 & 0x1u) != 0;

        matrix_fragment_36 = matrix_accumulator_95 * 0x40u;

        matrix_accumulator_97 = matrix_fragment_49 + 0x2000u;

        condition_3 = uint32_t(matrix_accumulator_92) == uint32_t(0x4u) || condition_4;

        condition_6 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);

        matrix_fragment_37 = condition_4 ? matrix_accumulator_86 : 0u;

        matrix_fragment_48 = matrix_fragment_36 & 0x1000u;

        condition_3 =
            int32_t(matrix_accumulator_86) < int32_t(uniform_index_word_10) || condition_3;

        matrix_fragment_38 = 0x1000u ^ matrix_fragment_48;

        matrix_fragment_37 = !condition_6 ? matrix_fragment_37 : matrix_accumulator_86;

        condition_2 = condition_3 & condition_2;

        matrix_accumulator_97 = (matrix_fragment_50 << 0x18u) + matrix_accumulator_97;

        if (!condition_5) {
            matrix_accumulator_97 = matrix_accumulator_97 + 0x8u;
        }
        if (!condition_2) {
            warp_lane_44 = threadIdx.x & 31u;

            uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_fragment_37 = matrix_fragment_89 + matrix_fragment_38;

            matrix_fragment_36 = uniform_address_word_6;

            matrix_fragment_39 = uniform_address_word_6;

            matrix_fragment_36 = permute_signed(matrix_fragment_36, matrix_fragment_39, 0x5410u);

            matrix_fragment_38 = matrix_fragment_36;

            matrix_fragment_39 = matrix_fragment_36;

            warp_lane_44 = warp_lane_44 * 0x10u + matrix_fragment_37;

            matrix_fragment_37 = matrix_fragment_36;

            store_four_shared_words<CheckBounds>(
                shared_tile, warp_lane_44, matrix_fragment_36, matrix_fragment_37,
                matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);
        } else {
            if (condition_2) {
                matrix_fragment_37 = matrix_fragment_88 + matrix_fragment_37;

                matrix_fragment_36 =
                    uint32_t(combine_words(matrix_accumulator_96, 0u) << 0x1bu >> 32) +
                    matrix_accumulator_87;
            }

            uniform_address_word_6 = 0xffffffffu;

            if (condition_2) {
                matrix_fragment_36 = matrix_fragment_37 * 0x10u + matrix_fragment_36;

                matrix_fragment_39 = matrix_fragment_36 * 0x80u;
            }

            split_address_words(matrix_fragment_36, matrix_fragment_37, 0ull);

            if (condition_2) {
                matrix_fragment_37 = uint32_t(int32_t(matrix_fragment_39) >> 0x1fu);

                matrix_fragment_36 = matrix_fragment_39;
            }

            condition_2 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_2) {
                matrix_accumulator_98 = parameters[parameter_input_low_0];
                matrix_accumulator_99 = parameters[parameter_input_high_1];

                uniform_address_word_8 = broadcast_active_lane(matrix_fragment_38);

                uniform_address_word_6 = broadcast_active_lane(matrix_fragment_89);

                uniform_index_word_7 = broadcast_active_lane(matrix_accumulator_97);

                uniform_address_word_6 = uniform_address_word_6 + uniform_address_word_8;

                uniform_address_word_8 = 0x20u;

                // Later control flow also consumes this address-overflow predicate.
                condition_2 =
                    (uint64_t(matrix_fragment_36) << 2) + matrix_accumulator_98 > 0xffffffffull;
                const uint64_t byte_address_1 =
                    scaled_word_address(combine_words(matrix_accumulator_98, matrix_accumulator_99),
                                        combine_words(matrix_fragment_36, matrix_fragment_37));
                matrix_fragment_38 = uint32_t(byte_address_1);

                matrix_fragment_36 = uint32_t(byte_address_1 >> 32);

                uniform_address_word_12 = broadcast_active_lane(matrix_fragment_38);

                uniform_address_word_13 = broadcast_active_lane(matrix_fragment_36);

                matrix_fragment_36 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_12, uniform_address_word_13),
                    uniform_address_word_8 * 16u, matrix_accumulator_97, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_12, uniform_address_word_13),
                    uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        warp_lane_44 = threadIdx.x & 31u;

        matrix_accumulator_105 = (matrix_fragment_50 << 0x18u) + matrix_fragment_49;

        __syncwarp(__activemask());

        matrix_fragment_36 = matrix_accumulator_105 + matrix_fragment_48;

        matrix_fragment_37 = matrix_accumulator_61 * 0x800u + matrix_fragment_36;

        matrix_accumulator_100 = warp_lane_44 * 0x10u + matrix_fragment_37;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_100 + 0x400u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_100, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_80,
                                 matrix_accumulator_81);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_100 + 0x600u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_28, matrix_fragment_29, matrix_fragment_52,
                                 matrix_fragment_53);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_100 + 0x200u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_68,
                                 matrix_accumulator_69);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_64,
                                 matrix_accumulator_65);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_58,
                                 matrix_accumulator_59);

        matrix_fragment_36 = parameters[parameter_trained_weights_low_4];
        matrix_fragment_37 = parameters[parameter_trained_weights_high_5];

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_74,
                                 matrix_accumulator_75);

        multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_72,
                                 matrix_accumulator_73);

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_70,
                                 matrix_accumulator_71);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_66,
                                 matrix_accumulator_67);

        multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_68,
                                 matrix_accumulator_69);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_56,
                                 matrix_accumulator_57);

        matrix_fragment_5 = matrix_accumulator_96 * 0x80u + packed_half_45;

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_36, matrix_fragment_37));

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_82,
                                 matrix_accumulator_83);

        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x200ull, matrix_fragment_28,
            matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x400ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x600ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_74,
                                 matrix_accumulator_75);

        multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_72,
                                 matrix_accumulator_73);

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_70,
                                 matrix_accumulator_71);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_66,
                                 matrix_accumulator_67);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_98,
                                 matrix_accumulator_99);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x4000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_64,
                                 matrix_accumulator_65);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x4200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_46,
                                 matrix_accumulator_47);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x4400ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_58,
                                 matrix_accumulator_59);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_38, matrix_fragment_39),
                                            matrix_fragment_32, matrix_fragment_33,
                                            matrix_fragment_34, matrix_fragment_35, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x4600ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = matrix_accumulator_97;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_48, matrix_fragment_49,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_48 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_2 = wait_copy_barrier(shared_tile, matrix_accumulator_97,
                                        async_tokens[(((matrix_accumulator_97) >> 3) & 1)]);
#else
        phoenix_architecture::finish_compatibility_copies();
        condition_2 = true;
#endif
        if (!condition_2) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_2 = wait_copy_barrier(shared_tile, matrix_accumulator_97,
                                                async_tokens[(((matrix_accumulator_97) >> 3) & 1)]);
#else
                phoenix_architecture::finish_compatibility_copies();
                condition_2 = true;
#endif
            } while (!condition_2);
        }
        condition_2 = uint32_t(matrix_accumulator_95) >= uint32_t(0x180u);

        matrix_accumulator_95 = matrix_accumulator_96;
    } while (!condition_2);
    matrix_accumulator_61 = matrix_accumulator_61 * 0x800u + matrix_accumulator_105;

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_76 = block_index_43 * 0x4u + matrix_accumulator_76;

    matrix_accumulator_92 = 0;
    matrix_accumulator_93 = 0;

    matrix_fragment_38 = warp_lane_44 * 0x10u + matrix_accumulator_61;

    matrix_accumulator_94 = 0;
    matrix_accumulator_95 = 0;

    address_word_107 = matrix_accumulator_76 * 0x20u;

    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_38 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
        matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

    split_address_words(address_word_106, address_word_107,
                        uint64_t(uint32_t(address_word_107)) * uint64_t(uint32_t(0x200u)) +
                            combine_words(matrix_fragment_36, matrix_fragment_37));

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_38 + 0x1400u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    split_address_words(address_word_106, address_word_107,
                        combine_words(address_word_106, address_word_107) + 0x42200u);

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    uniform_index_word_4 = 0u;

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_70,
                             matrix_accumulator_71);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                             matrix_fragment_55);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_38 + 0x1200u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_64,
                             matrix_accumulator_65);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_38 + 0x1600u, matrix_fragment_28, matrix_fragment_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_fragment_5 = matrix_accumulator_76 * 0x1000u;

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_36, matrix_fragment_37));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_72,
                             matrix_accumulator_73);

    split_address_words(address_word_108, address_word_109,
                        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x60600u);

    split_address_words(address_word_110, address_word_111,
                        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x40200u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_70,
                             matrix_accumulator_71);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_22,
                             matrix_fragment_23);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_accumulator_86 = 0;
    matrix_accumulator_87 = 0;

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;
    do {
        split_address_words(matrix_fragment_36, matrix_fragment_37,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_110, address_word_111));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_36, matrix_fragment_37) - 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_36, matrix_fragment_37),
                                            matrix_fragment_28, matrix_fragment_29,
                                            matrix_fragment_30, matrix_fragment_31, memory_ranges,
                                            error_counts);

        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_106, address_word_107));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) - 0x200ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_38, matrix_fragment_39),
                                            matrix_fragment_24, matrix_fragment_25,
                                            matrix_fragment_26, matrix_fragment_27, memory_ranges,
                                            error_counts);

        split_address_words(matrix_accumulator_116, matrix_accumulator_117,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_108, address_word_109));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_116, matrix_accumulator_117) - 0x600ull,
            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_116, matrix_accumulator_117) - 0x400ull,
            matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_116, matrix_accumulator_117) - 0x200ull,
            matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_116, matrix_accumulator_117), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        packed_half_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(0u));

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        matrix_accumulator_118 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                          select_half_operand<2, false, false>(0u));

        uniform_address_word_6 = uniform_index_word_4 + 0x20u;

        matrix_accumulator_119 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                          select_half_operand<2, false, false>(0u));

        split_address_words(address_word_106, address_word_107,
                            combine_words(address_word_106, address_word_107) + 0x400u);

        split_address_words(address_word_108, address_word_109,
                            combine_words(address_word_108, address_word_109) + 0x800u);

        split_address_words(address_word_110, address_word_111,
                            combine_words(address_word_110, address_word_111) + 0x400u);

        uniform_index_word_4 = uniform_address_word_6;

        matrix_accumulator_120 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(packed_half_45));

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(matrix_accumulator_118));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                          select_half_operand<2, false, false>(matrix_accumulator_119));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                          select_half_operand<2, false, false>(matrix_accumulator_120));

        matrix_accumulator_120 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                          select_half_operand<2, false, false>(0u));

        packed_half_130 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                                        select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_28, matrix_fragment_29, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_30, matrix_fragment_31, 0u, 0u);

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                          select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                          select_half_operand<2, false, false>(matrix_accumulator_120));

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        matrix_fragment_35 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                          select_half_operand<2, false, false>(matrix_fragment_35));

        matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                           select_half_operand<2, false, false>(packed_half_130));

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_28, matrix_fragment_29, 0u, 0u);

        matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                           select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        matrix_fragment_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_33 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                          select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_30, matrix_fragment_31, 0u, 0u);

        matrix_fragment_28 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_30 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_31 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                          select_half_operand<2, false, false>(matrix_fragment_30));

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_30 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_31 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_38 = 0x2b28u;

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_accumulator_127 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_accumulator_128 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0xc400c400u);

        matrix_fragment_33 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_35 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_fragment_32 = permute_signed(matrix_fragment_38, matrix_fragment_32, 0x7610u);

        matrix_fragment_34 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_120,
                                 matrix_accumulator_121);

        matrix_fragment_39 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

        matrix_accumulator_129 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_128),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_35 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        packed_half_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_39),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_34), 0x3b283b28u);

        matrix_fragment_33 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_35),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_24 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

        packed_half_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39),
            select_half_operand<2, false, false>(packed_half_45), 0x3b283b28u);

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35),
            select_half_operand<2, false, false>(matrix_fragment_38), 0x3b283b28u);

        matrix_fragment_35 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_36,
                                 matrix_fragment_37);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

        matrix_accumulator_129 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_128),
            select_half_operand<2, false, false>(matrix_accumulator_129), 0x3b283b28u);

        matrix_fragment_35 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_accumulator_126 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_35),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_accumulator_127 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_126),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_accumulator_125 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_20 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126),
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x3b283b28u);

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_21 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_20 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

        matrix_fragment_25 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_22 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                select_half_operand<2, false, false>(matrix_accumulator_129));

        matrix_fragment_23 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

        matrix_fragment_21 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_20),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_22 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_31 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_23),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_31 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

        matrix_fragment_21 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_20),
            select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_20 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_22),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_23),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_27 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_20 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_22),
            select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_27 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

        matrix_fragment_22 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_21));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        packed_half_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(packed_half_45));

        matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_31),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        packed_half_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_45),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_22 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                          select_half_operand<2, false, false>(matrix_fragment_35));

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_27),
            select_half_operand<0, false, true>(matrix_fragment_32), 0x37283728u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_27),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_23 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_20));

        matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_26 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                          select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_20 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(packed_half_45));

        matrix_fragment_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_104,
                                 matrix_accumulator_105);
    } while (!uniform_condition_0);
    matrix_fragment_9 = threadIdx.z;

    matrix_fragment_4 = parameters[parameter_output_low_2];
    matrix_fragment_5 = parameters[parameter_output_high_3];

    condition_0 = int32_t(address_word_3) >= int32_t(uniform_index_word_10) || condition_1;

    matrix_fragment_7 = address_word_3 + index_word_0;

    index_word_0 = address_word_3 + 0x1u;

    address_word_3 = matrix_fragment_7 * 0x200u + reused_word_40;

    condition_1 = int32_t(index_word_0) >= int32_t(uniform_index_word_10) || condition_1;

    address_word_3 = address_word_3 * 0x4u;

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    condition_2 = (index_word_42 | matrix_fragment_9) != 0;
    if (!condition_0) {
        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(0u));

        index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                     select_half_operand<2, false, false>(0u));

        address_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(index_word_0));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(address_word_3));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    matrix_accumulator_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    address_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                   select_half_operand<2, false, false>(0u));

    index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                 select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        split_address_words(warp_lane_44, packed_half_45,
                            int64_t(int32_t(warp_lane_44)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_8, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_9, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_accumulator_113, index_word_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_accumulator_112, address_word_3, 0x5410u);

        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_14, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_fragment_10, matrix_fragment_15, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(warp_lane_44, packed_half_45) + 0x2000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        matrix_fragment_9 = permute_signed(matrix_accumulator_85, matrix_fragment_12, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_84, matrix_fragment_13, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(warp_lane_44, packed_half_45) + 0x2200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    phoenix_architecture::finish_compatibility_copies();

    if (condition_2) {
        return;
    }

    matrix_fragment_4 = parameters[parameter_publication_low_12];
    matrix_fragment_5 = parameters[parameter_publication_high_13];

    address_word_2 = address_word_2 * 0x2u + block_index_43;

    address_word_3 = address_word_2 * uniform_index_word_5 + block_index_41;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__ __maxnreg__(166) void phoenix_global_attention_split_heads_512_feed_forward(
    KernelArguments parameters) {
    run_global_attention_split_heads_512_feed_forward<false>(parameters.words, nullptr, nullptr);
}
