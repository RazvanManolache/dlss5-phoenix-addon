// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/feed_forward_projection_input_view.cu
// 512-channel split-head stage. Separate projection/feed-forward modules use the same persistent feature layout.
// The filename identifies which stage is executed; wait/view variants preserve producer-consumer synchronization.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 13336
#include "../../../common/async_memory.cuh"
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y, uint32_t& word_z,
                       uint32_t& word_w, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4* word_pointer = reinterpret_cast<const uint4*>(address);
    uint4 result_word = __ldca(word_pointer);
    word_x = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    word_w = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_global_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                              uint32_t& word_z, uint32_t& word_w, const MemoryRange* memory_ranges,
                              uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4* word_pointer = reinterpret_cast<const uint4*>(address);
    uint4 result_word = __ldcg(word_pointer);
    word_x = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    word_w = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t word_x, uint32_t word_y, uint32_t word_z,
                           uint32_t word_w, const MemoryRange* memory_ranges,
                           uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    uint4* word_pointer = reinterpret_cast<uint4*>(address);
    __stcs(word_pointer, make_uint4(word_x, word_y, word_z, word_w));
}

__device__ __forceinline__ uint32_t broadcast_active_lane(uint32_t value) {
    const unsigned mask = __activemask();
    return __shfl_sync(mask, value, __ffs(mask) - 1);
}
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_input_skip_low_2 = 2,
    parameter_input_skip_high_3 = 3,
    parameter_output_low_4 = 4,
    parameter_output_high_5 = 5,
    parameter_trained_weights_low_6 = 6,
    parameter_trained_weights_high_7 = 7,
    parameter_tensor_height_8 = 8,
    parameter_tensor_width_9 = 9,
    parameter_publication_optional_low_12 = 12,
    parameter_publication_optional_high_13 = 13,
    parameter_wait_low_14 = 14,
    parameter_wait_high_15 = 15,
    parameter_optional_control_16 = 16,
    parameter_optional_control_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[18];
};
static_assert(sizeof(KernelArguments) == 72, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void
run_global_attention_split_heads_512_feed_forward_projection_input_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_accumulator_4 = 0;
    uint32_t matrix_accumulator_5 = 0;
    uint32_t matrix_accumulator_6 = 0;
    uint32_t matrix_accumulator_7 = 0;
    uint32_t matrix_accumulator_8 = 0;
    uint32_t matrix_accumulator_9 = 0;
    uint32_t matrix_accumulator_10 = 0;
    uint32_t matrix_accumulator_11 = 0;
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t index_word_14 = 0;
    uint32_t matrix_accumulator_15 = 0;
    uint32_t matrix_accumulator_16 = 0;
    uint32_t matrix_accumulator_17 = 0;
    uint32_t matrix_accumulator_18 = 0;
    uint32_t matrix_accumulator_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_accumulator_24 = 0;
    uint32_t matrix_accumulator_25 = 0;
    uint32_t matrix_accumulator_26 = 0;
    uint32_t matrix_accumulator_27 = 0;
    uint32_t address_word_28 = 0;
    uint32_t address_word_29 = 0;
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
    uint32_t matrix_accumulator_106 = 0;
    uint32_t matrix_accumulator_107 = 0;
    uint32_t matrix_accumulator_108 = 0;
    uint32_t matrix_accumulator_109 = 0;
    uint32_t index_word_110 = 0;
    uint32_t index_word_111 = 0;
    uint32_t reused_word_112 = 0;
    uint32_t index_word_113 = 0;
    uint32_t index_word_114 = 0;
    uint32_t index_word_115 = 0;
    uint32_t warp_lane_116 = 0;
    uint32_t warp_lane_117 = 0;
    uint32_t index_word_118 = 0;
    uint32_t tile_word_119 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_address_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_address_word_12 = 0;
    uint32_t uniform_block_index_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_index_word_15 = 0;
    uint32_t uniform_parameter_word_16 = 0;
    uint32_t uniform_index_word_17 = 0;
    uint32_t uniform_reused_word_18 = 0;
    uint32_t uniform_reused_word_19 = 0;
    uint32_t uniform_address_word_20 = 0;
    uint32_t uniform_address_word_21 = 0;
    uint32_t uniform_address_word_22 = 0;
    uint32_t uniform_address_word_23 = 0;
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
    bool uniform_condition_2 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    uniform_index_word_14 = parameters[parameter_tensor_width_9];

    matrix_accumulator_38 = threadIdx.y;

    uniform_block_index_13 = blockIdx.x;

    uniform_index_word_17 = parameters[parameter_optional_control_16];

    matrix_accumulator_5 = blockIdx.z;

    uniform_address_word_20 = parameters[parameter_optional_control_17];

    matrix_accumulator_9 = threadIdx.x & 31u;

    uniform_parameter_word_16 = parameters[parameter_tensor_height_8];

    uniform_reused_word_18 = 0u;
    uniform_reused_word_19 = 0u;

    uniform_index_word_4 = uniform_index_word_14 + 0xffffffffu;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x3u >> 32) + uniform_index_word_4;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_5) >> 0x3u);

    uniform_address_word_10 = uniform_index_word_7 + 0x1u;

    index_word_0 = uniform_address_word_10;

    matrix_accumulator_2 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    matrix_accumulator_4 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    uniform_index_word_11 = broadcast_active_lane(matrix_accumulator_2);

    matrix_accumulator_2 = blockIdx.x;

    matrix_accumulator_4 = 0u - matrix_accumulator_4;

    uniform_index_word_4 = __float_as_uint(__int2float_ru(uniform_index_word_11));

    matrix_accumulator_3 = reciprocal_bits(uniform_index_word_4);

    matrix_accumulator_2 =
        int32_t(matrix_accumulator_2) < 0 ? 0u - matrix_accumulator_2 : matrix_accumulator_2;

    index_word_0 = matrix_accumulator_2;

    uniform_index_word_4 = broadcast_active_lane(matrix_accumulator_3);

    uniform_index_word_8 = broadcast_active_lane(index_word_0);

    index_word_0 = matrix_accumulator_4;

    uniform_index_word_9 = broadcast_active_lane(index_word_0);

    uniform_index_word_4 = uniform_index_word_4 + 0xffffffeu;

    uniform_index_word_5 = __float2uint_rz(__uint_as_float(uniform_index_word_4));

    uniform_index_word_4 = 0u;

    matrix_accumulator_3 = 0u - uniform_index_word_5;

    uniform_address_word_6 = broadcast_active_lane(matrix_accumulator_3);

    matrix_accumulator_3 = threadIdx.x;

    uniform_address_word_6 = uniform_address_word_6 * uniform_index_word_11;

    uniform_index_word_4 = uint32_t(uint64_t(uniform_index_word_5) * uniform_address_word_6 +
                                        combine_words(uniform_index_word_4, uniform_index_word_5) >>
                                    32);

    split_address_words(uniform_index_word_4, uniform_index_word_5,
                        uint64_t(uint32_t(uniform_index_word_4)) *
                            uint64_t(uint32_t(uniform_index_word_8)));

    uniform_index_word_4 = uniform_index_word_5 * uniform_index_word_9 + uniform_index_word_8;

    uniform_condition_0 = uint32_t(uniform_index_word_11) > uint32_t(uniform_index_word_4);

    index_word_0 = matrix_accumulator_3 | matrix_accumulator_38;
    condition_0 = index_word_0 != 0;

    if (!uniform_condition_0) {
        uniform_index_word_4 = uniform_index_word_4 + (0u - uniform_index_word_11);

        uniform_index_word_5 = uniform_index_word_5 + 0x1u;
    }

    uniform_condition_1 = uint32_t(uniform_index_word_4) >= uint32_t(uniform_index_word_11);

    uniform_index_word_4 = uniform_address_word_10 ^ uniform_block_index_13;

    if (!condition_0) {
        matrix_accumulator_2 = blockDim.x;
    }

    uniform_condition_2 = vote_warp_predicate<true>(condition_0);

    uniform_condition_0 = int32_t(uniform_index_word_4) >= int32_t(0u);

    if (!condition_0) {
        matrix_accumulator_3 = blockDim.y;
    }

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_5 + 0x1u;
    }

    uniform_condition_1 = int32_t(uniform_address_word_10) != int32_t(0u);

    uniform_address_word_6 = uniform_index_word_5;

    if (!uniform_condition_2) {
        uniform_index_word_8 = 0u;
    }

    if (!uniform_condition_0) {
        uniform_address_word_6 = 0u - uniform_address_word_6;
    }

    if (!uniform_condition_2) {
        uniform_index_word_5 = 0x400u;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    if (!uniform_condition_1) {
        uniform_address_word_6 = ~uniform_address_word_10;
    }

    if (!uniform_condition_2) {
        uniform_index_word_5 = uniform_index_word_5 + 0x3000u;
    }

    uniform_condition_1 = vote_warp_predicate<true>(condition_0);

    matrix_accumulator_7 = uniform_address_word_6;

    if (!condition_0) {
        matrix_accumulator_4 = matrix_accumulator_2 * matrix_accumulator_3;
    }

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_6];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_7];

    reused_word_112 = matrix_accumulator_7 * 0x4u + matrix_accumulator_38;

    uniform_index_word_4 =
        broadcast_warp_word(matrix_accumulator_4, !condition_0, uniform_index_word_4);

    reused_word_112 = reused_word_112 * 0x40u;

    if (!uniform_condition_2) {
        uniform_index_word_8 = (uniform_index_word_8 << 0x18u) + uniform_index_word_5;
    }

    matrix_accumulator_7 = matrix_accumulator_5 * 0x2000u + reused_word_112;

    matrix_accumulator_7 = matrix_accumulator_7 * 0x8u;

    if (!uniform_condition_2) {
        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);
    }

    uniform_condition_2 = vote_warp_predicate<true>(condition_0);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_accumulator_7 = threadIdx.z;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_8);
#endif
    }

    matrix_accumulator_36 = blockIdx.y;

    uniform_address_word_12 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);

    uniform_index_word_15 = uint32_t(int32_t(uniform_parameter_word_16) >> 0x1fu);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_8 + 0x8u);
#endif
    }

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_8 + 0x10u);
#endif
    }

    __syncthreads();

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_index_word_17));

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_address_word_20));

    uniform_index_word_8 = 0u - uniform_index_word_4 + uniform_index_word_14;

    uniform_address_word_10 = 0u - uniform_index_word_5 + uniform_parameter_word_16;

    uniform_condition_0 = uint32_t(uniform_index_word_8) != 0u;

    uniform_condition_1 = uint32_t(uniform_address_word_10) != 0u;

    condition_0 = (index_word_0 | matrix_accumulator_7) != 0;

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_address_word_12, 0u) << 0x2u >> 32) + uniform_index_word_14;

    uniform_index_word_15 = uint32_t(combine_words(uniform_index_word_15, 0u) << 0x2u >> 32) +
                            uniform_parameter_word_16;

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    if (uniform_condition_0) {
        uniform_index_word_9 = uniform_index_word_8 + 0xffffffffu;
    }

    if (uniform_condition_1) {
        uniform_index_word_11 = uniform_address_word_10 + 0xffffffffu;
    }

    uniform_index_word_8 = uniform_index_word_7 * uniform_address_word_6 + uniform_address_word_6;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x400ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    if (uniform_condition_0) {
        uniform_address_word_10 = uint32_t(int32_t(uniform_index_word_9) >> 0x1fu);
    }

    if (uniform_condition_1) {
        uniform_address_word_12 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);
    }

    uniform_index_word_8 = 0u - uniform_index_word_8 + uniform_block_index_13;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x600ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    if (uniform_condition_0) {
        uniform_address_word_10 =
            uint32_t(combine_words(uniform_address_word_10, 0u) << 0x3u >> 32) +
            uniform_index_word_9;
    }

    if (uniform_condition_1) {
        uniform_address_word_12 =
            uint32_t(combine_words(uniform_address_word_12, 0u) << 0x3u >> 32) +
            uniform_index_word_11;
    }

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0u;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_14) >> 0x2u);

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_15) >> 0x2u);

    if (uniform_condition_0) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_address_word_10)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_1) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_address_word_12)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    uniform_block_index_13 = uniform_index_word_8 << (0x1u & 31u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4000ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4200ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4400ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4600ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_accumulator_3 = matrix_accumulator_36 * 0x8u;
    if (!condition_0) {
        index_word_0 = matrix_accumulator_3 + (0u - uniform_address_word_20);

        uniform_index_word_11 = (uniform_index_word_8 << 0x3u) + (0u - uniform_index_word_17);

        matrix_accumulator_4 = index_word_0 + 0x8u;

        matrix_accumulator_2 = index_word_0 + 0x7u;

        index_word_0 = max(int32_t(0u), int32_t(index_word_0));

        uniform_address_word_10 = uniform_index_word_11 + 0x7u;

        uniform_address_word_12 = uniform_index_word_11 + 0x8u;

        condition_0 = int32_t(matrix_accumulator_4) > int32_t(0u);

        matrix_accumulator_2 =
            uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x1du >> 32) + 0x1u;

        uniform_address_word_10 =
            uint32_t(combine_words(uniform_address_word_10, 0u) << 0x1du >> 32) + 0x1u;

        index_word_0 = uint32_t(uint32_t(index_word_0) >> 0x3u);

        uniform_condition_0 = int32_t(uniform_address_word_12) > int32_t(0u);

        matrix_accumulator_2 = condition_0 ? matrix_accumulator_2 : 0u;

        uniform_address_word_10 = uniform_condition_0 ? uniform_address_word_10 : 0u;

        matrix_accumulator_11 = min(int32_t(matrix_accumulator_2), int32_t(uniform_index_word_4));

        matrix_accumulator_2 = max(int32_t(0u), int32_t(uniform_index_word_11));

        condition_0 = int32_t(index_word_0) >= int32_t(matrix_accumulator_11);
        if (!condition_0) {
            matrix_accumulator_13 = uniform_address_word_10;

            matrix_accumulator_15 = uniform_index_word_5;

            matrix_accumulator_8 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

            matrix_accumulator_13 =
                min(int32_t(matrix_accumulator_13), int32_t(uniform_index_word_5));

            matrix_accumulator_15 = matrix_accumulator_15 * 0x2u;
            do {
                condition_0 = int32_t(matrix_accumulator_8) >= int32_t(matrix_accumulator_13);
                if (!condition_0) {
                    matrix_accumulator_2 = matrix_accumulator_5 + index_word_0;

                    matrix_accumulator_4 = matrix_accumulator_8;

                    matrix_accumulator_17 = matrix_accumulator_15 * matrix_accumulator_2;
                    do {
                        matrix_accumulator_6 = parameters[parameter_wait_low_14];
                        matrix_accumulator_7 = parameters[parameter_wait_high_15];

                        matrix_accumulator_9 = matrix_accumulator_17 + matrix_accumulator_4;

                        split_address_words(
                            matrix_accumulator_6, matrix_accumulator_7,
                            int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_6, matrix_accumulator_7));

                        matrix_accumulator_2 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_6, matrix_accumulator_7),
                            memory_ranges, error_counts);

                        matrix_accumulator_4 = matrix_accumulator_4 + 0x1u;

                        condition_1 =
                            uint32_t(matrix_accumulator_4) != uint32_t(matrix_accumulator_13);

                        condition_0 = int32_t(matrix_accumulator_2) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                matrix_accumulator_2 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_accumulator_6, matrix_accumulator_7),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(matrix_accumulator_2) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                index_word_0 = index_word_0 + 0x1u;

                condition_0 = uint32_t(index_word_0) != uint32_t(matrix_accumulator_11);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    uniform_index_word_14 = parameters[parameter_tensor_height_8];

    matrix_accumulator_9 = uint32_t(uint32_t(matrix_accumulator_38) >> 0x2u);

    uniform_index_word_15 = 0u;

    index_word_0 = uint32_t(uint32_t(matrix_accumulator_38) >> 0x1u);

    __syncthreads();

    matrix_accumulator_2 = matrix_accumulator_36 * 0x2u + matrix_accumulator_9;

    index_word_0 = index_word_0 & 0x1u;

    matrix_accumulator_4 = matrix_accumulator_38 * 0x80u;

    condition_0 = false;

    index_word_0 = index_word_0 + uniform_block_index_13;

    matrix_accumulator_6 = index_word_0;

    uniform_address_word_10 = uniform_index_word_14 & 0xfffffffcu;

    uniform_condition_0 = uint32_t(uniform_address_word_10) != uint32_t(0x4u);

    condition_4 = uniform_condition_0;

    condition_1 = int32_t(matrix_accumulator_2) < int32_t(uniform_index_word_9) || !condition_4;

    matrix_accumulator_2 = matrix_accumulator_2 * uniform_index_word_7;
    if (condition_1) {
        matrix_accumulator_6 = parameters[parameter_tensor_width_9];

        matrix_accumulator_6 = matrix_accumulator_6 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

        matrix_accumulator_6 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_2;

        if (condition_2) {
            matrix_accumulator_6 = index_word_0;
        }
    }
    if (condition_0) {
        matrix_accumulator_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];
    }

    matrix_accumulator_2 = condition_4 ? matrix_accumulator_2 : 0u;

    index_word_14 = matrix_accumulator_4 & 0x80u;

    uniform_parameter_word_16 = 0x400u;

    matrix_accumulator_12 = matrix_accumulator_4 & 0x100u;

    uniform_index_word_17 = (uniform_index_word_15 << 0x18u) + uniform_parameter_word_16;

    if (condition_0) {
        matrix_accumulator_7 = matrix_accumulator_5 + matrix_accumulator_2 + matrix_accumulator_6;
    }

    matrix_accumulator_4 = matrix_accumulator_38 & 0x1u;

    matrix_accumulator_9 = matrix_accumulator_9 * 0x200u + matrix_accumulator_12;

    if (condition_0) {
        matrix_accumulator_8 = matrix_accumulator_7 * 0x800u + index_word_14;
    }

    split_address_words(matrix_accumulator_6, matrix_accumulator_7, 0ull);

    matrix_accumulator_9 = matrix_accumulator_9 + index_word_14;

    if (condition_0) {
        matrix_accumulator_11 = uint32_t(int32_t(matrix_accumulator_8) >> 0x1fu);

        matrix_accumulator_6 = matrix_accumulator_8;

        matrix_accumulator_7 = matrix_accumulator_11;
    }

    split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

    if (condition_0) {
        const uint64_t byte_address_0 =
            scaled_word_address(combine_words(matrix_accumulator_16, matrix_accumulator_17),
                                combine_words(matrix_accumulator_6, matrix_accumulator_7));
        matrix_accumulator_8 = uint32_t(byte_address_0);

        matrix_accumulator_7 = uint32_t(byte_address_0 >> 32);
    }

    matrix_accumulator_6 = (matrix_accumulator_9 << 0x2u) + uniform_index_word_17;

    if (condition_0) {
        matrix_accumulator_10 = matrix_accumulator_8;

        matrix_accumulator_11 = matrix_accumulator_7;
    }
    if (!condition_0) {
        matrix_accumulator_7 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_8 = uniform_address_word_10;

        matrix_accumulator_9 = uniform_address_word_10;

        matrix_accumulator_8 = permute_signed(matrix_accumulator_8, matrix_accumulator_9, 0x5410u);

        matrix_accumulator_9 = matrix_accumulator_8;

        matrix_accumulator_10 = matrix_accumulator_8;

        matrix_accumulator_11 = matrix_accumulator_8;

        matrix_accumulator_7 = matrix_accumulator_7 * 0x10u + matrix_accumulator_6;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_7,
                                             matrix_accumulator_8, matrix_accumulator_9,
                                             matrix_accumulator_10, matrix_accumulator_11,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_10 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_22 = broadcast_active_lane(matrix_accumulator_10);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_23 = broadcast_active_lane(matrix_accumulator_11);

            uniform_address_word_20 = broadcast_active_lane(matrix_accumulator_6);

            uniform_address_word_21 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            matrix_accumulator_7 = 0x200u;

            uniform_address_word_10 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_20,
                combine_words(uniform_address_word_22, uniform_address_word_23),
                uniform_address_word_10 * 16u, uniform_address_word_21, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_20,
                                   combine_words(uniform_address_word_22, uniform_address_word_23),
                                   uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_7 = matrix_accumulator_38 + 0x4u;

    condition_0 = false;

    matrix_accumulator_9 = index_word_0;

    matrix_accumulator_7 = uint32_t(uint32_t(matrix_accumulator_7) >> 0x2u);

    matrix_accumulator_8 = matrix_accumulator_36 * 0x2u + matrix_accumulator_7;

    condition_3 = int32_t(matrix_accumulator_8) < int32_t(uniform_index_word_9) || !condition_4;

    matrix_accumulator_8 = matrix_accumulator_8 * uniform_index_word_7;
    if (condition_3) {
        matrix_accumulator_9 = parameters[parameter_tensor_width_9];

        matrix_accumulator_9 = matrix_accumulator_9 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_9) != uint32_t(0x4u);

        matrix_accumulator_9 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_2;

        if (condition_2) {
            matrix_accumulator_9 = index_word_0;
        }
    }
    if (condition_0) {
        matrix_accumulator_18 = parameters[parameter_input_low_0];
        matrix_accumulator_19 = parameters[parameter_input_high_1];
    }

    matrix_accumulator_8 = condition_4 ? matrix_accumulator_8 : 0u;

    split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

    matrix_accumulator_7 = matrix_accumulator_7 * 0x200u + matrix_accumulator_12;

    if (condition_0) {
        matrix_accumulator_9 = matrix_accumulator_5 + matrix_accumulator_8 + matrix_accumulator_9;
    }

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    matrix_accumulator_7 = index_word_14 + matrix_accumulator_7;

    if (condition_0) {
        matrix_accumulator_9 = matrix_accumulator_9 * 0x800u + index_word_14;
    }

    matrix_accumulator_7 = (matrix_accumulator_7 << 0x2u) + uniform_index_word_17;

    if (condition_0) {
        matrix_accumulator_16 = uint32_t(int32_t(matrix_accumulator_9) >> 0x1fu);

        matrix_accumulator_10 = matrix_accumulator_9;

        matrix_accumulator_11 = matrix_accumulator_16;

        const uint64_t byte_address_1 =
            scaled_word_address(combine_words(matrix_accumulator_18, matrix_accumulator_19),
                                combine_words(matrix_accumulator_10, matrix_accumulator_11));
        matrix_accumulator_9 = uint32_t(byte_address_1);

        matrix_accumulator_10 = uint32_t(byte_address_1 >> 32);

        matrix_accumulator_12 = matrix_accumulator_9;

        matrix_accumulator_13 = matrix_accumulator_10;
    }
    if (!condition_0) {
        matrix_accumulator_10 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_10;

        matrix_accumulator_11 = uniform_address_word_10;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_11, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        index_word_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_9 = matrix_accumulator_10 * 0x10u + matrix_accumulator_7;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_9, matrix_accumulator_12, matrix_accumulator_13,
            index_word_14, matrix_accumulator_15, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_10 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_22 = broadcast_active_lane(matrix_accumulator_12);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_23 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_20 = broadcast_active_lane(matrix_accumulator_7);

            uniform_address_word_21 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            matrix_accumulator_9 = 0x200u;

            uniform_address_word_10 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_20,
                combine_words(uniform_address_word_22, uniform_address_word_23),
                uniform_address_word_10 * 16u, uniform_address_word_21, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_20,
                                   combine_words(uniform_address_word_22, uniform_address_word_23),
                                   uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_9 = index_word_0;
    if (condition_1) {
        matrix_accumulator_9 = parameters[parameter_tensor_width_9];

        matrix_accumulator_9 = matrix_accumulator_9 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_9) != uint32_t(0x4u);

        matrix_accumulator_9 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_2;

        if (condition_2) {
            matrix_accumulator_9 = index_word_0;
        }
    }
    matrix_accumulator_5 = (matrix_accumulator_5 << 0x9u) + 0x40u;
    if (!condition_0) {
        matrix_accumulator_9 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_10;

        matrix_accumulator_11 = uniform_address_word_10;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_11, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        index_word_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_9 = matrix_accumulator_9 * 0x10u + matrix_accumulator_6;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_9 + 0x1000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             index_word_14, matrix_accumulator_15, shared_bytes,
                                             memory_ranges, error_counts);
    } else {
        if (condition_0) {
            index_word_14 = parameters[parameter_input_low_0];
            matrix_accumulator_15 = parameters[parameter_input_high_1];

            matrix_accumulator_9 = matrix_accumulator_2 + matrix_accumulator_9;

            matrix_accumulator_10 =
                uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x1bu >> 32) +
                matrix_accumulator_4;
        }

        uniform_address_word_10 = 0xffffffffu;

        if (condition_0) {
            matrix_accumulator_9 = matrix_accumulator_9 * 0x10u + matrix_accumulator_10;
        }

        split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

        if (condition_0) {
            matrix_accumulator_9 = matrix_accumulator_9 * 0x80u;

            matrix_accumulator_12 = uint32_t(int32_t(matrix_accumulator_9) >> 0x1fu);

            matrix_accumulator_10 = matrix_accumulator_9;

            matrix_accumulator_11 = matrix_accumulator_12;
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        if (condition_0) {
            const uint64_t byte_address_2 =
                scaled_word_address(combine_words(index_word_14, matrix_accumulator_15),
                                    combine_words(matrix_accumulator_10, matrix_accumulator_11));
            matrix_accumulator_9 = uint32_t(byte_address_2);

            matrix_accumulator_10 = uint32_t(byte_address_2 >> 32);

            matrix_accumulator_12 = matrix_accumulator_9;

            matrix_accumulator_13 = matrix_accumulator_10;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_11 = broadcast_active_lane(matrix_accumulator_6);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_20 = broadcast_active_lane(matrix_accumulator_12);

            uniform_address_word_22 = 0x20u;

            uniform_address_word_21 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_12 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            matrix_accumulator_9 = 0x200u;

            uniform_address_word_10 = uniform_index_word_11 + 0x1000u;

            uniform_index_word_11 = uniform_address_word_12 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_10,
                combine_words(uniform_address_word_20, uniform_address_word_21),
                uniform_address_word_22 * 16u, uniform_address_word_12 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_10,
                                   combine_words(uniform_address_word_20, uniform_address_word_21),
                                   uniform_address_word_22 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_9 = index_word_0;
    if (condition_3) {
        matrix_accumulator_9 = parameters[parameter_tensor_width_9];

        matrix_accumulator_9 = matrix_accumulator_9 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_9) != uint32_t(0x4u);

        matrix_accumulator_9 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_2;

        if (condition_2) {
            matrix_accumulator_9 = index_word_0;
        }
    }
    if (!condition_0) {
        matrix_accumulator_10 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_10;

        matrix_accumulator_9 = uniform_address_word_10;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_9, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        index_word_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_10 = matrix_accumulator_10 * 0x10u + matrix_accumulator_7;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_10 + 0x1000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             index_word_14, matrix_accumulator_15, shared_bytes,
                                             memory_ranges, error_counts);
    } else {
        if (condition_0) {
            index_word_14 = parameters[parameter_input_low_0];
            matrix_accumulator_15 = parameters[parameter_input_high_1];

            matrix_accumulator_9 = matrix_accumulator_8 + matrix_accumulator_9;

            matrix_accumulator_10 =
                uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x1bu >> 32) +
                matrix_accumulator_4;
        }

        uniform_address_word_10 = 0xffffffffu;

        if (condition_0) {
            matrix_accumulator_9 = matrix_accumulator_9 * 0x10u + matrix_accumulator_10;
        }

        split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

        if (condition_0) {
            matrix_accumulator_9 = matrix_accumulator_9 * 0x80u;

            matrix_accumulator_12 = uint32_t(int32_t(matrix_accumulator_9) >> 0x1fu);

            matrix_accumulator_10 = matrix_accumulator_9;

            matrix_accumulator_11 = matrix_accumulator_12;
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        if (condition_0) {
            const uint64_t byte_address_3 =
                scaled_word_address(combine_words(index_word_14, matrix_accumulator_15),
                                    combine_words(matrix_accumulator_10, matrix_accumulator_11));
            matrix_accumulator_9 = uint32_t(byte_address_3);

            matrix_accumulator_10 = uint32_t(byte_address_3 >> 32);

            matrix_accumulator_12 = matrix_accumulator_9;

            matrix_accumulator_13 = matrix_accumulator_10;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_11 = broadcast_active_lane(matrix_accumulator_7);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_20 = broadcast_active_lane(matrix_accumulator_12);

            uniform_address_word_22 = 0x20u;

            uniform_address_word_21 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_12 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            matrix_accumulator_9 = 0x200u;

            uniform_address_word_10 = uniform_index_word_11 + 0x1000u;

            uniform_index_word_11 = uniform_address_word_12 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_10,
                combine_words(uniform_address_word_20, uniform_address_word_21),
                uniform_address_word_22 * 16u, uniform_address_word_12 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_10,
                                   combine_words(uniform_address_word_20, uniform_address_word_21),
                                   uniform_address_word_22 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_9 = index_word_0;
    if (condition_1) {
        matrix_accumulator_9 = parameters[parameter_tensor_width_9];

        matrix_accumulator_9 = matrix_accumulator_9 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_9) != uint32_t(0x4u);

        matrix_accumulator_9 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_1;

        if (condition_1) {
            matrix_accumulator_9 = index_word_0;
        }
    }
    matrix_accumulator_5 = matrix_accumulator_5 + 0x40u;
    if (!condition_0) {
        matrix_accumulator_9 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_10;

        matrix_accumulator_11 = uniform_address_word_10;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_11, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        index_word_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_9 = matrix_accumulator_9 * 0x10u + matrix_accumulator_6;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_9 + 0x2000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             index_word_14, matrix_accumulator_15, shared_bytes,
                                             memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_12 = parameters[parameter_input_low_0];
            matrix_accumulator_13 = parameters[parameter_input_high_1];

            matrix_accumulator_2 = matrix_accumulator_2 + matrix_accumulator_9;

            matrix_accumulator_9 =
                uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x1bu >> 32) +
                matrix_accumulator_4;
        }

        split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

        uniform_address_word_10 = 0xffffffffu;

        split_address_words(index_word_14, matrix_accumulator_15, 0ull);

        if (condition_0) {
            matrix_accumulator_2 = matrix_accumulator_2 * 0x10u + matrix_accumulator_9;

            matrix_accumulator_2 = matrix_accumulator_2 * 0x80u;

            matrix_accumulator_9 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);

            matrix_accumulator_10 = matrix_accumulator_2;

            matrix_accumulator_11 = matrix_accumulator_9;

            const uint64_t byte_address_4 =
                scaled_word_address(combine_words(matrix_accumulator_12, matrix_accumulator_13),
                                    combine_words(matrix_accumulator_10, matrix_accumulator_11));
            matrix_accumulator_2 = uint32_t(byte_address_4);

            matrix_accumulator_11 = uint32_t(byte_address_4 >> 32);

            index_word_14 = matrix_accumulator_2;

            matrix_accumulator_15 = matrix_accumulator_11;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_11 = broadcast_active_lane(matrix_accumulator_6);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_20 = broadcast_active_lane(index_word_14);

            uniform_address_word_22 = 0x20u;

            uniform_address_word_21 = broadcast_active_lane(matrix_accumulator_15);

            uniform_address_word_12 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            matrix_accumulator_2 = 0x200u;

            uniform_address_word_10 = uniform_index_word_11 + 0x2000u;

            uniform_index_word_11 = uniform_address_word_12 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_10,
                combine_words(uniform_address_word_20, uniform_address_word_21),
                uniform_address_word_22 * 16u, uniform_address_word_12 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_10,
                                   combine_words(uniform_address_word_20, uniform_address_word_21),
                                   uniform_address_word_22 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_9 = index_word_0;
    if (condition_3) {
        matrix_accumulator_2 = parameters[parameter_tensor_width_9];

        matrix_accumulator_9 = 0u;

        matrix_accumulator_2 = matrix_accumulator_2 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_0) < int32_t(uniform_index_word_7) || !condition_1;

        if (condition_1) {
            matrix_accumulator_9 = index_word_0;
        }
    }
    if (!condition_0) {
        index_word_0 = threadIdx.x & 31u;

        uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));

        matrix_accumulator_4 = uniform_address_word_10;

        matrix_accumulator_5 = uniform_address_word_10;

        matrix_accumulator_4 = permute_signed(matrix_accumulator_4, matrix_accumulator_5, 0x5410u);

        matrix_accumulator_5 = matrix_accumulator_4;

        matrix_accumulator_6 = matrix_accumulator_4;

        index_word_0 = index_word_0 * 0x10u + matrix_accumulator_7;

        matrix_accumulator_7 = matrix_accumulator_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, index_word_0 + 0x2000u, matrix_accumulator_4, matrix_accumulator_5,
            matrix_accumulator_6, matrix_accumulator_7, shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_10 = parameters[parameter_input_low_0];
            matrix_accumulator_11 = parameters[parameter_input_high_1];

            matrix_accumulator_9 = matrix_accumulator_8 + matrix_accumulator_9;

            index_word_0 = uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x1bu >> 32) +
                           matrix_accumulator_4;
        }

        split_address_words(matrix_accumulator_4, matrix_accumulator_5, 0ull);

        uniform_address_word_10 = 0xffffffffu;

        if (condition_0) {
            index_word_0 = matrix_accumulator_9 * 0x10u + index_word_0;

            index_word_0 = index_word_0 * 0x80u;

            matrix_accumulator_9 = uint32_t(int32_t(index_word_0) >> 0x1fu);

            matrix_accumulator_4 = index_word_0;

            matrix_accumulator_5 = matrix_accumulator_9;
        }

        split_address_words(matrix_accumulator_8, matrix_accumulator_9, 0ull);

        if (condition_0) {
            const uint64_t byte_address_5 =
                scaled_word_address(combine_words(matrix_accumulator_10, matrix_accumulator_11),
                                    combine_words(matrix_accumulator_4, matrix_accumulator_5));
            index_word_0 = uint32_t(byte_address_5);

            matrix_accumulator_5 = uint32_t(byte_address_5 >> 32);

            matrix_accumulator_8 = index_word_0;

            matrix_accumulator_9 = matrix_accumulator_5;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_11 = broadcast_active_lane(matrix_accumulator_7);

            uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

            uniform_address_word_20 = broadcast_active_lane(matrix_accumulator_8);

            uniform_address_word_22 = 0x20u;

            uniform_address_word_21 = broadcast_active_lane(matrix_accumulator_9);

            uniform_address_word_12 = (uniform_index_word_15 << 0x18u) + uniform_address_word_10;

            index_word_0 = 0x200u;

            uniform_address_word_10 = uniform_index_word_11 + 0x2000u;

            uniform_index_word_11 = uniform_address_word_12 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_10,
                combine_words(uniform_address_word_20, uniform_address_word_21),
                uniform_address_word_22 * 16u, uniform_address_word_12 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_10,
                                   combine_words(uniform_address_word_20, uniform_address_word_21),
                                   uniform_address_word_22 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_9 = 0u;

    uniform_address_word_10 = uniform_parameter_word_16 + 0x3000u;

    index_word_14 = parameters[parameter_tensor_width_9];

    matrix_accumulator_11 = uniform_index_word_8;

    matrix_accumulator_13 = threadIdx.x & 31u;

    matrix_accumulator_15 = matrix_accumulator_15 & 0xfffffffdu;

    index_word_0 = uniform_address_word_10;

    matrix_accumulator_9 = (matrix_accumulator_9 << 0x18u) + index_word_0;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_accumulator_9;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_6, matrix_accumulator_7,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_6 = 0u;
#endif

    matrix_accumulator_21 = matrix_accumulator_13;

    matrix_accumulator_4 = matrix_accumulator_13;

    index_word_0 = uint32_t(int32_t(matrix_accumulator_21) >> 0x2u);

    matrix_accumulator_3 = uint32_t(int64_t(int32_t(matrix_accumulator_4)) * (1ll << 0x1cu) >> 32) +
                           matrix_accumulator_3;

    matrix_accumulator_2 = uint32_t(combine_words(index_word_0, 0u) << 0x2u >> 32) + index_word_0;

    condition_0 = int32_t(matrix_accumulator_3) >= int32_t(uniform_index_word_14);

    matrix_accumulator_5 = matrix_accumulator_2 & 0xfffffffcu;

    condition_1 = int32_t(matrix_accumulator_3) < int32_t(0u) || condition_0;

    matrix_accumulator_20 = index_word_0 + (0u - matrix_accumulator_5);

    matrix_accumulator_5 = uniform_index_word_14;

    matrix_accumulator_20 = matrix_accumulator_11 * 0x8u + matrix_accumulator_20;

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_4 = wait_copy_barrier(shared_tile, matrix_accumulator_9,
                                    async_tokens[(((matrix_accumulator_9) >> 3) & 1)]);
#else
    __syncthreads();
    condition_4 = true;
#endif

    condition_3 = (uint32_t(matrix_accumulator_5) != uint32_t(0x1u)) && (condition_1);

    condition_1 = uint32_t(matrix_accumulator_5) == uint32_t(0x1u) || !condition_1;

    matrix_accumulator_11 = uniform_address_word_6;

    condition_2 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_0 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_5 = uint32_t(index_word_14) == uint32_t(0x1u) || condition_3;

    condition_2 = int32_t(matrix_accumulator_20) > -1 && !condition_2;

    if (condition_1) {
        matrix_accumulator_15 = 0x2u | matrix_accumulator_15;
    }

    condition_1 = int32_t(matrix_accumulator_20) > -1 && !condition_0;

    matrix_accumulator_15 = matrix_accumulator_15 & 0xfffffffeu;

    if (condition_5) {
        matrix_accumulator_15 = 0x1u | matrix_accumulator_15;
    }
    if (!condition_4) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_0 = wait_copy_barrier(shared_tile, matrix_accumulator_9,
                                            async_tokens[(((matrix_accumulator_9) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
        } while (!condition_0);
    }
    condition_5 = uint32_t(index_word_14) == uint32_t(0x1u) || condition_3;

    condition_6 = (matrix_accumulator_15 & 0x2u) != 0;

    uniform_condition_0 = uint32_t(uniform_index_word_14) != uint32_t(0x1u);

    matrix_accumulator_24 = matrix_accumulator_11 * 0x4u + matrix_accumulator_38;

    matrix_accumulator_30 = 0u;

    condition_0 = condition_6 & (condition_5 | condition_2);

    matrix_accumulator_7 = index_word_14 * 0x4u;

    condition_1 = condition_6 & (condition_5 | condition_1);

    matrix_accumulator_24 = matrix_accumulator_24 * 0x4u;

    condition_2 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_4 = !uniform_condition_0;

    matrix_fragment_73 = matrix_accumulator_20 + 0x4u;

    matrix_accumulator_4 = matrix_accumulator_24 + 0x1u;

    condition_2 = int32_t(matrix_accumulator_20) > -1 && !condition_2;

    matrix_accumulator_19 = !condition_4 ? matrix_accumulator_3 : 0u;

    matrix_accumulator_16 = matrix_accumulator_24 + 0x2u;

    condition_4 = (matrix_accumulator_15 & 0x1u) != 0;

    if (condition_0) {
        matrix_accumulator_8 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_9 = parameters[parameter_input_skip_high_3];

        index_word_0 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_2 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_19;
    }

    condition_2 = condition_6 & (condition_5 | condition_2);

    matrix_fragment_72 = 0u;

    if (condition_0) {
        index_word_0 = 0u - index_word_0 + matrix_accumulator_13;
    }

    if (condition_1) {
        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_26 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_27 = parameters[parameter_input_skip_high_3];
    }

    condition_5 = uint32_t(index_word_14) == uint32_t(0x1u) && !condition_3;

    condition_3 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    if (condition_0) {
        matrix_accumulator_11 = matrix_accumulator_7 * matrix_accumulator_2 + index_word_0;
    }

    if (condition_1) {
        matrix_accumulator_17 = 0u - matrix_accumulator_6 + matrix_accumulator_13;
    }

    if (condition_0) {
        index_word_0 = matrix_accumulator_20 * 0x4u;
    }

    matrix_accumulator_22 = matrix_accumulator_24 + 0x3u;

    if (condition_1) {
        matrix_accumulator_2 = matrix_accumulator_20 * 0x4u;
    }

    if (condition_2) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];
    }

    if (condition_1) {
        matrix_accumulator_6 = matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_19;
    }

    if (condition_0) {
        index_word_0 = !condition_5 ? index_word_0 : 0u;
    }

    condition_3 = int32_t(matrix_accumulator_20) > -1 && !condition_3;

    if (condition_1) {
        matrix_accumulator_2 = !condition_5 ? matrix_accumulator_2 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_17;
    }

    if (condition_2) {
        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;
    }

    matrix_fragment_79 = 0u;

    if (condition_0) {
        matrix_accumulator_11 = index_word_0 + matrix_accumulator_11;
    }

    if (condition_1) {
        matrix_accumulator_17 = matrix_accumulator_2 + matrix_accumulator_17;
    }

    if (condition_2) {
        matrix_accumulator_23 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        index_word_0 = matrix_accumulator_20 * 0x4u;
    }

    condition_3 = condition_6 & (condition_4 | condition_3);

    if (condition_0) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(matrix_accumulator_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    matrix_accumulator_10 = 0;
    matrix_accumulator_11 = 0;

    matrix_accumulator_15 = matrix_accumulator_15 & 0xffffffdfu;

    matrix_accumulator_88 = 0;
    matrix_accumulator_89 = 0;

    if (condition_2) {
        matrix_accumulator_6 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_19;

        index_word_0 = !condition_5 ? index_word_0 : 0u;
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    if (condition_0) {
        matrix_accumulator_30 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_8, matrix_accumulator_9), memory_ranges, error_counts);
    }

    condition_0 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_4;

    if (condition_5) {
        matrix_accumulator_15 = 0x20u | matrix_accumulator_15;
    }

    if (condition_2) {
        matrix_accumulator_23 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_23;
    }

    if (condition_1) {
        matrix_accumulator_11 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_1 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_4;

    condition_0 = condition_0 & condition_6;

    if (condition_3) {
        matrix_accumulator_34 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_35 = parameters[parameter_input_skip_high_3];
    }

    if (condition_2) {
        matrix_accumulator_23 = index_word_0 + matrix_accumulator_23;
    }

    matrix_fragment_80 = 0;
    matrix_fragment_81 = 0;

    condition_1 = condition_1 & condition_6;

    matrix_accumulator_90 = 0u;

    if (condition_2) {
        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    if (condition_3) {
        index_word_0 = matrix_accumulator_21 & 0xfffffffcu;
    }

    matrix_fragment_82 = 0;
    matrix_fragment_83 = 0;

    matrix_fragment_76 = reused_word_112 + 0x28u;

    uniform_address_word_10 = parameters[parameter_optional_control_16];
    uniform_index_word_11 = parameters[parameter_optional_control_17];

    if (condition_2) {
        matrix_accumulator_10 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_2 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_4;

    if (condition_3) {
        matrix_accumulator_17 = 0u - index_word_0 + matrix_accumulator_13;

        index_word_0 = matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_19;
    }

    if (condition_0) {
        matrix_accumulator_2 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_26 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_27 = parameters[parameter_input_skip_high_3];
    }

    if (condition_1) {
        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;
    }

    condition_2 = condition_2 & condition_6;

    if (condition_3) {
        matrix_accumulator_17 = matrix_accumulator_7 * index_word_0 + matrix_accumulator_17;
    }

    if (condition_0) {
        matrix_accumulator_2 = 0u - matrix_accumulator_2 + matrix_accumulator_13;
    }

    if (condition_3) {
        index_word_0 = matrix_accumulator_20 * 0x4u;
    }

    if (condition_1) {
        matrix_accumulator_12 = 0u - matrix_accumulator_6 + matrix_accumulator_13;
    }

    if (condition_0) {
        matrix_accumulator_6 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_19;
    }

    if (condition_1) {
        matrix_accumulator_8 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_9 = parameters[parameter_input_skip_high_3];
    }

    if (condition_3) {
        index_word_0 = !condition_5 ? index_word_0 : 0u;
    }

    if (condition_0) {
        matrix_accumulator_23 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_2;

        matrix_accumulator_2 = matrix_fragment_73 * 0x4u;
    }

    if (condition_1) {
        matrix_accumulator_18 =
            matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_19;
    }

    if (condition_2) {
        matrix_accumulator_32 = matrix_accumulator_21 & 0xfffffffcu;
    }

    if (condition_1) {
        matrix_accumulator_6 = matrix_fragment_73 * 0x4u;
    }

    if (condition_3) {
        matrix_accumulator_17 = index_word_0 + matrix_accumulator_17;
    }

    if (condition_0) {
        matrix_accumulator_2 = !condition_5 ? matrix_accumulator_2 : 0u;
    }

    if (condition_1) {
        matrix_fragment_75 = matrix_accumulator_7 * matrix_accumulator_18 + matrix_accumulator_12;
    }

    if (condition_2) {
        matrix_accumulator_32 = 0u - matrix_accumulator_32 + matrix_accumulator_13;

        matrix_accumulator_12 = matrix_fragment_73 * 0x4u;
    }

    if (condition_1) {
        matrix_accumulator_6 = !condition_5 ? matrix_accumulator_6 : 0u;
    }

    if (condition_2) {
        matrix_accumulator_18 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_19;

        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_34, matrix_accumulator_35,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_34, matrix_accumulator_35));
    }

    if (condition_0) {
        matrix_accumulator_17 = matrix_accumulator_23 + matrix_accumulator_2;
    }

    matrix_accumulator_23 = matrix_accumulator_3 + 0x2u;

    if (condition_2) {
        matrix_accumulator_12 = !condition_5 ? matrix_accumulator_12 : 0u;

        matrix_accumulator_25 =
            matrix_accumulator_7 * matrix_accumulator_18 + matrix_accumulator_32;
    }

    if (condition_1) {
        matrix_fragment_75 = matrix_fragment_75 + matrix_accumulator_6;
    }

    if (condition_3) {
        matrix_fragment_72 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_34, matrix_accumulator_35), memory_ranges,
            error_counts);
    }

    condition_4 = int32_t(matrix_accumulator_23) >= int32_t(uniform_index_word_14);

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    if (condition_2) {
        matrix_accumulator_25 = matrix_accumulator_25 + matrix_accumulator_12;
    }

    matrix_accumulator_12 = 0u;

    if (condition_0) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    condition_4 = int32_t(matrix_accumulator_23) < int32_t(0u) || condition_4;

    if (condition_1) {
        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    if (condition_0) {
        matrix_accumulator_32 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    if (condition_1) {
        matrix_accumulator_12 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_1 = uint32_t(matrix_accumulator_5) != uint32_t(0x1u) && condition_4;

    condition_4 = uint32_t(matrix_accumulator_5) == uint32_t(0x1u) || !condition_4;

    condition_0 = int32_t(matrix_accumulator_20) > -1 && !condition_0;

    condition_3 = uint32_t(index_word_14) == uint32_t(0x1u) || condition_1;

    condition_0 = condition_4 & (condition_3 | condition_0);

    if (condition_2) {
        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    matrix_accumulator_9 = 0u;

    if (condition_2) {
        matrix_accumulator_9 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_2 = !uniform_condition_0;

    if (condition_0) {
        matrix_accumulator_26 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_27 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_2 = matrix_accumulator_21 & 0xfffffffcu;
    }

    matrix_accumulator_23 = !condition_2 ? matrix_accumulator_23 : 0u;

    condition_5 = uint32_t(index_word_14) == uint32_t(0x1u) && !condition_1;

    if (condition_0) {
        index_word_0 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_17 = 0u - matrix_accumulator_2 + matrix_accumulator_13;

        matrix_accumulator_2 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_23;

        index_word_0 = !condition_5 ? index_word_0 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_2 + matrix_accumulator_17;

        matrix_accumulator_17 = index_word_0 + matrix_accumulator_17;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));

        matrix_accumulator_33 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_0 = int32_t(matrix_accumulator_20) > -1 && !condition_0;

    condition_0 = condition_4 & (condition_3 | condition_0);

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_2 = matrix_accumulator_21 & 0xfffffffcu;

        index_word_0 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_17 = 0u - matrix_accumulator_2 + matrix_accumulator_13;

        matrix_accumulator_2 = matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_23;

        index_word_0 = !condition_5 ? index_word_0 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_2 + matrix_accumulator_17;
    }

    matrix_accumulator_18 = 0u;

    if (condition_0) {
        matrix_accumulator_17 = index_word_0 + matrix_accumulator_17;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_18 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_0 = int32_t(matrix_accumulator_20) > -1 && !condition_0;

    condition_0 = condition_4 & (condition_3 | condition_0);

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_2 = matrix_accumulator_21 & 0xfffffffcu;

        index_word_0 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_17 = 0u - matrix_accumulator_2 + matrix_accumulator_13;

        matrix_accumulator_2 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_23;

        index_word_0 = !condition_5 ? index_word_0 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_2 + matrix_accumulator_17;
    }

    matrix_accumulator_2 = 0u;

    if (condition_0) {
        matrix_accumulator_17 = index_word_0 + matrix_accumulator_17;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_2 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_0 = int32_t(matrix_accumulator_20) > -1 && !condition_0;

    condition_0 = condition_4 & (condition_3 | condition_0);

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        index_word_0 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_17 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        matrix_accumulator_6 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_23;

        index_word_0 = !condition_5 ? index_word_0 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_17;

        matrix_accumulator_17 = index_word_0 + matrix_accumulator_17;
    }

    index_word_0 = 0u;

    if (condition_0) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        index_word_0 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_3;

    condition_0 = condition_0 & condition_4;

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_8 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_6 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_17 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_23;

        matrix_accumulator_8 = 0u - matrix_accumulator_8 + matrix_accumulator_13;

        matrix_accumulator_6 = !condition_5 ? matrix_accumulator_6 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_17 + matrix_accumulator_8;
    }

    matrix_accumulator_34 = 0;
    matrix_accumulator_35 = 0;

    if (condition_0) {
        matrix_accumulator_17 = matrix_accumulator_17 + matrix_accumulator_6;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_34 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_3;

    condition_0 = condition_0 & condition_4;

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_8 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_6 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_17 =
            matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_23;

        matrix_accumulator_8 = 0u - matrix_accumulator_8 + matrix_accumulator_13;

        matrix_accumulator_6 = !condition_5 ? matrix_accumulator_6 : 0u;

        matrix_accumulator_17 = matrix_accumulator_7 * matrix_accumulator_17 + matrix_accumulator_8;

        matrix_accumulator_17 = matrix_accumulator_17 + matrix_accumulator_6;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_fragment_79 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_3;

    condition_0 = condition_0 & condition_4;

    if (condition_0) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_8 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_6 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_25 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_23;

        matrix_accumulator_8 = 0u - matrix_accumulator_8 + matrix_accumulator_13;

        matrix_accumulator_6 = !condition_5 ? matrix_accumulator_6 : 0u;

        matrix_accumulator_25 = matrix_accumulator_7 * matrix_accumulator_25 + matrix_accumulator_8;
    }

    matrix_accumulator_8 = 0u;

    if (condition_0) {
        matrix_accumulator_25 = matrix_accumulator_25 + matrix_accumulator_6;
    }

    matrix_accumulator_6 = matrix_accumulator_3 + 0x4u;

    condition_2 = int32_t(matrix_accumulator_6) >= int32_t(uniform_index_word_14);

    if (condition_0) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    matrix_accumulator_15 = matrix_accumulator_15 & 0xffffffefu;

    condition_2 = int32_t(matrix_accumulator_6) < int32_t(0u) || condition_2;

    if (condition_0) {
        matrix_accumulator_8 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    condition_0 = !uniform_condition_0;

    if (condition_5) {
        matrix_accumulator_15 = 0x10u | matrix_accumulator_15;
    }

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    matrix_accumulator_25 = !condition_0 ? matrix_accumulator_6 : 0u;

    condition_0 = uint32_t(matrix_accumulator_5) != uint32_t(0x1u) && condition_2;

    condition_2 = uint32_t(matrix_accumulator_5) == uint32_t(0x1u) || !condition_2;

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_1 = uint32_t(index_word_14) == uint32_t(0x1u) || condition_0;

    condition_5 = condition_2 & (condition_1 | condition_5);

    matrix_accumulator_17 = uint32_t(condition_3) << 3;

    condition_3 = (matrix_accumulator_15 & 0x2u) != 0;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_26 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_6 = matrix_accumulator_20 * 0x4u;
    }

    matrix_accumulator_37 = uint32_t(condition_3) << 3;

    condition_3 = uint32_t(index_word_14) == uint32_t(0x1u) && !condition_0;

    if (condition_5) {
        matrix_accumulator_27 = 0u - matrix_accumulator_26 + matrix_accumulator_13;

        matrix_accumulator_26 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_6 = !condition_3 ? matrix_accumulator_6 : 0u;

        matrix_accumulator_27 =
            matrix_accumulator_7 * matrix_accumulator_26 + matrix_accumulator_27;
    }

    matrix_accumulator_3 = matrix_accumulator_3 + 0x6u;

    condition_0 = !uniform_condition_0;

    if (condition_5) {
        matrix_accumulator_27 = matrix_accumulator_6 + matrix_accumulator_27;

        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    matrix_accumulator_27 = !condition_0 ? matrix_accumulator_3 : 0u;

    condition_0 = int32_t(matrix_accumulator_3) >= int32_t(uniform_index_word_14);

    if (condition_5) {
        matrix_accumulator_35 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_0 = int32_t(matrix_accumulator_3) < int32_t(0u) || condition_0;

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    matrix_accumulator_31 = uint32_t(condition_4) << 4;

    condition_6 = uint32_t(matrix_accumulator_5) != uint32_t(0x1u) && condition_0;

    condition_4 = uint32_t(matrix_accumulator_5) == uint32_t(0x1u) || !condition_0;

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_0 = uint32_t(index_word_14) == uint32_t(0x1u) || condition_6;

    condition_5 = condition_4 & (condition_0 | condition_5);

    condition_6 = uint32_t(index_word_14) == uint32_t(0x1u) && !condition_6;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_3 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_5 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        matrix_accumulator_6 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_3 = !condition_6 ? matrix_accumulator_3 : 0u;

        matrix_accumulator_6 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_5;

        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_6;

        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_88 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_5 = condition_2 & (condition_1 | condition_5);

    if (condition_5) {
        matrix_fragment_74 = parameters[parameter_input_skip_low_2];
        matrix_fragment_75 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_3 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_5 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        matrix_accumulator_6 = matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_3 = !condition_3 ? matrix_accumulator_3 : 0u;

        matrix_accumulator_6 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_5;

        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_6;

        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_74, matrix_fragment_75));

        matrix_fragment_80 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_5 = condition_4 & (condition_0 | condition_5);

    if (condition_5) {
        matrix_fragment_74 = parameters[parameter_input_skip_low_2];
        matrix_fragment_75 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_3 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_5 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        matrix_accumulator_6 = matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_3 = !condition_6 ? matrix_accumulator_3 : 0u;

        matrix_accumulator_6 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_5;

        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_6;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_74, matrix_fragment_75));

        matrix_fragment_81 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_5 = condition_2 & (condition_1 | condition_5);

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_6 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_3 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_5 = 0u - matrix_accumulator_6 + matrix_accumulator_13;

        matrix_accumulator_6 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_3 = !condition_3 ? matrix_accumulator_3 : 0u;

        matrix_accumulator_6 = matrix_accumulator_7 * matrix_accumulator_6 + matrix_accumulator_5;

        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_6;
    }

    matrix_accumulator_6 = 0u;

    if (condition_5) {
        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_6 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_5 = int32_t(matrix_accumulator_20) > -1 && !condition_5;

    condition_5 = condition_4 & (condition_0 | condition_5);

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_26 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_3 = matrix_accumulator_20 * 0x4u;

        matrix_accumulator_5 = 0u - matrix_accumulator_26 + matrix_accumulator_13;

        matrix_accumulator_26 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_3 = !condition_6 ? matrix_accumulator_3 : 0u;

        matrix_accumulator_26 = matrix_accumulator_7 * matrix_accumulator_26 + matrix_accumulator_5;

        matrix_accumulator_3 = matrix_accumulator_3 + matrix_accumulator_26;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    matrix_accumulator_3 = 0u;

    if (condition_5) {
        matrix_accumulator_3 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_1;

    condition_5 = condition_5 & condition_2;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_26 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_5 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_39 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_26 = 0u - matrix_accumulator_26 + matrix_accumulator_13;

        matrix_accumulator_5 = !condition_3 ? matrix_accumulator_5 : 0u;

        matrix_accumulator_26 =
            matrix_accumulator_7 * matrix_accumulator_39 + matrix_accumulator_26;

        matrix_accumulator_5 = matrix_accumulator_26 + matrix_accumulator_5;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_89 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_0;

    condition_5 = condition_5 & condition_4;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_26 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_5 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_24 =
            matrix_accumulator_24 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_26 = 0u - matrix_accumulator_26 + matrix_accumulator_13;

        matrix_accumulator_5 = !condition_6 ? matrix_accumulator_5 : 0u;

        matrix_accumulator_24 =
            matrix_accumulator_7 * matrix_accumulator_24 + matrix_accumulator_26;

        matrix_accumulator_5 = matrix_accumulator_24 + matrix_accumulator_5;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_accumulator_90 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_1;

    condition_5 = condition_5 & condition_2;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_24 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_5 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_26 =
            matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_24 = 0u - matrix_accumulator_24 + matrix_accumulator_13;

        matrix_accumulator_5 = !condition_3 ? matrix_accumulator_5 : 0u;

        matrix_accumulator_24 =
            matrix_accumulator_7 * matrix_accumulator_26 + matrix_accumulator_24;

        matrix_accumulator_5 = matrix_accumulator_24 + matrix_accumulator_5;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_fragment_82 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_0;

    condition_5 = condition_5 & condition_4;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_24 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_5 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_4 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_24 = 0u - matrix_accumulator_24 + matrix_accumulator_13;

        matrix_accumulator_5 = !condition_6 ? matrix_accumulator_5 : 0u;

        matrix_accumulator_4 = matrix_accumulator_7 * matrix_accumulator_4 + matrix_accumulator_24;

        matrix_accumulator_5 = matrix_accumulator_4 + matrix_accumulator_5;

        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));

        matrix_fragment_83 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_1;

    condition_5 = condition_5 & condition_2;

    if (condition_5) {
        matrix_accumulator_4 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_5 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_26 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_24 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_39 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_25;

        matrix_accumulator_26 = 0u - matrix_accumulator_26 + matrix_accumulator_13;

        matrix_accumulator_24 = !condition_3 ? matrix_accumulator_24 : 0u;

        matrix_accumulator_39 =
            matrix_accumulator_7 * matrix_accumulator_39 + matrix_accumulator_26;

        matrix_fragment_75 = matrix_accumulator_39 + matrix_accumulator_24;

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_4, matrix_accumulator_5));
    }

    matrix_accumulator_4 = 0u;

    if (condition_5) {
        matrix_accumulator_4 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), memory_ranges, error_counts);
    }

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_0;

    condition_5 = condition_5 & condition_4;

    if (condition_5) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_24 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_5 = matrix_fragment_73 * 0x4u;

        matrix_accumulator_16 =
            matrix_accumulator_16 * uniform_index_word_14 + matrix_accumulator_27;

        matrix_accumulator_24 = 0u - matrix_accumulator_24 + matrix_accumulator_13;

        matrix_accumulator_5 = !condition_6 ? matrix_accumulator_5 : 0u;
    }

    matrix_accumulator_15 = matrix_accumulator_15 & 0xfffffff7u;

    if (condition_5) {
        matrix_accumulator_16 =
            matrix_accumulator_7 * matrix_accumulator_16 + matrix_accumulator_24;
    }

    if (condition_3) {
        matrix_accumulator_15 = 0x8u | matrix_accumulator_15;
    }

    if (condition_5) {
        matrix_accumulator_5 = matrix_accumulator_16 + matrix_accumulator_5;
    }

    matrix_accumulator_15 = matrix_accumulator_15 & 0xfffffffbu;

    if (condition_5) {
        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    matrix_accumulator_5 = 0u;

    if (condition_4) {
        matrix_accumulator_15 = 0x4u | matrix_accumulator_15;
    }

    if (condition_5) {
        matrix_accumulator_5 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    condition_5 = (matrix_accumulator_15 & 0x1u) != 0;

    condition_3 = int32_t(matrix_accumulator_37) != int32_t(0u);

    condition_5 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_5;

    condition_5 = condition_5 & condition_3;

    condition_3 = int32_t(matrix_accumulator_17) != int32_t(0u);

    condition_4 = int32_t(matrix_accumulator_31) != int32_t(0u);

    condition_3 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_3;

    condition_4 = condition_3 & condition_4;

    condition_3 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    matrix_accumulator_37 = condition_6 ? 0x40u : 0u;

    condition_3 = int32_t(matrix_accumulator_20) > -1 && !condition_3;

    condition_6 = (matrix_accumulator_15 & 0x8u) != 0;

    condition_3 = condition_2 & (condition_1 | condition_3);

    condition_1 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_1;

    matrix_accumulator_31 = condition_6 ? 0x40u : 0u;

    condition_6 = (matrix_accumulator_15 & 0x10u) != 0;

    condition_2 = condition_1 & condition_2;

    matrix_accumulator_26 = condition_6 ? 0x40u : 0u;

    condition_1 = int32_t(matrix_accumulator_20) >= int32_t(index_word_14);

    condition_6 = (matrix_accumulator_15 & 0x20u) != 0;

    matrix_accumulator_16 = matrix_accumulator_21 & 0x7ffffffcu;

    matrix_accumulator_24 = condition_6 ? 0x40u : 0u;

    condition_1 = int32_t(matrix_accumulator_20) > -1 && !condition_1;

    condition_6 = (matrix_accumulator_15 & 0x4u) != 0;

    matrix_accumulator_39 = 0u - matrix_accumulator_16 + matrix_accumulator_13;

    condition_1 = condition_6 & (condition_0 | condition_1);

    condition_0 = int32_t(matrix_fragment_73) < int32_t(index_word_14) || condition_0;

    index_word_14 = parameters[parameter_trained_weights_low_6];
    matrix_accumulator_15 = parameters[parameter_trained_weights_high_7];

    matrix_accumulator_16 = matrix_accumulator_39 * 0x2u + reused_word_112;

    condition_0 = condition_0 & condition_6;

    if (condition_5) {
        matrix_accumulator_85 = matrix_fragment_73 * 0x4u;
    }

    condition_6 = int32_t(matrix_accumulator_24) != int32_t(0u);

    address_word_29 = uint32_t(int32_t(matrix_accumulator_16) >> 0x1u);

    matrix_accumulator_16 = reused_word_112 + 0x8u;

    if (condition_5) {
        matrix_accumulator_85 = !condition_6 ? matrix_accumulator_85 : 0u;
    }

    condition_6 = int32_t(matrix_accumulator_26) != int32_t(0u);

    matrix_accumulator_16 = matrix_accumulator_39 * 0x2u + matrix_accumulator_16;

    matrix_accumulator_26 = reused_word_112 + 0x18u;

    if (condition_4) {
        matrix_accumulator_84 = matrix_fragment_73 * 0x4u;
    }

    matrix_accumulator_24 = reused_word_112 + 0x10u;

    matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_16) >> 0x1u);

    matrix_accumulator_26 = matrix_accumulator_39 * 0x2u + matrix_accumulator_26;

    if (condition_4) {
        matrix_accumulator_84 = !condition_6 ? matrix_accumulator_84 : 0u;
    }

    condition_6 = int32_t(matrix_accumulator_31) != int32_t(0u);

    if (condition_3) {
        matrix_fragment_78 = matrix_accumulator_20 * 0x4u;
    }

    if (condition_2) {
        matrix_fragment_77 = matrix_fragment_73 * 0x4u;
    }

    matrix_accumulator_24 = matrix_accumulator_39 * 0x2u + matrix_accumulator_24;

    matrix_fragment_74 = reused_word_112 + 0x20u;

    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    matrix_accumulator_26 =
        uint32_t(combine_words(matrix_accumulator_26, 0u) << 0x1u >> 32) + matrix_accumulator_26;

    split_address_words(address_word_28, address_word_29,
                        int64_t(int32_t(address_word_29)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    if (condition_3) {
        matrix_fragment_78 = !condition_6 ? matrix_fragment_78 : 0u;
    }

    if (condition_2) {
        matrix_fragment_77 = !condition_6 ? matrix_fragment_77 : 0u;
    }

    matrix_accumulator_87 = uint32_t(int32_t(matrix_accumulator_24) >> 0x1u);

    matrix_accumulator_31 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x40000ull, memory_ranges,
        error_counts);

    condition_6 = int32_t(matrix_accumulator_37) != int32_t(0u);

    matrix_accumulator_24 = reused_word_112 + 0x30u;

    matrix_fragment_74 = matrix_accumulator_39 * 0x2u + matrix_fragment_74;

    matrix_accumulator_37 = load_aligned_word<CheckBounds>(
        combine_words(address_word_28, address_word_29) + 0x40000ull, memory_ranges, error_counts);

    matrix_accumulator_91 = matrix_accumulator_39 * 0x2u + matrix_fragment_76;

    matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_26) >> 0x1u);

    matrix_fragment_76 = matrix_accumulator_39 * 0x2u + matrix_accumulator_24;

    address_word_28 = reused_word_112 + 0x38u;

    split_address_words(matrix_accumulator_86, matrix_accumulator_87,
                        int64_t(int32_t(matrix_accumulator_87)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_74) >> 0x1u);

    if (condition_4) {
        matrix_fragment_74 = matrix_accumulator_21 & 0xfffffffcu;
    }

    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    if (condition_3) {
        matrix_accumulator_94 = matrix_accumulator_21 & 0xfffffffcu;
    }

    matrix_accumulator_86 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_86, matrix_accumulator_87) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_3) {
        matrix_accumulator_24 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_25;
    }

    if (condition_2) {
        matrix_accumulator_98 = matrix_accumulator_21 & 0xfffffffcu;
    }

    matrix_accumulator_39 = matrix_accumulator_39 * 0x2u + address_word_28;

    if (condition_5) {
        matrix_accumulator_93 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_19;
    }

    if (condition_4) {
        address_word_28 = parameters[parameter_input_skip_low_2];
        address_word_29 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_23 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_23;
    }

    if (condition_1) {
        matrix_accumulator_96 = matrix_accumulator_21 & 0xfffffffcu;
    }

    if (condition_2) {
        matrix_accumulator_25 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_25;
    }

    matrix_accumulator_19 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_1) {
        matrix_accumulator_92 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_27;
    }

    if (condition_0) {
        matrix_accumulator_97 =
            matrix_accumulator_22 * uniform_index_word_14 + matrix_accumulator_27;
    }

    if (condition_5) {
        matrix_accumulator_22 = matrix_accumulator_21 & 0xfffffffcu;

        matrix_accumulator_26 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_27 = parameters[parameter_input_skip_high_3];
    }

    if (condition_3) {
        matrix_accumulator_87 = 0u - matrix_accumulator_94 + matrix_accumulator_13;
    }

    if (condition_2) {
        matrix_accumulator_98 = 0u - matrix_accumulator_98 + matrix_accumulator_13;
    }

    if (condition_1) {
        matrix_accumulator_96 = 0u - matrix_accumulator_96 + matrix_accumulator_13;
    }

    if (condition_5) {
        matrix_accumulator_22 = 0u - matrix_accumulator_22 + matrix_accumulator_13;
    }

    if (condition_4) {
        matrix_accumulator_16 = 0u - matrix_fragment_74 + matrix_accumulator_13;
    }

    if (condition_0) {
        matrix_accumulator_100 = matrix_accumulator_21 & 0xfffffffcu;
    }

    if (condition_3) {
        matrix_accumulator_87 =
            matrix_accumulator_7 * matrix_accumulator_24 + matrix_accumulator_87;
    }

    if (condition_5) {
        matrix_accumulator_94 =
            matrix_accumulator_7 * matrix_accumulator_93 + matrix_accumulator_22;
    }

    if (condition_4) {
        matrix_accumulator_95 =
            matrix_accumulator_7 * matrix_accumulator_23 + matrix_accumulator_16;
    }

    if (condition_2) {
        matrix_accumulator_98 =
            matrix_accumulator_7 * matrix_accumulator_25 + matrix_accumulator_98;
    }

    if (condition_3) {
        matrix_accumulator_16 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_17 = parameters[parameter_input_skip_high_3];
    }

    if (condition_1) {
        matrix_accumulator_93 =
            matrix_accumulator_7 * matrix_accumulator_92 + matrix_accumulator_96;
    }

    if (condition_0) {
        matrix_accumulator_100 = 0u - matrix_accumulator_100 + matrix_accumulator_13;
    }

    if (condition_1) {
        matrix_accumulator_92 = matrix_accumulator_20 * 0x4u;
    }

    if (condition_2) {
        matrix_accumulator_24 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_25 = parameters[parameter_input_skip_high_3];
    }

    if (condition_0) {
        matrix_accumulator_100 =
            matrix_accumulator_7 * matrix_accumulator_97 + matrix_accumulator_100;

        matrix_accumulator_22 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_23 = parameters[parameter_input_skip_high_3];
    }

    if (condition_4) {
        matrix_accumulator_7 = matrix_accumulator_95 + matrix_accumulator_84;
    }

    if (condition_1) {
        matrix_accumulator_20 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_21 = parameters[parameter_input_skip_high_3];
    }

    if (condition_0) {
        matrix_fragment_73 = matrix_fragment_73 * 0x4u;
    }

    matrix_accumulator_91 =
        uint32_t(combine_words(matrix_accumulator_91, 0u) << 0x1u >> 32) + matrix_accumulator_91;

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    if (condition_1) {
        matrix_accumulator_92 = !condition_6 ? matrix_accumulator_92 : 0u;
    }

    if (condition_4) {
        split_address_words(address_word_28, address_word_29,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, address_word_29));
    }

    if (condition_0) {
        matrix_accumulator_7 = !condition_6 ? matrix_fragment_73 : 0u;
    }

    if (condition_5) {
        matrix_accumulator_85 = matrix_accumulator_94 + matrix_accumulator_85;
    }

    matrix_accumulator_13 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_3) {
        matrix_accumulator_87 = matrix_fragment_78 + matrix_accumulator_87;
    }

    matrix_accumulator_91 = uint32_t(int32_t(matrix_accumulator_91) >> 0x1u);

    if (condition_2) {
        matrix_fragment_77 = matrix_accumulator_98 + matrix_fragment_77;
    }

    if (condition_1) {
        matrix_accumulator_93 = matrix_accumulator_92 + matrix_accumulator_93;
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            int64_t(int32_t(matrix_accumulator_85)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));
    }

    if (condition_0) {
        matrix_fragment_75 = matrix_accumulator_100 + matrix_accumulator_7;
    }

    split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                        int64_t(int32_t(matrix_accumulator_91)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    matrix_fragment_73 = 0u;

    matrix_fragment_74 = 0u;

    if (condition_3) {
        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            int64_t(int32_t(matrix_accumulator_87)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, matrix_accumulator_17));
    }

    matrix_accumulator_87 = uint32_t(int32_t(matrix_fragment_76) >> 0x1u);

    matrix_accumulator_7 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_84, matrix_accumulator_85) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_2) {
        split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                            int64_t(int32_t(matrix_fragment_77)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    matrix_fragment_76 = 0;
    matrix_fragment_77 = 0;

    matrix_fragment_78 = 0u;

    matrix_accumulator_39 =
        uint32_t(combine_words(matrix_accumulator_39, 0u) << 0x1u >> 32) + matrix_accumulator_39;

    if (condition_0) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_22, matrix_accumulator_23));
    }

    matrix_fragment_75 = 0u;

    if (condition_5) {
        matrix_fragment_73 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
            error_counts);
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(matrix_accumulator_93)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));
    }

    matrix_accumulator_39 = uint32_t(int32_t(matrix_accumulator_39) >> 0x1u);

    if (condition_4) {
        matrix_fragment_74 = load_aligned_word<CheckBounds>(
            combine_words(address_word_28, address_word_29), memory_ranges, error_counts);
    }

    if (condition_3) {
        matrix_fragment_75 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_16, matrix_accumulator_17), memory_ranges,
            error_counts);
    }

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        int64_t(int32_t(matrix_accumulator_87)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    if (condition_1) {
        matrix_fragment_76 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_20, matrix_accumulator_21), memory_ranges,
            error_counts);
    }

    if (condition_2) {
        matrix_fragment_77 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_24, matrix_accumulator_25), memory_ranges,
            error_counts);
    }

    if (condition_0) {
        matrix_fragment_78 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_22, matrix_accumulator_23), memory_ranges,
            error_counts);
    }

    split_address_words(index_word_14, matrix_accumulator_15,
                        int64_t(int32_t(matrix_accumulator_39)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_14, matrix_accumulator_15));

    matrix_accumulator_87 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_26, matrix_accumulator_27) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_91 = load_aligned_word<CheckBounds>(
        combine_words(index_word_14, matrix_accumulator_15) + 0x40000ull, memory_ranges,
        error_counts);

    address_word_29 = threadIdx.x;

    address_word_28 = threadIdx.z;

    matrix_accumulator_36 = (matrix_accumulator_36 << 0x3u) + (0u - uniform_index_word_11);

    matrix_accumulator_16 = matrix_accumulator_36 + 0x8u;

    matrix_accumulator_17 = matrix_accumulator_36 + 0x7u;

    matrix_accumulator_36 = max(int32_t(0u), int32_t(matrix_accumulator_36));

    condition_0 = int32_t(matrix_accumulator_16) > int32_t(0u);

    matrix_accumulator_17 =
        uint32_t(combine_words(matrix_accumulator_17, 0u) << 0x1du >> 32) + 0x1u;

    matrix_accumulator_15 = uint32_t(uint32_t(matrix_accumulator_36) >> 0x3u);

    matrix_accumulator_17 = condition_0 ? matrix_accumulator_17 : 0u;

    matrix_accumulator_84 = unpack_fp8_bits(matrix_accumulator_30);

    matrix_accumulator_85 = unpack_fp8_bits(matrix_accumulator_33);

    matrix_accumulator_39 = unpack_fp8_bits(matrix_accumulator_34);

    matrix_accumulator_106 = unpack_fp8_bits(matrix_accumulator_35);

    matrix_accumulator_107 = unpack_fp8_bits(matrix_accumulator_88);

    address_word_29 = address_word_29 | matrix_accumulator_38;

    matrix_accumulator_38 = unpack_fp8_bits(matrix_accumulator_32);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_accumulator_89);

    index_word_14 = unpack_fp8_bits(matrix_accumulator_90);

    index_word_110 = min(int32_t(matrix_accumulator_17), int32_t(uniform_index_word_4));

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

    address_word_28 = address_word_28 | address_word_29;

    condition_0 = int32_t(matrix_accumulator_15) >= int32_t(index_word_110);

    matrix_accumulator_104 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

    matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_12);

    condition_0 = uint32_t(address_word_28) != 0u || condition_0;

    matrix_accumulator_20 = unpack_fp8_bits(matrix_accumulator_12 >> 16);

    address_word_28 = unpack_fp8_bits(matrix_accumulator_11);

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_11 >> 16);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_10);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_2);

    matrix_accumulator_15 = unpack_fp8_bits(matrix_accumulator_8);

    matrix_accumulator_94 = unpack_fp8_bits(matrix_accumulator_6);

    matrix_accumulator_95 = unpack_fp8_bits(matrix_accumulator_3);

    matrix_accumulator_12 = unpack_fp8_bits(matrix_accumulator_4);

    matrix_accumulator_11 = unpack_fp8_bits(matrix_accumulator_5);

    uniform_address_word_10 = (uniform_index_word_8 << 0x3u) + (0u - uniform_address_word_10);

    matrix_accumulator_96 = unpack_fp8_bits(matrix_accumulator_6 >> 16);

    matrix_accumulator_10 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

    uniform_address_word_12 = uniform_address_word_10 + 0x8u;

    uniform_index_word_11 = uniform_address_word_10 + 0x7u;

    matrix_accumulator_97 = unpack_fp8_bits(matrix_accumulator_3 >> 16);

    matrix_accumulator_6 = unpack_fp8_bits(matrix_accumulator_4 >> 16);

    address_word_29 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_fragment_79);

    matrix_accumulator_102 = unpack_fp8_bits(matrix_fragment_80);

    matrix_accumulator_103 = unpack_fp8_bits(matrix_fragment_81);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_fragment_82);

    matrix_accumulator_25 = unpack_fp8_bits(matrix_fragment_83);

    matrix_accumulator_5 = unpack_fp8_bits(matrix_accumulator_5 >> 16);

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x1du >> 32) + 0x1u;

    uniform_condition_0 = int32_t(uniform_address_word_12) > int32_t(0u);

    matrix_accumulator_30 = unpack_fp8_bits(matrix_accumulator_30 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

    matrix_accumulator_105 = unpack_fp8_bits(matrix_accumulator_88 >> 16);

    matrix_accumulator_89 = unpack_fp8_bits(matrix_accumulator_89 >> 16);

    uniform_index_word_11 = uniform_condition_0 ? uniform_index_word_11 : 0u;

    matrix_accumulator_90 = unpack_fp8_bits(matrix_accumulator_90 >> 16);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_fragment_79 >> 16);

    matrix_accumulator_4 = unpack_fp8_bits(matrix_fragment_72);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_92 = unpack_fp8_bits(matrix_fragment_80 >> 16);

    matrix_accumulator_93 = unpack_fp8_bits(matrix_fragment_81 >> 16);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_fragment_82 >> 16);

    matrix_accumulator_88 = unpack_fp8_bits(matrix_fragment_72 >> 16);

    matrix_fragment_83 = unpack_fp8_bits(matrix_fragment_83 >> 16);

    index_word_111 = max(int32_t(0u), int32_t(uniform_address_word_10));

    index_word_113 = 0u;

    index_word_111 = uint32_t(uint32_t(index_word_111) >> 0x3u);

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(index_word_14),
                            select_half_operand<2, false, false>(matrix_accumulator_37));

    index_word_14 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(index_word_14),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    index_word_14 = unpack_fp8_bits(matrix_accumulator_9);

    matrix_accumulator_9 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

    address_word_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_28),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    address_word_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_29),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_89 = unpack_fp8_bits(index_word_0 >> 16);

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    index_word_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(index_word_14),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_11),
                            select_half_operand<2, false, false>(matrix_accumulator_13));

    matrix_accumulator_11 = unpack_fp8_bits(matrix_accumulator_2 >> 16);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_accumulator_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_9),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_10),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_11),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_6),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_73);

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                            select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_5 = unpack_fp8_bits(index_word_0);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_74);

    matrix_accumulator_98 = unpack_fp8_bits(matrix_fragment_75);

    matrix_accumulator_99 = unpack_fp8_bits(matrix_fragment_76);

    matrix_accumulator_86 = unpack_fp8_bits(matrix_fragment_77);

    matrix_fragment_79 = unpack_fp8_bits(matrix_fragment_78);

    matrix_accumulator_108 = unpack_fp8_bits(matrix_fragment_73 >> 16);

    matrix_accumulator_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_4),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_109 = unpack_fp8_bits(matrix_fragment_74 >> 16);

    matrix_accumulator_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_100 = unpack_fp8_bits(matrix_fragment_75 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_101 = unpack_fp8_bits(matrix_fragment_76 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_90 = unpack_fp8_bits(matrix_fragment_77 >> 16);

    matrix_accumulator_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_78 = unpack_fp8_bits(matrix_fragment_78 >> 16);

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_79 = uniform_index_word_11;

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    index_word_0 = min(int32_t(matrix_fragment_79), int32_t(uniform_index_word_5));

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_91));
    do {
        index_word_118 = uint32_t(uint32_t(index_word_113) >> 0x6u);

        index_word_115 = 0u;

        warp_lane_116 = 0x400u;

        matrix_fragment_72 = permute_signed(index_word_118, 0u, 0x9910u);

        warp_lane_117 = threadIdx.x & 31u;

        __syncwarp(__activemask());

        matrix_fragment_72 = matrix_fragment_72 * 0xabu;

        condition_1 = uint32_t(index_word_113) > uint32_t(0x1bfu);

        matrix_fragment_72 = matrix_fragment_72 & 0xffffu;

        matrix_fragment_72 = uint32_t(uint32_t(matrix_fragment_72) >> 0x9u);

        matrix_fragment_72 = permute_signed(matrix_fragment_72, 0u, 0x9910u);

        matrix_fragment_72 = matrix_fragment_72 * 0x3u;

        matrix_fragment_72 = 0u - matrix_fragment_72;

        matrix_fragment_73 = permute_signed(matrix_fragment_72, 0u, 0x7710u);

        index_word_118 = index_word_118 + matrix_fragment_73;

        matrix_fragment_73 = (index_word_115 << 0x18u) + warp_lane_116;

        index_word_114 = index_word_118 & 0xffu;

        index_word_114 = index_word_114 * 0x1000u + matrix_fragment_73;

        tile_word_119 = warp_lane_117 * 0x10u + index_word_114;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119, matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
            matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0x400u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0x200u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(address_word_28, address_word_29, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, address_word_28,
                                 address_word_29);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_88,
                                 matrix_accumulator_89);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0x800u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_100,
                                 matrix_accumulator_101);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0xa00u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(index_word_14, matrix_accumulator_15, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, index_word_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0xc00u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(address_word_28, address_word_29, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, address_word_28,
                                 address_word_29);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_88,
                                 matrix_accumulator_89);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0x600u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_100,
                                 matrix_accumulator_101);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_119 + 0xe00u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_90,
                                 matrix_accumulator_91);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(index_word_14, matrix_accumulator_15, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, index_word_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_90,
                                 matrix_accumulator_91);
        if (!condition_1) {
            matrix_fragment_43 = blockIdx.z;

            matrix_fragment_74 = index_word_113 + 0x40u;

            matrix_fragment_40 = parameters[parameter_trained_weights_low_6];
            matrix_fragment_41 = parameters[parameter_trained_weights_high_7];

            matrix_fragment_43 = matrix_fragment_43 * 0x200u + matrix_fragment_74;

            matrix_fragment_74 = uint32_t(uint32_t(matrix_fragment_74) >> 0x6u);

            matrix_fragment_43 = matrix_fragment_43 * 0x10u + reused_word_112;

            matrix_fragment_75 = permute_signed(matrix_fragment_74, 0u, 0x9910u);

            matrix_fragment_43 = matrix_fragment_43 * 0x8u;

            matrix_fragment_75 = matrix_fragment_75 * 0xabu;

            split_address_words(matrix_fragment_40, matrix_fragment_41,
                                int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_40, matrix_fragment_41));

            matrix_fragment_75 = matrix_fragment_75 & 0xffffu;

            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(warp_lane_117)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_40, matrix_fragment_41));

            matrix_fragment_75 = uint32_t(uint32_t(matrix_fragment_75) >> 0x9u);

            matrix_fragment_75 = permute_signed(matrix_fragment_75, 0u, 0x9910u);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73), matrix_fragment_40,
                matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
                error_counts);

            matrix_fragment_75 = matrix_fragment_75 * 0x3u;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x200ull,
                matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                memory_ranges, error_counts);

            matrix_fragment_75 = 0u - matrix_fragment_75;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x400ull,
                matrix_fragment_48, matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x600ull,
                matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4000ull,
                matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4200ull,
                matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4400ull,
                matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4600ull,
                matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                memory_ranges, error_counts);

            warp_lane_116 = warp_lane_116 + 0x3000u;

            matrix_fragment_73 = permute_signed(matrix_fragment_75, 0u, 0x7710u);

            matrix_fragment_75 = (index_word_115 << 0x18u) + warp_lane_116;

            matrix_fragment_74 = matrix_fragment_74 + matrix_fragment_73;

            matrix_fragment_74 = matrix_fragment_74 & 0xffu;

            matrix_fragment_75 = matrix_fragment_74 * 0x8u + matrix_fragment_75;

#if DLSS5_ENABLE_ASYNC_COPIES
            {
                const uint32_t operand_a = matrix_fragment_75;
                async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
                split_address_words(matrix_fragment_72, matrix_fragment_73,
                                    async_tokens[operand_a >> 3 & 1]);
            }
#else
            matrix_fragment_72 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_fragment_75,
                                            async_tokens[(((matrix_fragment_75) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
            if (!condition_1) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_1 =
                        wait_copy_barrier(shared_tile, matrix_fragment_75,
                                          async_tokens[(((matrix_fragment_75) >> 3) & 1)]);
#else
                    __syncthreads();
                    condition_1 = true;
#endif
                } while (!condition_1);
            }
        }
        matrix_fragment_72 = parameters[parameter_tensor_height_8];

        condition_1 = uint32_t(index_word_113) > uint32_t(0x13fu);

        matrix_fragment_83 = index_word_118 & 0xffu;

        matrix_fragment_78 = parameters[parameter_tensor_width_9];

        matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_72) >> 0x1fu);

        matrix_fragment_75 =
            uint32_t(combine_words(matrix_fragment_75, 0u) << 0x2u >> 32) + matrix_fragment_72;

        matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_78) >> 0x1fu);

        matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

        matrix_fragment_73 =
            uint32_t(combine_words(matrix_fragment_73, 0u) << 0x2u >> 32) + matrix_fragment_78;

        matrix_fragment_76 = uint32_t(int32_t(matrix_fragment_73) >> 0x2u);
        if (!condition_1) {
            matrix_fragment_72 = index_word_113 + 0xc0u;

            condition_1 = (matrix_fragment_72 & 0xc0u) != 0;
            if (!condition_1) {
                if (!condition_0) {
                    uniform_address_word_10 = blockIdx.y;

                    uniform_index_word_11 = parameters[parameter_optional_control_17];

                    uniform_address_word_10 =
                        (uniform_address_word_10 << 0x3u) + (0u - uniform_index_word_11);

                    uniform_address_word_10 = max(int32_t(0u), int32_t(uniform_address_word_10));

                    uniform_address_word_10 = uint32_t(uint32_t(uniform_address_word_10) >> 0x3u);

                    matrix_fragment_75 = uniform_address_word_10;
                    do {
                        condition_1 = int32_t(index_word_111) >= int32_t(index_word_0);
                        if (!condition_1) {
                            matrix_fragment_72 = blockIdx.z;

                            matrix_fragment_79 = index_word_111;

                            matrix_fragment_72 = matrix_fragment_72 * 0x200u + index_word_113;

                            matrix_fragment_72 = matrix_fragment_72 + 0xc0u;

                            matrix_fragment_72 = uint32_t(uint32_t(matrix_fragment_72) >> 0x8u);

                            matrix_fragment_72 = matrix_fragment_75 * 0x2u + matrix_fragment_72;

                            matrix_fragment_74 = matrix_fragment_72 * uniform_index_word_5;
                            do {
                                matrix_fragment_72 = parameters[parameter_wait_low_14];
                                matrix_fragment_73 = parameters[parameter_wait_high_15];

                                matrix_fragment_81 = matrix_fragment_74 + matrix_fragment_79;

                                split_address_words(
                                    matrix_fragment_72, matrix_fragment_73,
                                    int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_72, matrix_fragment_73));

                                matrix_fragment_80 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_72, matrix_fragment_73),
                                    memory_ranges, error_counts);

                                matrix_fragment_79 = matrix_fragment_79 + 0x1u;

                                condition_2 =
                                    uint32_t(matrix_fragment_79) != uint32_t(index_word_0);

                                condition_1 = int32_t(matrix_fragment_80) > -1;
                                if (!condition_1) {
                                    do {
                                        __nanosleep(0x40u);

                                        matrix_fragment_80 = pipeline_read<CheckBounds>(
                                            combine_words(matrix_fragment_72, matrix_fragment_73),
                                            memory_ranges, error_counts);

                                        condition_1 = int32_t(matrix_fragment_80) > -1;
                                    } while (!condition_1);
                                }
                            } while (condition_2);
                        }
                        matrix_fragment_75 = matrix_fragment_75 + 0x1u;

                        condition_1 = uint32_t(matrix_fragment_75) != uint32_t(index_word_110);
                    } while (condition_1);
                }
                __syncwarp(__activemask());

                __syncthreads();
            }
            matrix_fragment_80 = blockIdx.y;

            warp_lane_116 = parameters[parameter_tensor_height_8];
            warp_lane_117 = parameters[parameter_tensor_width_9];

            split_address_words(matrix_fragment_74, matrix_fragment_75, 0ull);

            matrix_fragment_79 = threadIdx.y;

            index_word_115 = warp_lane_116 & 0xfffffffcu;

            warp_lane_117 = warp_lane_117 & 0xfffffffcu;

            condition_1 = uint32_t(index_word_115) != uint32_t(0x4u);

            matrix_fragment_80 = matrix_fragment_80 * 0x2u;

            condition_4 = uint32_t(warp_lane_117) != uint32_t(0x4u);

            warp_lane_116 =
                uint32_t(combine_words(matrix_fragment_79, 0u) << 0x1eu >> 32) + matrix_fragment_80;

            matrix_fragment_72 = uint32_t(uint32_t(matrix_fragment_79) >> 0x1u);

            matrix_fragment_82 = matrix_fragment_79 & 0x1u;

            matrix_fragment_72 = matrix_fragment_72 & 0x1u;

            condition_5 = int32_t(warp_lane_116) >= int32_t(matrix_fragment_77) && condition_1;

            condition_2 = int32_t(warp_lane_116) >= int32_t(matrix_fragment_77);

            matrix_fragment_81 = matrix_fragment_72 + uniform_block_index_13;

            condition_3 = uint32_t(warp_lane_117) == uint32_t(0x4u) || condition_5;

            condition_2 = uint32_t(index_word_115) == uint32_t(0x4u) || !condition_2;

            index_word_118 = condition_5 ? matrix_fragment_81 : 0u;

            condition_3 = int32_t(matrix_fragment_81) < int32_t(matrix_fragment_76) || condition_3;

            condition_2 = condition_3 & condition_2;

            if (condition_2) {
                matrix_fragment_72 = parameters[parameter_input_low_0];
                matrix_fragment_73 = parameters[parameter_input_high_1];

                matrix_fragment_74 = blockIdx.z;

                condition_3 = uint32_t(index_word_115) != uint32_t(0x4u);

                matrix_fragment_75 = !condition_4 ? index_word_118 : matrix_fragment_81;

                if (condition_3) {
                    matrix_fragment_75 = warp_lane_116 * matrix_fragment_76 + matrix_fragment_75;
                }

                matrix_fragment_74 = matrix_fragment_74 * 0x200u + index_word_113;

                tile_word_119 = matrix_fragment_74 + 0xc0u;

                matrix_fragment_74 =
                    uint32_t(combine_words(tile_word_119, 0u) << 0x1bu >> 32) + matrix_fragment_82;

                matrix_fragment_74 = matrix_fragment_75 * 0x10u + matrix_fragment_74;

                matrix_fragment_74 = matrix_fragment_74 * 0x80u;

                matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_74) >> 0x1fu);
            }
            uniform_index_word_11 = 0u;

            uniform_address_word_10 = 0x400u;

            uint64_t byte_address_6;
            if (condition_2) {
                byte_address_6 =
                    scaled_word_address(combine_words(matrix_fragment_72, matrix_fragment_73),
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
                warp_lane_116 = uint32_t(byte_address_6);
            }

            uniform_address_word_10 = uniform_address_word_10 + 0x3000u;

            if (condition_2) {
                matrix_fragment_75 = uint32_t(byte_address_6 >> 32);
            }

            split_address_words(matrix_fragment_72, matrix_fragment_73, 0ull);

            if (condition_2) {
                matrix_fragment_72 = warp_lane_116;

                matrix_fragment_73 = matrix_fragment_75;
            }

            uniform_address_word_10 = (uniform_index_word_11 << 0x18u) + uniform_address_word_10;

            matrix_fragment_83 = (matrix_fragment_83 << 0x3u) + uniform_address_word_10;
            if (!condition_2) {
                warp_lane_116 = threadIdx.x & 31u;

                matrix_fragment_72 = matrix_fragment_79 * 0x200u;

                uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                        select_half_operand<2, false, false>(0u));

                matrix_fragment_73 = matrix_fragment_72 & 0x7fc00u;

                matrix_fragment_72 = uniform_address_word_10;

                matrix_fragment_75 = uniform_address_word_10;

                matrix_fragment_73 = matrix_fragment_82 * 0x200u + matrix_fragment_73;

                matrix_fragment_72 =
                    permute_signed(matrix_fragment_72, matrix_fragment_75, 0x5410u);

                matrix_fragment_73 = index_word_114 + matrix_fragment_73;

                matrix_fragment_74 = matrix_fragment_72;

                matrix_fragment_75 = matrix_fragment_72;

                warp_lane_116 = warp_lane_116 * 0x10u + matrix_fragment_73;

                matrix_fragment_73 = matrix_fragment_72;

                store_four_shared_words<CheckBounds>(shared_tile, warp_lane_116, matrix_fragment_72,
                                                     matrix_fragment_73, matrix_fragment_74,
                                                     matrix_fragment_75, shared_bytes,
                                                     memory_ranges, error_counts);
            } else {
                uniform_address_word_10 = 0xffffffffu;

                condition_2 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_2) {
                    uniform_address_word_10 = broadcast_active_lane(matrix_fragment_79);

                    uniform_address_word_12 = broadcast_active_lane(index_word_114);

                    uniform_address_word_20 = broadcast_active_lane(matrix_fragment_72);

                    uniform_address_word_21 = broadcast_active_lane(matrix_fragment_73);

                    uniform_index_word_11 = broadcast_active_lane(matrix_fragment_83);

                    matrix_fragment_72 = 0x200u;

                    uniform_address_word_10 =
                        (uniform_address_word_10 << 0x9u) + uniform_address_word_12;

                    uniform_address_word_12 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_10,
                        combine_words(uniform_address_word_20, uniform_address_word_21),
                        uniform_address_word_12 * 16u, matrix_fragment_83, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_10,
                        combine_words(uniform_address_word_20, uniform_address_word_21),
                        uniform_address_word_12 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            matrix_fragment_73 = matrix_fragment_79 + 0x4u;

            condition_5 = uint32_t(index_word_115) != uint32_t(0x4u);

            matrix_fragment_80 =
                uint32_t(combine_words(matrix_fragment_73, 0u) << 0x1eu >> 32) + matrix_fragment_80;

            split_address_words(matrix_fragment_72, matrix_fragment_73, 0ull);

            condition_1 = int32_t(matrix_fragment_80) >= int32_t(matrix_fragment_77) && condition_1;

            condition_2 = int32_t(matrix_fragment_80) < int32_t(matrix_fragment_77) || !condition_5;

            condition_3 = uint32_t(warp_lane_117) == uint32_t(0x4u) || condition_1;

            matrix_fragment_74 = condition_1 ? matrix_fragment_81 : 0u;

            condition_3 = int32_t(matrix_fragment_81) < int32_t(matrix_fragment_76) || condition_3;

            matrix_fragment_74 = !condition_4 ? matrix_fragment_74 : matrix_fragment_81;

            condition_2 = condition_3 & condition_2;
            if (condition_2) {
                matrix_fragment_72 = blockIdx.z;

                matrix_fragment_73 = matrix_fragment_74;

                if (condition_5) {
                    matrix_fragment_73 =
                        matrix_fragment_80 * matrix_fragment_76 + matrix_fragment_73;
                }

                matrix_fragment_72 = matrix_fragment_72 * 0x200u + index_word_113;

                matrix_fragment_75 = matrix_fragment_72 + 0xc0u;

                matrix_fragment_72 =
                    uint32_t(combine_words(matrix_fragment_75, 0u) << 0x1bu >> 32) +
                    matrix_fragment_82;

                matrix_fragment_72 = matrix_fragment_73 * 0x10u + matrix_fragment_72;

                matrix_fragment_72 = matrix_fragment_72 * 0x80u;

                matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_72) >> 0x1fu);
            }
            if (!condition_2) {
                matrix_fragment_75 = threadIdx.x & 31u;

                matrix_fragment_79 = matrix_fragment_79 * 0x200u;

                uniform_address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                        select_half_operand<2, false, false>(0u));

                matrix_fragment_72 = matrix_fragment_79 + 0x800u;

                matrix_fragment_73 = uniform_address_word_10;

                matrix_fragment_72 = matrix_fragment_72 & 0xff800u;

                matrix_fragment_79 = matrix_fragment_72 | 0x400u & matrix_fragment_79;

                matrix_fragment_72 = uniform_address_word_10;

                matrix_fragment_79 = matrix_fragment_82 * 0x200u + matrix_fragment_79;

                matrix_fragment_72 =
                    permute_signed(matrix_fragment_72, matrix_fragment_73, 0x5410u);

                index_word_114 = index_word_114 + matrix_fragment_79;

                matrix_fragment_73 = matrix_fragment_72;

                matrix_fragment_74 = matrix_fragment_72;

                matrix_fragment_79 = matrix_fragment_75 * 0x10u + index_word_114;

                matrix_fragment_75 = matrix_fragment_72;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_79,
                                                     matrix_fragment_72, matrix_fragment_73,
                                                     matrix_fragment_74, matrix_fragment_75,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (condition_2) {
                    matrix_fragment_80 = parameters[parameter_input_low_0];
                    matrix_fragment_81 = parameters[parameter_input_high_1];
                }

                uniform_address_word_10 = 0xffffffffu;

                split_address_words(matrix_fragment_74, matrix_fragment_75, 0ull);

                if (condition_2) {
                    const uint64_t byte_address_7 =
                        scaled_word_address(combine_words(matrix_fragment_80, matrix_fragment_81),
                                            combine_words(matrix_fragment_72, matrix_fragment_73));
                    index_word_115 = uint32_t(byte_address_7);

                    matrix_fragment_75 = uint32_t(byte_address_7 >> 32);

                    matrix_fragment_74 = index_word_115;
                }

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    matrix_fragment_73 = matrix_fragment_79 * 0x200u;

                    uniform_address_word_12 = broadcast_active_lane(index_word_114);

                    uniform_address_word_20 = broadcast_active_lane(matrix_fragment_74);

                    matrix_fragment_72 = matrix_fragment_73 + 0x800u;

                    uniform_address_word_21 = broadcast_active_lane(matrix_fragment_75);

                    uniform_index_word_11 = broadcast_active_lane(matrix_fragment_83);

                    matrix_fragment_72 = matrix_fragment_72 & 0xff800u;

                    matrix_fragment_73 = matrix_fragment_72 | 0x400u & matrix_fragment_73;

                    matrix_fragment_72 = 0x200u;

                    matrix_fragment_73 = matrix_fragment_82 * 0x200u + matrix_fragment_73;

                    uniform_address_word_10 = broadcast_active_lane(matrix_fragment_73);

                    uniform_address_word_10 = uniform_address_word_12 + uniform_address_word_10;

                    uniform_address_word_12 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_10,
                        combine_words(uniform_address_word_20, uniform_address_word_21),
                        uniform_address_word_12 * 16u, matrix_fragment_83, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_10,
                        combine_words(uniform_address_word_20, uniform_address_word_21),
                        uniform_address_word_12 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_1 = uint32_t(index_word_113) >= uint32_t(0x1c0u);

        index_word_113 = index_word_113 + 0x40u;
    } while (!condition_1);
    index_word_0 = blockIdx.y;

    matrix_fragment_50 = parameters[parameter_output_low_4];
    matrix_fragment_51 = parameters[parameter_output_high_5];

    condition_0 = int32_t(matrix_fragment_76) <= int32_t(uniform_block_index_13);

    matrix_fragment_53 = reused_word_112 * 0x4u;

    matrix_fragment_54 = threadIdx.x;

    matrix_fragment_57 = threadIdx.y;

    matrix_fragment_56 = threadIdx.z;

    index_word_0 = index_word_0 * 0x2u;

    matrix_fragment_40 = index_word_0 * matrix_fragment_76 + uniform_block_index_13;

    condition_1 = int32_t(index_word_0) >= int32_t(matrix_fragment_77) || condition_0;

    matrix_fragment_52 = index_word_0 * matrix_fragment_76 + matrix_fragment_76;

    matrix_fragment_49 = matrix_fragment_40 * 0x800u + matrix_fragment_53;

    split_address_words(matrix_fragment_48, matrix_fragment_49,
                        int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_50, matrix_fragment_51));
    if (!condition_1) {
        matrix_fragment_55 = threadIdx.x & 31u;

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_5),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_accumulator_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(address_word_29),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                          select_half_operand<2, false, false>(matrix_accumulator_5));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_4),
                          select_half_operand<2, false, false>(matrix_accumulator_23));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(address_word_28),
                          select_half_operand<2, false, false>(matrix_accumulator_11));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                          select_half_operand<2, false, false>(matrix_accumulator_17));

        split_address_words(matrix_accumulator_4, matrix_accumulator_5,
                            int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_4, matrix_accumulator_5), matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_4, matrix_accumulator_5) + 0x200ull,
            matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
            memory_ranges, error_counts);
    }
    uniform_address_word_10 = uniform_block_index_13 + 0x1u;

    matrix_accumulator_10 = index_word_0 + 0x1u;

    matrix_fragment_52 = matrix_fragment_52 + uniform_block_index_13;

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    condition_1 = int32_t(matrix_fragment_76) <= int32_t(uniform_address_word_10);

    matrix_accumulator_5 = matrix_fragment_52 * 0x800u + matrix_fragment_53;

    condition_0 = int32_t(matrix_accumulator_10) >= int32_t(matrix_fragment_77) || condition_0;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(0u));

    condition_2 = int32_t(matrix_accumulator_10) >= int32_t(matrix_fragment_77) || condition_1;

    condition_1 = int32_t(index_word_0) >= int32_t(matrix_fragment_77) || condition_1;

    matrix_accumulator_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_4, matrix_accumulator_5,
                        int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_50, matrix_fragment_51));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    condition_3 = (matrix_fragment_54 | matrix_fragment_56 | matrix_fragment_57) != 0;

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(0u));

    index_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_14),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        matrix_accumulator_27 = threadIdx.x & 31u;

        matrix_accumulator_11 =
            permute_signed(matrix_accumulator_11, matrix_accumulator_20, 0x5410u);

        matrix_accumulator_10 =
            permute_signed(matrix_accumulator_10, matrix_accumulator_23, 0x5410u);

        matrix_accumulator_9 =
            permute_signed(matrix_accumulator_39, matrix_accumulator_16, 0x5410u);

        matrix_accumulator_8 =
            permute_signed(matrix_accumulator_38, matrix_accumulator_17, 0x5410u);

        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_2, matrix_accumulator_109, 0x5410u);

        matrix_accumulator_22 =
            permute_signed(matrix_accumulator_22, matrix_accumulator_3, 0x5410u);

        matrix_accumulator_21 =
            permute_signed(matrix_accumulator_21, matrix_accumulator_26, 0x5410u);

        matrix_accumulator_20 = permute_signed(index_word_14, matrix_accumulator_15, 0x5410u);

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_48, matrix_fragment_49) + 0x2000ull, matrix_accumulator_8,
            matrix_accumulator_9, matrix_accumulator_10, matrix_accumulator_11, memory_ranges,
            error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_48, matrix_fragment_49) + 0x2200ull,
            matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_22,
            matrix_accumulator_23, memory_ranges, error_counts);
    }
    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    index_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        address_word_29 = threadIdx.x & 31u;

        matrix_accumulator_9 = permute_signed(matrix_accumulator_9, matrix_accumulator_2, 0x5410u);

        matrix_accumulator_8 = permute_signed(matrix_accumulator_8, matrix_accumulator_3, 0x5410u);

        matrix_accumulator_11 = permute_signed(matrix_accumulator_11, index_word_14, 0x5410u);

        matrix_accumulator_10 =
            permute_signed(matrix_accumulator_10, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_23, matrix_accumulator_26, 0x5410u);

        matrix_accumulator_22 =
            permute_signed(matrix_accumulator_22, matrix_accumulator_27, 0x5410u);

        matrix_accumulator_21 =
            permute_signed(matrix_accumulator_21, matrix_accumulator_16, 0x5410u);

        matrix_accumulator_20 =
            permute_signed(matrix_accumulator_20, matrix_accumulator_17, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_29)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_4, matrix_accumulator_5));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_accumulator_8,
            matrix_accumulator_9, matrix_accumulator_10, matrix_accumulator_11, memory_ranges,
            error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull,
            matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_22,
            matrix_accumulator_23, memory_ranges, error_counts);
    }
    index_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_30 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_6),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        matrix_accumulator_19 = threadIdx.x & 31u;

        matrix_accumulator_8 = permute_signed(matrix_accumulator_8, matrix_accumulator_31, 0x5410u);

        matrix_accumulator_9 = permute_signed(matrix_accumulator_9, matrix_accumulator_30, 0x5410u);

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_13 = permute_signed(index_word_14, matrix_accumulator_13, 0x5410u);

        matrix_accumulator_7 = permute_signed(matrix_accumulator_25, matrix_accumulator_2, 0x5410u);

        matrix_accumulator_6 = permute_signed(matrix_accumulator_24, matrix_accumulator_3, 0x5410u);

        matrix_accumulator_11 =
            permute_signed(matrix_accumulator_11, matrix_accumulator_16, 0x5410u);

        matrix_accumulator_10 =
            permute_signed(matrix_accumulator_10, matrix_accumulator_17, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_19)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_4, matrix_accumulator_5));

        matrix_accumulator_4 = matrix_accumulator_9;

        matrix_accumulator_5 = matrix_accumulator_8;

        matrix_accumulator_8 = matrix_accumulator_13;

        matrix_accumulator_9 = matrix_accumulator_12;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x2000ull,
            matrix_accumulator_4, matrix_accumulator_5, matrix_accumulator_6, matrix_accumulator_7,
            memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x2200ull,
            matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_10,
            matrix_accumulator_11, memory_ranges, error_counts);
    }
    __syncwarp(__activemask());

    __syncthreads();

    if (condition_3) {
        return;
    }

    uniform_address_word_10 = blockIdx.x;

    matrix_fragment_78 = matrix_fragment_78 + 0xffffffffu;

    matrix_accumulator_5 = uint32_t(int32_t(matrix_fragment_78) >> 0x1fu);

    matrix_accumulator_2 = parameters[parameter_publication_optional_low_12];
    matrix_accumulator_3 = parameters[parameter_publication_optional_high_13];

    matrix_accumulator_5 =
        uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x3u >> 32) + matrix_fragment_78;

    matrix_accumulator_5 = uint32_t(int32_t(matrix_accumulator_5) >> 0x3u);

    matrix_accumulator_5 = matrix_accumulator_5 + 0x1u;

    matrix_accumulator_5 = matrix_accumulator_5 * index_word_0 + uniform_address_word_10;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(248) void phoenix_global_attention_split_heads_512_feed_forward_projection_input_view(
    KernelArguments parameters) {
    run_global_attention_split_heads_512_feed_forward_projection_input_view<false>(
        parameters.words, nullptr, nullptr);
}
