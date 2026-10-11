// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/projection_output_view_wait.cu
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
    parameter_wait_low_14 = 14,
    parameter_wait_high_15 = 15,
    parameter_optional_control_16 = 16,
    parameter_scalar_control_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[18];
};
static_assert(sizeof(KernelArguments) == 72, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_projection_output_view_wait(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t matrix_accumulator_4 = 0;
    uint32_t matrix_accumulator_5 = 0;
    uint32_t matrix_accumulator_6 = 0;
    uint32_t matrix_accumulator_7 = 0;
    uint32_t matrix_accumulator_8 = 0;
    uint32_t address_word_9 = 0;
    uint32_t matrix_accumulator_10 = 0;
    uint32_t address_word_11 = 0;
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t matrix_accumulator_14 = 0;
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
    uint32_t reused_word_110 = 0;
    uint32_t index_word_111 = 0;
    uint32_t address_word_112 = 0;
    uint32_t thread_row_113 = 0;
    uint32_t index_word_114 = 0;
    uint32_t index_word_115 = 0;
    uint32_t index_word_116 = 0;
    uint32_t index_word_117 = 0;
    uint32_t index_word_118 = 0;
    uint32_t warp_lane_119 = 0;
    uint32_t index_word_120 = 0;
    uint32_t index_word_121 = 0;
    uint32_t tile_word_122 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_reused_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
    uint32_t uniform_block_index_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_address_word_12 = 0;
    uint32_t uniform_address_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_address_word_16 = 0;
    uint32_t uniform_address_word_17 = 0;
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

    matrix_accumulator_13 = parameters[parameter_tensor_width_9];

    matrix_accumulator_8 = blockIdx.x;

    uniform_address_word_7 = parameters[parameter_scalar_control_17];

    matrix_accumulator_6 = threadIdx.x;

    uniform_reused_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    thread_row_113 = threadIdx.y;

    index_word_0 = matrix_accumulator_13 + 0xffffffffu;

    address_word_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    address_word_3 = uint32_t(combine_words(address_word_3, 0u) << 0x3u >> 32) + index_word_0;

    matrix_accumulator_5 = uint32_t(int32_t(address_word_3) >> 0x3u);

    matrix_accumulator_6 = matrix_accumulator_6 | thread_row_113;
    condition_1 = matrix_accumulator_6 != 0;

    index_word_0 = matrix_accumulator_5 + 0x1u;

    matrix_accumulator_7 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    address_word_11 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    matrix_accumulator_4 = __float_as_uint(__int2float_ru(matrix_accumulator_7));

    // All three decisions use the same vote over the current active lanes.
    uniform_condition_0 = vote_warp_predicate<true>(condition_1);
    uniform_condition_1 = uniform_condition_0;
    uniform_condition_2 = uniform_condition_0;

    matrix_accumulator_4 = reciprocal_bits(matrix_accumulator_4);

    address_word_2 = matrix_accumulator_4 + 0xffffffeu;

    address_word_3 = __float2uint_rz(__uint_as_float(address_word_2));

    address_word_2 = 0u;

    matrix_accumulator_10 = 0u - address_word_3;

    address_word_9 = matrix_accumulator_10 * matrix_accumulator_7;

    matrix_accumulator_10 =
        int32_t(matrix_accumulator_8) < 0 ? 0u - matrix_accumulator_8 : matrix_accumulator_8;

    address_word_3 = uint32_t(uint64_t(address_word_3) * address_word_9 +
                                  combine_words(address_word_2, address_word_3) >>
                              32);

    address_word_9 = 0u - address_word_11;

    address_word_2 = matrix_accumulator_10;

    address_word_11 = threadIdx.x & 31u;

    address_word_112 = uint32_t(uint64_t(address_word_3) * address_word_2 >> 32);

    if (!condition_1) {
        address_word_3 = blockDim.x;
    }

    address_word_2 = address_word_112 * address_word_9 + address_word_2;

    if (!condition_1) {
        address_word_9 = 0u;
    }

    condition_2 = uint32_t(matrix_accumulator_7) > uint32_t(address_word_2);

    if (!condition_1) {
        matrix_accumulator_4 = blockDim.y;
    }

    if (!condition_2) {
        address_word_2 = address_word_2 + (0u - matrix_accumulator_7);

        address_word_112 = address_word_112 + 0x1u;
    }

    condition_0 = uint32_t(address_word_2) >= uint32_t(matrix_accumulator_7);

    address_word_2 = index_word_0 ^ matrix_accumulator_8;

    matrix_accumulator_7 = blockIdx.z;

    condition_2 = int32_t(address_word_2) >= int32_t(0u);

    if (!condition_1) {
        matrix_accumulator_10 = address_word_3 * matrix_accumulator_4;

        matrix_accumulator_4 = 0x400u;
    }

    address_word_2 = parameters[parameter_trained_weights_low_6];
    address_word_3 = parameters[parameter_trained_weights_high_7];

    uniform_address_word_4 =
        broadcast_warp_word(matrix_accumulator_10, !condition_1, uniform_address_word_4);

    if (condition_0) {
        address_word_112 = address_word_112 + 0x1u;
    }

    condition_0 = int32_t(index_word_0) != int32_t(0u);

    if (!condition_1) {
        matrix_accumulator_4 = matrix_accumulator_4 + 0x3000u;
    }

    matrix_accumulator_10 = parameters[parameter_optional_control_16];

    if (!condition_2) {
        address_word_112 = 0u - address_word_112;
    }

    if (!condition_1) {
        matrix_accumulator_4 = (address_word_9 << 0x18u) + matrix_accumulator_4;
    }

    uniform_address_word_6 =
        broadcast_warp_word(matrix_accumulator_4, !condition_1, uniform_address_word_6);

    if (!uniform_condition_0) {
        uniform_address_word_4 = 0u - uniform_address_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_address_word_4 << (0xbu & 31u);

        uniform_address_word_4 = uniform_address_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_1);

    if (!condition_0) {
        address_word_112 = ~index_word_0;
    }

    reused_word_110 = address_word_112 * 0x4u + thread_row_113;

    reused_word_110 = reused_word_110 * 0x40u;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    address_word_9 = matrix_accumulator_7 * 0x2000u + reused_word_110;

    address_word_9 = address_word_9 * 0x8u;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    index_word_0 = min(int32_t(0u), int32_t(matrix_accumulator_10));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_11)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_2, address_word_3));

    address_word_9 = threadIdx.z;

    address_word_11 = parameters[parameter_tensor_height_8];

    index_word_0 = 0u - index_word_0 + matrix_accumulator_13;

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x10u);
#endif
    }

    __syncthreads();

    condition_1 = uint32_t(index_word_0) != 0u;

    uniform_address_word_4 = min(int32_t(0u), int32_t(uniform_address_word_7));

    matrix_accumulator_4 = address_word_11 + (0u - uniform_address_word_4);

    uniform_block_index_10 = blockIdx.y;

    condition_2 = uint32_t(matrix_accumulator_4) != 0u;

    if (condition_1) {
        index_word_0 = index_word_0 + 0xffffffffu;
    }

    matrix_accumulator_6 = address_word_9 | matrix_accumulator_6;
    condition_0 = matrix_accumulator_6 != 0;

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3),
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x200ull,
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x400ull,
                                        matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
                                        matrix_fragment_51, memory_ranges, error_counts);

    if (condition_2) {
        matrix_accumulator_4 = matrix_accumulator_4 + 0xffffffffu;
    }

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x600ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x4000ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x4200ull,
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x4400ull,
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x4600ull,
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);

    index_word_111 = matrix_accumulator_5 * address_word_112 + address_word_112;

    if (condition_2) {
        matrix_accumulator_5 = uint32_t(int32_t(matrix_accumulator_4) >> 0x1fu);
    }

    index_word_114 = 0;
    index_word_115 = 0;

    uniform_index_word_11 = uniform_block_index_10 << (0x1u & 31u);

    index_word_111 = 0u - index_word_111 + matrix_accumulator_8;

    if (condition_2) {
        matrix_accumulator_5 =
            uint32_t(combine_words(matrix_accumulator_5, 0u) << 0x3u >> 32) + matrix_accumulator_4;
    }

    if (condition_1) {
        address_word_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);
    }

    matrix_accumulator_4 = index_word_111 * 0x2u;

    address_word_2 = uint32_t(int32_t(matrix_accumulator_13) >> 0x1fu);

    if (condition_1) {
        address_word_3 = uint32_t(combine_words(address_word_3, 0u) << 0x3u >> 32) + index_word_0;
    }

    index_word_0 = uint32_t(int32_t(address_word_11) >> 0x1fu);

    address_word_2 =
        uint32_t(combine_words(address_word_2, 0u) << 0x2u >> 32) + matrix_accumulator_13;

    index_word_0 = uint32_t(combine_words(index_word_0, 0u) << 0x2u >> 32) + address_word_11;

    if (condition_1) {
        index_word_114 = uint32_t(int64_t(int32_t(address_word_3)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (condition_2) {
        index_word_115 =
            uint32_t(int64_t(int32_t(matrix_accumulator_5)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    address_word_3 = uint32_t(int32_t(address_word_2) >> 0x2u);

    matrix_accumulator_5 = uint32_t(int32_t(index_word_0) >> 0x2u);

    address_word_2 = reused_word_110 + 0x20u;
    if (!condition_0) {
        uniform_address_word_4 = (uniform_block_index_10 << 0x3u) + (0u - uniform_address_word_7);

        index_word_0 = index_word_111 * 0x8u + (0u - matrix_accumulator_10);

        uniform_index_word_5 = uniform_address_word_4 + 0x7u;

        uniform_address_word_6 = uniform_address_word_4 + 0x8u;

        uniform_address_word_4 = max(int32_t(0u), int32_t(uniform_address_word_4));

        address_word_9 = index_word_0 + 0x8u;

        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x1du >> 32) + 0x1u;

        uniform_condition_0 = int32_t(uniform_address_word_6) > int32_t(0u);

        uniform_address_word_4 = uint32_t(uint32_t(uniform_address_word_4) >> 0x3u);

        matrix_accumulator_8 = index_word_0 + 0x7u;

        condition_0 = int32_t(address_word_9) > int32_t(0u);

        uniform_index_word_5 = uniform_condition_0 ? uniform_index_word_5 : 0u;

        index_word_0 = max(int32_t(0u), int32_t(index_word_0));

        matrix_accumulator_8 =
            uint32_t(combine_words(matrix_accumulator_8, 0u) << 0x1du >> 32) + 0x1u;

        matrix_accumulator_13 = min(int32_t(index_word_115), int32_t(uniform_index_word_5));

        address_word_9 = condition_0 ? matrix_accumulator_8 : 0u;

        condition_1 = int32_t(matrix_accumulator_13) <= int32_t(uniform_address_word_4);
        if (!condition_1) {
            matrix_accumulator_12 = uint32_t(uint32_t(index_word_0) >> 0x3u);

            matrix_accumulator_17 = index_word_114 * 0x4u;

            matrix_accumulator_15 = min(int32_t(address_word_9), int32_t(index_word_114));

            index_word_0 = uniform_address_word_4;
            do {
                condition_0 = int32_t(matrix_accumulator_12) >= int32_t(matrix_accumulator_15);
                if (!condition_0) {
                    matrix_accumulator_8 = matrix_accumulator_7 + index_word_0;

                    matrix_accumulator_10 = matrix_accumulator_12;

                    matrix_accumulator_19 = matrix_accumulator_17 * matrix_accumulator_8;
                    do {
                        matrix_accumulator_8 = parameters[parameter_wait_low_14];
                        address_word_9 = parameters[parameter_wait_high_15];

                        address_word_11 = matrix_accumulator_19 + matrix_accumulator_10;

                        split_address_words(
                            matrix_accumulator_8, address_word_9,
                            int64_t(int32_t(address_word_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_8, address_word_9));

                        address_word_11 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_8, address_word_9), memory_ranges,
                            error_counts);

                        matrix_accumulator_10 = matrix_accumulator_10 + 0x1u;

                        condition_1 =
                            uint32_t(matrix_accumulator_10) != uint32_t(matrix_accumulator_15);

                        condition_0 = int32_t(address_word_11) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                address_word_11 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_accumulator_8, address_word_9),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(address_word_11) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                index_word_0 = index_word_0 + 0x1u;

                condition_0 = uint32_t(index_word_0) != uint32_t(matrix_accumulator_13);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    matrix_accumulator_8 = parameters[parameter_tensor_height_8];

    address_word_11 = uint32_t(uint32_t(thread_row_113) >> 0x2u);

    index_word_0 = uint32_t(uint32_t(thread_row_113) >> 0x1u);

    __syncthreads();

    condition_0 = false;

    matrix_accumulator_10 = address_word_11 + uniform_index_word_11;

    address_word_9 = index_word_0 & 0x1u;

    uniform_address_word_13 = 0u;

    index_word_0 = matrix_accumulator_4 + address_word_9;

    address_word_9 = thread_row_113 * 0x80u;

    matrix_accumulator_12 = index_word_0;

    matrix_accumulator_8 = matrix_accumulator_8 & 0xfffffffcu;

    condition_3 = uint32_t(matrix_accumulator_8) != uint32_t(0x4u);

    condition_1 = int32_t(matrix_accumulator_10) < int32_t(matrix_accumulator_5) || !condition_3;

    matrix_accumulator_10 = matrix_accumulator_10 * address_word_3;
    if (condition_1) {
        matrix_accumulator_12 = parameters[parameter_tensor_width_9];

        matrix_accumulator_12 = matrix_accumulator_12 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_12) != uint32_t(0x4u);

        matrix_accumulator_12 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_2;

        if (condition_2) {
            matrix_accumulator_12 = index_word_0;
        }
    }
    if (condition_0) {
        matrix_accumulator_20 = parameters[parameter_input_low_0];
        matrix_accumulator_21 = parameters[parameter_input_high_1];
    }

    matrix_accumulator_10 = condition_3 ? matrix_accumulator_10 : 0u;

    matrix_accumulator_18 = address_word_9 & 0x80u;

    uniform_address_word_12 = 0x400u;

    matrix_accumulator_16 = address_word_9 & 0x100u;

    uniform_address_word_7 = (uniform_address_word_13 << 0x18u) + uniform_address_word_12;

    if (condition_0) {
        matrix_accumulator_13 =
            matrix_accumulator_7 + matrix_accumulator_10 + matrix_accumulator_12;
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    address_word_11 = address_word_11 * 0x200u + matrix_accumulator_16;

    address_word_9 = thread_row_113 & 0x1u;

    if (condition_0) {
        matrix_accumulator_14 = matrix_accumulator_13 * 0x800u + matrix_accumulator_18;
    }

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    address_word_11 = address_word_11 + matrix_accumulator_18;

    if (condition_0) {
        matrix_accumulator_15 = uint32_t(int32_t(matrix_accumulator_14) >> 0x1fu);

        matrix_accumulator_12 = matrix_accumulator_14;
    }

    address_word_11 = (address_word_11 << 0x2u) + uniform_address_word_7;

    if (condition_0) {
        matrix_accumulator_13 = matrix_accumulator_15;

        const uint64_t byte_address_0 =
            scaled_word_address(combine_words(matrix_accumulator_20, matrix_accumulator_21),
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));
        matrix_accumulator_14 = uint32_t(byte_address_0);

        matrix_accumulator_13 = uint32_t(byte_address_0 >> 32);

        matrix_accumulator_22 = matrix_accumulator_14;

        matrix_accumulator_23 = matrix_accumulator_13;
    }
    if (!condition_0) {
        matrix_accumulator_12 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_13 = uniform_address_word_4;

        matrix_accumulator_14 = uniform_address_word_4;

        matrix_accumulator_17 = matrix_accumulator_12 * 0x10u + address_word_11;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_13, matrix_accumulator_14, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_17,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_22);

            uniform_address_word_4 = uniform_address_word_12 + 0x3000u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_23);

            uniform_address_word_14 = broadcast_active_lane(address_word_11);

            uniform_address_word_15 = (uniform_address_word_13 << 0x18u) + uniform_address_word_4;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_4 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_4 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_12 = thread_row_113 + 0x4u;

    condition_0 = false;

    matrix_accumulator_14 = index_word_0;

    matrix_accumulator_13 = uint32_t(uint32_t(matrix_accumulator_12) >> 0x2u);

    matrix_accumulator_12 = matrix_accumulator_13 + uniform_index_word_11;

    condition_2 = int32_t(matrix_accumulator_12) < int32_t(matrix_accumulator_5) || !condition_3;

    matrix_accumulator_12 = matrix_accumulator_12 * address_word_3;
    if (condition_2) {
        matrix_accumulator_14 = parameters[parameter_tensor_width_9];

        matrix_accumulator_14 = matrix_accumulator_14 & 0xfffffffcu;

        condition_4 = uint32_t(matrix_accumulator_14) != uint32_t(0x4u);

        matrix_accumulator_14 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_4;

        if (condition_4) {
            matrix_accumulator_14 = index_word_0;
        }
    }
    if (condition_0) {
        matrix_accumulator_22 = parameters[parameter_input_low_0];
        matrix_accumulator_23 = parameters[parameter_input_high_1];
    }

    matrix_accumulator_12 = condition_3 ? matrix_accumulator_12 : 0u;

    matrix_accumulator_13 = matrix_accumulator_13 * 0x200u + matrix_accumulator_16;

    if (condition_0) {
        matrix_accumulator_15 =
            matrix_accumulator_7 + matrix_accumulator_12 + matrix_accumulator_14;
    }

    matrix_accumulator_13 = matrix_accumulator_18 + matrix_accumulator_13;

    if (condition_0) {
        matrix_accumulator_17 = matrix_accumulator_15 * 0x800u + matrix_accumulator_18;
    }

    split_address_words(matrix_accumulator_14, matrix_accumulator_15, 0ull);

    matrix_accumulator_13 = (matrix_accumulator_13 << 0x2u) + uniform_address_word_7;

    if (condition_0) {
        matrix_accumulator_20 = uint32_t(int32_t(matrix_accumulator_17) >> 0x1fu);

        matrix_accumulator_14 = matrix_accumulator_17;

        matrix_accumulator_15 = matrix_accumulator_20;
    }

    split_address_words(matrix_accumulator_20, matrix_accumulator_21, 0ull);

    if (condition_0) {
        const uint64_t byte_address_1 =
            scaled_word_address(combine_words(matrix_accumulator_22, matrix_accumulator_23),
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));
        matrix_accumulator_16 = uint32_t(byte_address_1);

        matrix_accumulator_15 = uint32_t(byte_address_1 >> 32);

        matrix_accumulator_20 = matrix_accumulator_16;

        matrix_accumulator_21 = matrix_accumulator_15;
    }
    if (!condition_0) {
        matrix_accumulator_14 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_16 = uniform_address_word_4;

        matrix_accumulator_15 = uniform_address_word_4;

        matrix_accumulator_16 =
            permute_signed(matrix_accumulator_16, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_17 = matrix_accumulator_16;

        matrix_accumulator_18 = matrix_accumulator_16;

        matrix_accumulator_19 = matrix_accumulator_16;

        matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + matrix_accumulator_13;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_14,
                                             matrix_accumulator_16, matrix_accumulator_17,
                                             matrix_accumulator_18, matrix_accumulator_19,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_20);

            uniform_address_word_4 = uniform_address_word_12 + 0x3000u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_21);

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_15 = (uniform_address_word_13 << 0x18u) + uniform_address_word_4;

            matrix_accumulator_14 = 0x200u;

            uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_4 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_4 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_15 = index_word_0;
    if (condition_1) {
        matrix_accumulator_14 = parameters[parameter_tensor_width_9];

        matrix_accumulator_15 = 0u;

        matrix_accumulator_14 = matrix_accumulator_14 & 0xfffffffcu;

        condition_3 = uint32_t(matrix_accumulator_14) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_3;

        if (condition_3) {
            matrix_accumulator_15 = index_word_0;
        }
    }
    matrix_accumulator_16 = (matrix_accumulator_7 << 0x9u) + 0x40u;
    if (!condition_0) {
        matrix_accumulator_14 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_20 = uniform_address_word_4;

        matrix_accumulator_7 = uniform_address_word_4;

        matrix_accumulator_20 =
            permute_signed(matrix_accumulator_20, matrix_accumulator_7, 0x5410u);

        matrix_accumulator_21 = matrix_accumulator_20;

        matrix_accumulator_22 = matrix_accumulator_20;

        matrix_accumulator_23 = matrix_accumulator_20;

        matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + address_word_11;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_14 + 0x1000u,
                                             matrix_accumulator_20, matrix_accumulator_21,
                                             matrix_accumulator_22, matrix_accumulator_23,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_20 = parameters[parameter_input_low_0];
            matrix_accumulator_21 = parameters[parameter_input_high_1];

            matrix_accumulator_7 = matrix_accumulator_10 + matrix_accumulator_15;

            matrix_accumulator_14 =
                uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x1bu >> 32) + address_word_9;
        }

        uniform_address_word_4 = 0xffffffffu;

        if (condition_0) {
            matrix_accumulator_7 = matrix_accumulator_7 * 0x10u + matrix_accumulator_14;
        }

        split_address_words(matrix_accumulator_14, matrix_accumulator_15, 0ull);

        if (condition_0) {
            matrix_accumulator_7 = matrix_accumulator_7 * 0x80u;

            matrix_accumulator_18 = uint32_t(int32_t(matrix_accumulator_7) >> 0x1fu);

            matrix_accumulator_14 = matrix_accumulator_7;

            matrix_accumulator_15 = matrix_accumulator_18;
        }

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            const uint64_t byte_address_2 =
                scaled_word_address(combine_words(matrix_accumulator_20, matrix_accumulator_21),
                                    combine_words(matrix_accumulator_14, matrix_accumulator_15));
            matrix_accumulator_7 = uint32_t(byte_address_2);

            matrix_accumulator_14 = uint32_t(byte_address_2 >> 32);

            matrix_accumulator_18 = matrix_accumulator_7;

            matrix_accumulator_19 = matrix_accumulator_14;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_5 = broadcast_active_lane(address_word_11);

            uniform_address_word_4 = uniform_address_word_12 + 0x3000u;

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_18);

            uniform_address_word_16 = 0x20u;

            uniform_address_word_15 = broadcast_active_lane(matrix_accumulator_19);

            uniform_address_word_6 = (uniform_address_word_13 << 0x18u) + uniform_address_word_4;

            matrix_accumulator_7 = 0x200u;

            uniform_address_word_4 = uniform_index_word_5 + 0x1000u;

            uniform_index_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_4,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_4,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_7 = index_word_0;
    if (condition_2) {
        matrix_accumulator_7 = parameters[parameter_tensor_width_9];

        matrix_accumulator_7 = matrix_accumulator_7 & 0xfffffffcu;

        condition_3 = uint32_t(matrix_accumulator_7) != uint32_t(0x4u);

        matrix_accumulator_7 = 0u;

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_3;

        if (condition_3) {
            matrix_accumulator_7 = index_word_0;
        }
    }
    if (!condition_0) {
        matrix_accumulator_14 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_20 = uniform_address_word_4;

        matrix_accumulator_7 = uniform_address_word_4;

        matrix_accumulator_20 =
            permute_signed(matrix_accumulator_20, matrix_accumulator_7, 0x5410u);

        matrix_accumulator_21 = matrix_accumulator_20;

        matrix_accumulator_22 = matrix_accumulator_20;

        matrix_accumulator_23 = matrix_accumulator_20;

        matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + matrix_accumulator_13;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_14 + 0x1000u,
                                             matrix_accumulator_20, matrix_accumulator_21,
                                             matrix_accumulator_22, matrix_accumulator_23,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_20 = parameters[parameter_input_low_0];
            matrix_accumulator_21 = parameters[parameter_input_high_1];

            matrix_accumulator_7 = matrix_accumulator_12 + matrix_accumulator_7;

            matrix_accumulator_14 =
                uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x1bu >> 32) + address_word_9;
        }

        uniform_address_word_4 = 0xffffffffu;

        if (condition_0) {
            matrix_accumulator_7 = matrix_accumulator_7 * 0x10u + matrix_accumulator_14;
        }

        split_address_words(matrix_accumulator_14, matrix_accumulator_15, 0ull);

        if (condition_0) {
            matrix_accumulator_7 = matrix_accumulator_7 * 0x80u;

            matrix_accumulator_18 = uint32_t(int32_t(matrix_accumulator_7) >> 0x1fu);

            matrix_accumulator_14 = matrix_accumulator_7;

            matrix_accumulator_15 = matrix_accumulator_18;
        }

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            const uint64_t byte_address_3 =
                scaled_word_address(combine_words(matrix_accumulator_20, matrix_accumulator_21),
                                    combine_words(matrix_accumulator_14, matrix_accumulator_15));
            matrix_accumulator_7 = uint32_t(byte_address_3);

            matrix_accumulator_14 = uint32_t(byte_address_3 >> 32);

            matrix_accumulator_18 = matrix_accumulator_7;

            matrix_accumulator_19 = matrix_accumulator_14;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_5 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_4 = uniform_address_word_12 + 0x3000u;

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_18);

            uniform_address_word_16 = 0x20u;

            uniform_address_word_15 = broadcast_active_lane(matrix_accumulator_19);

            uniform_address_word_6 = (uniform_address_word_13 << 0x18u) + uniform_address_word_4;

            matrix_accumulator_7 = 0x200u;

            uniform_address_word_4 = uniform_index_word_5 + 0x1000u;

            uniform_index_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_4,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_4,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = uint32_t(matrix_accumulator_6) != 0u;

    matrix_accumulator_16 = matrix_accumulator_16 + 0x40u;
    if (!condition_0) {
        uniform_address_word_14 = parameters[parameter_optional_control_16];
        uniform_address_word_15 = parameters[parameter_scalar_control_17];

        uniform_address_word_4 = (uniform_block_index_10 << 0x3u) + (0u - uniform_address_word_15);

        matrix_accumulator_6 = (index_word_111 << 0x3u) + (0u - uniform_address_word_14);

        uniform_index_word_5 = uniform_address_word_4 + 0x7u;

        uniform_address_word_6 = uniform_address_word_4 + 0x8u;

        uniform_address_word_4 = max(int32_t(0u), int32_t(uniform_address_word_4));

        matrix_accumulator_14 = matrix_accumulator_6 + 0x8u;

        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x1du >> 32) + 0x1u;

        uniform_condition_0 = int32_t(uniform_address_word_6) > int32_t(0u);

        uniform_address_word_4 = uint32_t(uint32_t(uniform_address_word_4) >> 0x3u);

        matrix_accumulator_7 = matrix_accumulator_6 + 0x7u;

        condition_3 = int32_t(matrix_accumulator_14) > int32_t(0u);

        uniform_index_word_5 = uniform_condition_0 ? uniform_index_word_5 : 0u;

        matrix_accumulator_6 = max(int32_t(0u), int32_t(matrix_accumulator_6));

        matrix_accumulator_7 =
            uint32_t(combine_words(matrix_accumulator_7, 0u) << 0x1du >> 32) + 0x1u;

        matrix_accumulator_21 = min(int32_t(index_word_115), int32_t(uniform_index_word_5));

        matrix_accumulator_7 = condition_3 ? matrix_accumulator_7 : 0u;

        condition_0 = int32_t(matrix_accumulator_21) <= int32_t(uniform_address_word_4);
        if (!condition_0) {
            matrix_accumulator_20 = uint32_t(uint32_t(matrix_accumulator_6) >> 0x3u);

            matrix_accumulator_22 = min(int32_t(matrix_accumulator_7), int32_t(index_word_114));

            matrix_accumulator_14 = uniform_address_word_4;

            matrix_accumulator_15 = uint32_t(uint32_t(matrix_accumulator_16) >> 0x7u);
            do {
                condition_0 = int32_t(matrix_accumulator_20) >= int32_t(matrix_accumulator_22);
                if (!condition_0) {
                    matrix_accumulator_7 = matrix_accumulator_14 * 0x4u + matrix_accumulator_15;

                    matrix_accumulator_17 = matrix_accumulator_20;

                    matrix_accumulator_18 = matrix_accumulator_7 * index_word_114;
                    do {
                        matrix_accumulator_6 = parameters[parameter_wait_low_14];
                        matrix_accumulator_7 = parameters[parameter_wait_high_15];

                        matrix_accumulator_19 = matrix_accumulator_18 + matrix_accumulator_17;

                        split_address_words(
                            matrix_accumulator_6, matrix_accumulator_7,
                            int64_t(int32_t(matrix_accumulator_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_6, matrix_accumulator_7));

                        matrix_accumulator_19 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_6, matrix_accumulator_7),
                            memory_ranges, error_counts);

                        matrix_accumulator_17 = matrix_accumulator_17 + 0x1u;

                        condition_3 =
                            uint32_t(matrix_accumulator_17) != uint32_t(matrix_accumulator_22);

                        condition_0 = int32_t(matrix_accumulator_19) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                matrix_accumulator_19 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_accumulator_6, matrix_accumulator_7),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(matrix_accumulator_19) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_3);
                }
                matrix_accumulator_14 = matrix_accumulator_14 + 0x1u;

                condition_0 = uint32_t(matrix_accumulator_14) != uint32_t(matrix_accumulator_21);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    __syncthreads();

    condition_0 = false;

    matrix_accumulator_7 = index_word_0;
    if (condition_1) {
        matrix_accumulator_6 = parameters[parameter_tensor_width_9];

        matrix_accumulator_7 = 0u;

        matrix_accumulator_6 = matrix_accumulator_6 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_1;

        if (condition_1) {
            matrix_accumulator_7 = index_word_0;
        }
    }
    if (!condition_0) {
        matrix_accumulator_6 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_20 = uniform_address_word_4;

        matrix_accumulator_7 = uniform_address_word_4;

        matrix_accumulator_20 =
            permute_signed(matrix_accumulator_20, matrix_accumulator_7, 0x5410u);

        matrix_accumulator_21 = matrix_accumulator_20;

        matrix_accumulator_22 = matrix_accumulator_20;

        matrix_accumulator_23 = matrix_accumulator_20;

        matrix_accumulator_6 = matrix_accumulator_6 * 0x10u + address_word_11;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_6 + 0x2000u,
                                             matrix_accumulator_20, matrix_accumulator_21,
                                             matrix_accumulator_22, matrix_accumulator_23,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_18 = parameters[parameter_input_low_0];
            matrix_accumulator_19 = parameters[parameter_input_high_1];

            matrix_accumulator_10 = matrix_accumulator_10 + matrix_accumulator_7;

            matrix_accumulator_7 =
                uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x1bu >> 32) + address_word_9;
        }

        uniform_address_word_4 = 0xffffffffu;

        if (condition_0) {
            matrix_accumulator_10 = matrix_accumulator_10 * 0x10u + matrix_accumulator_7;
        }

        split_address_words(matrix_accumulator_6, matrix_accumulator_7, 0ull);

        if (condition_0) {
            matrix_accumulator_10 = matrix_accumulator_10 * 0x80u;

            matrix_accumulator_15 = uint32_t(int32_t(matrix_accumulator_10) >> 0x1fu);

            matrix_accumulator_6 = matrix_accumulator_10;

            matrix_accumulator_7 = matrix_accumulator_15;
        }

        split_address_words(matrix_accumulator_14, matrix_accumulator_15, 0ull);

        if (condition_0) {
            // Later control flow also consumes this address-overflow predicate.
            condition_1 =
                (uint64_t(matrix_accumulator_6) << 2) + matrix_accumulator_18 > 0xffffffffull;
            const uint64_t byte_address_4 =
                scaled_word_address(combine_words(matrix_accumulator_18, matrix_accumulator_19),
                                    combine_words(matrix_accumulator_6, matrix_accumulator_7));
            matrix_accumulator_10 = uint32_t(byte_address_4);

            matrix_accumulator_7 = uint32_t(byte_address_4 >> 32);

            matrix_accumulator_14 = matrix_accumulator_10;

            matrix_accumulator_15 = matrix_accumulator_7;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_5 = 0u;

            uniform_address_word_7 = broadcast_active_lane(address_word_11);

            uniform_address_word_4 = 0x400u;

            uniform_address_word_12 = broadcast_active_lane(matrix_accumulator_14);

            uniform_address_word_4 = uniform_address_word_4 + 0x3000u;

            uniform_address_word_13 = broadcast_active_lane(matrix_accumulator_15);

            matrix_accumulator_6 = 0x200u;

            uniform_address_word_6 = (uniform_index_word_5 << 0x18u) + uniform_address_word_4;

            uniform_address_word_4 = uniform_address_word_7 + 0x2000u;

            uniform_index_word_5 = uniform_address_word_6 + 0x10u;

            uniform_address_word_7 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_4,
                combine_words(uniform_address_word_12, uniform_address_word_13),
                uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_4,
                                   combine_words(uniform_address_word_12, uniform_address_word_13),
                                   uniform_address_word_7 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_7 = index_word_0;
    if (condition_2) {
        matrix_accumulator_6 = parameters[parameter_tensor_width_9];

        matrix_accumulator_7 = 0u;

        matrix_accumulator_6 = matrix_accumulator_6 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_6) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_0) < int32_t(address_word_3) || !condition_1;

        if (condition_1) {
            matrix_accumulator_7 = index_word_0;
        }
    }
    if (!condition_0) {
        matrix_accumulator_6 = threadIdx.x & 31u;

        uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_4;

        matrix_accumulator_7 = uniform_address_word_4;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_7, 0x5410u);

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_6 = matrix_accumulator_6 * 0x10u + matrix_accumulator_13;

        matrix_accumulator_13 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_6 + 0x2000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_14 = parameters[parameter_input_low_0];
            matrix_accumulator_15 = parameters[parameter_input_high_1];

            matrix_accumulator_12 = matrix_accumulator_12 + matrix_accumulator_7;

            address_word_9 =
                uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x1bu >> 32) + address_word_9;
        }

        split_address_words(matrix_accumulator_6, matrix_accumulator_7, 0ull);

        uniform_address_word_4 = 0xffffffffu;

        if (condition_0) {
            address_word_9 = matrix_accumulator_12 * 0x10u + address_word_9;

            address_word_9 = address_word_9 * 0x80u;

            matrix_accumulator_10 = uint32_t(int32_t(address_word_9) >> 0x1fu);

            matrix_accumulator_6 = address_word_9;

            matrix_accumulator_7 = matrix_accumulator_10;
        }

        split_address_words(matrix_accumulator_10, address_word_11, 0ull);

        if (condition_0) {
            // Later control flow also consumes this address-overflow predicate.
            condition_1 =
                (uint64_t(matrix_accumulator_6) << 2) + matrix_accumulator_14 > 0xffffffffull;
            const uint64_t byte_address_5 =
                scaled_word_address(combine_words(matrix_accumulator_14, matrix_accumulator_15),
                                    combine_words(matrix_accumulator_6, matrix_accumulator_7));
            address_word_9 = uint32_t(byte_address_5);

            matrix_accumulator_6 = uint32_t(byte_address_5 >> 32);

            matrix_accumulator_10 = address_word_9;

            address_word_11 = matrix_accumulator_6;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_index_word_5 = 0u;

            uniform_address_word_7 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_4 = 0x400u;

            uniform_address_word_12 = broadcast_active_lane(matrix_accumulator_10);

            uniform_address_word_4 = uniform_address_word_4 + 0x3000u;

            uniform_address_word_13 = broadcast_active_lane(address_word_11);

            matrix_accumulator_6 = 0x200u;

            uniform_address_word_6 = (uniform_index_word_5 << 0x18u) + uniform_address_word_4;

            uniform_address_word_4 = uniform_address_word_7 + 0x2000u;

            uniform_index_word_5 = uniform_address_word_6 + 0x10u;

            uniform_address_word_7 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_4,
                combine_words(uniform_address_word_12, uniform_address_word_13),
                uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_4,
                                   combine_words(uniform_address_word_12, uniform_address_word_13),
                                   uniform_address_word_7 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_7 = 0u;

    matrix_accumulator_6 = 0x400u;

    matrix_accumulator_10 = parameters[parameter_tensor_width_9];

    condition_6 = uint32_t(matrix_accumulator_8) != uint32_t(0x4u);

    condition_0 = uint32_t(matrix_accumulator_8) != uint32_t(0x4u);

    matrix_accumulator_6 = matrix_accumulator_6 + 0x3000u;

    condition_2 = int32_t(matrix_accumulator_5) <= int32_t(uniform_index_word_11) && condition_6;

    matrix_accumulator_10 = matrix_accumulator_10 & 0xfffffffcu;

    address_word_9 = (matrix_accumulator_7 << 0x18u) + matrix_accumulator_6;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = address_word_9;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_6, matrix_accumulator_7,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_6 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 =
        wait_copy_barrier(shared_tile, address_word_9, async_tokens[(((address_word_9) >> 3) & 1)]);
#else
    __syncthreads();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, address_word_9,
                                            async_tokens[(((address_word_9) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    condition_4 = uint32_t(matrix_accumulator_10) == uint32_t(0x4u) || condition_2;

    matrix_accumulator_16 = matrix_accumulator_4 + 0x1u;

    uniform_address_word_4 = uniform_index_word_11 + 0x1u;

    matrix_accumulator_8 = address_word_3 * uniform_index_word_11;

    condition_5 = int32_t(matrix_accumulator_5) > int32_t(uniform_index_word_11) || !condition_0;

    condition_3 = int32_t(matrix_accumulator_4) < int32_t(address_word_3) || condition_4;

    condition_4 = int32_t(matrix_accumulator_16) < int32_t(address_word_3) || condition_4;

    condition_2 = uint32_t(matrix_accumulator_10) == uint32_t(0x4u) && !condition_2;

    condition_1 = int32_t(matrix_accumulator_5) <= int32_t(uniform_address_word_4) && condition_6;

    matrix_accumulator_6 = condition_0 ? matrix_accumulator_8 : 0u;

    matrix_accumulator_8 = matrix_accumulator_8 + address_word_3;

    condition_3 = condition_3 & condition_5;

    matrix_accumulator_30 = threadIdx.x & 31u;

    condition_5 = condition_4 & condition_5;

    matrix_accumulator_28 = parameters[parameter_trained_weights_low_6];
    matrix_accumulator_29 = parameters[parameter_trained_weights_high_7];

    condition_4 = uint32_t(matrix_accumulator_10) == uint32_t(0x4u) || condition_1;

    matrix_accumulator_8 = condition_0 ? matrix_accumulator_8 : 0u;

    condition_0 = int32_t(matrix_accumulator_5) > int32_t(uniform_address_word_4) || !condition_0;

    matrix_accumulator_31 = !condition_2 ? matrix_accumulator_4 : 0u;

    matrix_accumulator_5 = !condition_2 ? matrix_accumulator_16 : 0u;

    condition_2 = int32_t(matrix_accumulator_4) < int32_t(address_word_3) || condition_4;

    condition_1 = uint32_t(matrix_accumulator_10) == uint32_t(0x4u) && !condition_1;

    condition_4 = int32_t(matrix_accumulator_16) < int32_t(address_word_3) || condition_4;

    matrix_accumulator_32 = reused_word_110 + 0x8u;

    if (condition_3) {
        matrix_accumulator_10 = parameters[parameter_input_skip_low_2];
        address_word_11 = parameters[parameter_input_skip_high_3];
    }

    condition_2 = condition_2 & condition_0;

    uniform_address_word_4 = parameters[parameter_scalar_control_17];

    condition_4 = condition_4 & condition_0;

    address_word_9 = !condition_1 ? matrix_accumulator_4 : 0u;

    if (condition_3) {
        matrix_accumulator_7 = matrix_accumulator_6 + matrix_accumulator_31;
    }

    if (condition_5) {
        matrix_accumulator_13 = matrix_accumulator_6 + matrix_accumulator_5;

        matrix_accumulator_20 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_21 = parameters[parameter_input_skip_high_3];
    }

    if (!condition_3) {
        matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x200u + reused_word_110;
    }

    if (!condition_5) {
        matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_5) {
        matrix_accumulator_13 = matrix_accumulator_13 * 0x200u + reused_word_110;
    }

    if (condition_2) {
        address_word_3 = matrix_accumulator_8 + address_word_9;
    }

    if (!condition_3) {
        matrix_accumulator_15 =
            permute_signed(matrix_accumulator_15, matrix_accumulator_15, 0x5410u);
    }

    if (condition_2) {
        matrix_accumulator_24 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_25 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_4 = address_word_3 * 0x200u + reused_word_110;
    }

    if (!condition_2) {
        matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        address_word_3 = matrix_accumulator_7 * 0x4u;
    }

    if (!condition_3) {
        matrix_accumulator_12 = matrix_accumulator_15;
    }

    if (condition_5) {
        matrix_accumulator_7 = matrix_accumulator_13 * 0x4u;
    }

    if (condition_4) {
        matrix_accumulator_38 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_39 = parameters[parameter_input_skip_high_3];
    }

    if (condition_2) {
        matrix_accumulator_13 = matrix_accumulator_4 * 0x4u;
    }

    matrix_accumulator_4 = matrix_accumulator_30;

    if (condition_3) {
        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));
    }

    address_word_3 = !condition_1 ? matrix_accumulator_16 : 0u;

    if (!condition_3) {
        matrix_accumulator_14 = matrix_accumulator_15;
    }

    matrix_accumulator_33 = matrix_accumulator_4 & 0x7ffffffcu;

    if (condition_5) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));
    }

    if (condition_4) {
        matrix_accumulator_7 = matrix_accumulator_8 + address_word_3;
    }

    if (condition_3) {
        matrix_accumulator_34 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_35 = parameters[parameter_input_skip_high_3];
    }

    matrix_accumulator_33 = 0u - matrix_accumulator_33 + matrix_accumulator_30;

    if (condition_3) {
        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_10, address_word_11));
    }

    if (!condition_2) {
        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_23, matrix_accumulator_23, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                            int64_t(int32_t(matrix_accumulator_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    if (!condition_3) {
        matrix_accumulator_13 = matrix_accumulator_15;
    }

    if (condition_4) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x200u + reused_word_110;
    }

    matrix_accumulator_4 = matrix_accumulator_33 * 0x2u + reused_word_110;

    if (!condition_5) {
        matrix_accumulator_19 =
            permute_signed(matrix_accumulator_19, matrix_accumulator_19, 0x5410u);
    }

    if (condition_4) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x4u;
    }

    if (!condition_4) {
        matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_36, matrix_accumulator_37,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));
    }

    matrix_fragment_73 = uint32_t(int32_t(matrix_accumulator_4) >> 0x1u);

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_10, address_word_11), matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }

    if (!condition_2) {
        matrix_accumulator_20 = matrix_accumulator_23;

        matrix_accumulator_21 = matrix_accumulator_23;

        matrix_accumulator_22 = matrix_accumulator_23;
    }

    matrix_accumulator_4 = reused_word_110 + 0x10u;

    if (condition_4) {
        split_address_words(matrix_accumulator_38, matrix_accumulator_39,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_38, matrix_accumulator_39));
    }

    if (condition_3) {
        matrix_accumulator_7 = matrix_accumulator_6 + matrix_accumulator_31;
    }

    if (!condition_5) {
        matrix_accumulator_16 = matrix_accumulator_19;

        matrix_accumulator_17 = matrix_accumulator_19;

        matrix_accumulator_18 = matrix_accumulator_19;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_24, matrix_accumulator_25));
    }

    if (!condition_4) {
        matrix_accumulator_27 =
            permute_signed(matrix_accumulator_27, matrix_accumulator_27, 0x5410u);
    }

    matrix_accumulator_4 = matrix_accumulator_33 * 0x2u + matrix_accumulator_4;

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_10, address_word_11), matrix_accumulator_20,
            matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
            error_counts);
    }

    if (condition_3) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x200u + address_word_2;
    }

    if (!condition_4) {
        matrix_accumulator_24 = matrix_accumulator_27;

        matrix_accumulator_25 = matrix_accumulator_27;

        matrix_accumulator_26 = matrix_accumulator_27;
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_36, matrix_accumulator_37), matrix_accumulator_16,
            matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
            error_counts);
    }

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        int64_t(int32_t(matrix_fragment_73)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_5) {
        matrix_accumulator_10 = parameters[parameter_input_skip_low_2];
        address_word_11 = parameters[parameter_input_skip_high_3];
    }

    if (condition_4) {
        split_address_words(matrix_accumulator_38, matrix_accumulator_39,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_38, matrix_accumulator_39));
    }

    matrix_accumulator_31 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_32 = matrix_accumulator_33 * 0x2u + matrix_accumulator_32;

    matrix_accumulator_37 = uint32_t(int32_t(matrix_accumulator_4) >> 0x1u);

    if (condition_3) {
        matrix_accumulator_7 = matrix_accumulator_7 * 0x4u;
    }

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_38, matrix_accumulator_39), matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);
    }

    matrix_fragment_79 = uint32_t(int32_t(matrix_accumulator_32) >> 0x1u);

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        int64_t(int32_t(matrix_accumulator_37)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_2) {
        matrix_accumulator_36 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_37 = parameters[parameter_input_skip_high_3];
    }

    if (condition_5) {
        matrix_fragment_81 = matrix_accumulator_6 + matrix_accumulator_5;
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_38, matrix_accumulator_39,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_34, matrix_accumulator_35));
    }

    matrix_accumulator_32 = reused_word_110 + 0x18u;

    matrix_fragment_77 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_4) {
        matrix_accumulator_34 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_35 = parameters[parameter_input_skip_high_3];
    }

    split_address_words(matrix_fragment_78, matrix_fragment_79,
                        int64_t(int32_t(matrix_fragment_79)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_5) {
        matrix_fragment_81 = matrix_fragment_81 * 0x200u + address_word_2;
    }

    matrix_accumulator_32 = matrix_accumulator_33 * 0x2u + matrix_accumulator_32;

    matrix_fragment_78 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_78, matrix_fragment_79) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_5) {
        matrix_fragment_81 = matrix_fragment_81 * 0x4u;
    }

    if (!condition_3) {
        matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_75 = uint32_t(int32_t(matrix_accumulator_32) >> 0x1u);

    if (condition_5) {
        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));
    }

    if (condition_2) {
        matrix_fragment_79 = matrix_accumulator_8 + address_word_9;
    }

    if (condition_4) {
        address_word_9 = matrix_accumulator_8 + address_word_3;
    }

    if (!condition_3) {
        matrix_accumulator_7 = permute_signed(matrix_accumulator_7, matrix_accumulator_7, 0x5410u);
    }

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_2) {
        matrix_fragment_79 = matrix_fragment_79 * 0x200u + address_word_2;
    }

    if (condition_4) {
        matrix_fragment_81 = address_word_9 * 0x200u + address_word_2;
    }

    if (!condition_5) {
        matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_fragment_79 = matrix_fragment_79 * 0x4u;
    }

    if (!condition_3) {
        matrix_accumulator_4 = matrix_accumulator_7;
    }

    if (condition_4) {
        matrix_fragment_81 = matrix_fragment_81 * 0x4u;
    }

    if (!condition_3) {
        matrix_accumulator_5 = matrix_accumulator_7;

        matrix_accumulator_6 = matrix_accumulator_7;
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_38, matrix_accumulator_39,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_38, matrix_accumulator_39));
    }

    address_word_3 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0x40000ull, memory_ranges,
        error_counts);

    if (!condition_5) {
        address_word_11 = permute_signed(matrix_accumulator_32, matrix_accumulator_32, 0x5410u);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_80, matrix_fragment_81,
                            int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_34, matrix_accumulator_35));
    }

    if (!condition_2) {
        matrix_accumulator_32 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_38, matrix_accumulator_39), matrix_accumulator_4,
            matrix_accumulator_5, matrix_accumulator_6, matrix_accumulator_7, memory_ranges,
            error_counts);
    }

    if (condition_2) {
        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_fragment_79)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_36, matrix_accumulator_37));
    }

    if (condition_5) {
        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_72, matrix_fragment_73));
    }

    if (!condition_2) {
        matrix_accumulator_39 =
            permute_signed(matrix_accumulator_32, matrix_accumulator_32, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_34, matrix_accumulator_35,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_74, matrix_fragment_75));
    }

    if (condition_4) {
        split_address_words(matrix_fragment_80, matrix_fragment_81,
                            int64_t(int32_t(matrix_accumulator_30)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));
    }

    matrix_accumulator_30 = reused_word_110 + 0x28u;

    if (!condition_2) {
        matrix_accumulator_36 = matrix_accumulator_39;

        matrix_accumulator_37 = matrix_accumulator_39;

        matrix_accumulator_38 = matrix_accumulator_39;
    }

    matrix_accumulator_30 = matrix_accumulator_33 * 0x2u + matrix_accumulator_30;

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_34, matrix_accumulator_35), matrix_accumulator_36,
            matrix_accumulator_37, matrix_accumulator_38, matrix_accumulator_39, memory_ranges,
            error_counts);
    }

    if (!condition_4) {
        matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_5) {
        matrix_accumulator_8 = address_word_11;
    }

    matrix_accumulator_35 = uint32_t(int32_t(matrix_accumulator_30) >> 0x1u);

    matrix_accumulator_30 = reused_word_110 + 0x30u;

    if (!condition_5) {
        address_word_9 = address_word_11;

        matrix_accumulator_10 = address_word_11;
    }

    matrix_accumulator_32 = reused_word_110 + 0x38u;

    matrix_accumulator_30 = matrix_accumulator_33 * 0x2u + matrix_accumulator_30;

    if (!condition_4) {
        matrix_fragment_75 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5410u);
    }

    address_word_2 = matrix_accumulator_33 * 0x2u + address_word_2;

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73), matrix_accumulator_8,
            address_word_9, matrix_accumulator_10, address_word_11, memory_ranges, error_counts);
    }

    matrix_fragment_79 = uint32_t(int32_t(matrix_accumulator_30) >> 0x1u);

    matrix_accumulator_30 = matrix_accumulator_33 * 0x2u + matrix_accumulator_32;

    if (!condition_4) {
        matrix_fragment_74 = matrix_fragment_75;
    }

    matrix_fragment_83 = uint32_t(int32_t(address_word_2) >> 0x1u);

    split_address_words(matrix_accumulator_32, matrix_accumulator_33,
                        int64_t(int32_t(matrix_fragment_79)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (!condition_4) {
        matrix_fragment_72 = matrix_fragment_75;

        matrix_fragment_73 = matrix_fragment_75;
    }

    matrix_fragment_79 = uint32_t(int32_t(matrix_accumulator_30) >> 0x1u);

    split_address_words(matrix_fragment_82, matrix_fragment_83,
                        int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    matrix_accumulator_87 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_34, matrix_accumulator_35,
                        int64_t(int32_t(matrix_accumulator_35)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_80, matrix_fragment_81), matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);
    }

    split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                        int64_t(int32_t(matrix_fragment_79)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    address_word_2 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_fragment_76 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_34, matrix_accumulator_35) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_89 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_28, matrix_accumulator_29) + 0x40000ull, memory_ranges,
        error_counts);

    uniform_address_word_4 = (uniform_block_index_10 << 0x3u) + (0u - uniform_address_word_4);

    uniform_index_word_5 = uniform_address_word_4 + 0x8u;

    uniform_address_word_4 = uniform_address_word_4 + 0x7u;

    uniform_condition_0 = int32_t(uniform_index_word_5) > int32_t(0u);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_address_word_4, 0u) << 0x1du >> 32) + 0x1u;

    uniform_address_word_4 = uniform_condition_0 ? uniform_address_word_4 : 0u;

    index_word_116 = 0u;

    index_word_115 = min(int32_t(index_word_115), int32_t(uniform_address_word_4));

    matrix_accumulator_29 = unpack_fp8_bits(matrix_accumulator_12);

    matrix_fragment_81 = unpack_fp8_bits(matrix_accumulator_12 >> 16);

    matrix_fragment_83 = unpack_fp8_bits(matrix_accumulator_14);

    matrix_accumulator_12 = unpack_fp8_bits(matrix_accumulator_14 >> 16);

    matrix_accumulator_35 = unpack_fp8_bits(matrix_accumulator_13);

    matrix_fragment_82 = unpack_fp8_bits(matrix_accumulator_13 >> 16);

    matrix_accumulator_84 = unpack_fp8_bits(matrix_accumulator_15);

    matrix_accumulator_102 = unpack_fp8_bits(matrix_accumulator_20);

    matrix_accumulator_103 = unpack_fp8_bits(matrix_accumulator_21);

    matrix_accumulator_85 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_accumulator_14 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_16);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_accumulator_13 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    matrix_accumulator_30 = unpack_fp8_bits(matrix_accumulator_24);

    matrix_accumulator_19 = unpack_fp8_bits(matrix_accumulator_25);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    matrix_accumulator_90 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

    matrix_accumulator_91 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

    matrix_accumulator_92 = unpack_fp8_bits(matrix_accumulator_22);

    matrix_accumulator_93 = unpack_fp8_bits(matrix_accumulator_23);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_24 >> 16);

    matrix_accumulator_25 = unpack_fp8_bits(matrix_accumulator_25 >> 16);

    matrix_accumulator_86 = unpack_fp8_bits(matrix_accumulator_26);

    matrix_accumulator_88 = unpack_fp8_bits(matrix_accumulator_27);

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_15 = unpack_fp8_bits(matrix_accumulator_15 >> 16);

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_94 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_95 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_79 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_80 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_106 = unpack_fp8_bits(matrix_accumulator_6 >> 16);

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_78 = unpack_fp8_bits(matrix_accumulator_6);

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_107 = unpack_fp8_bits(matrix_accumulator_7 >> 16);

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_77 = unpack_fp8_bits(matrix_accumulator_7);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_fragment_81 = unpack_fp8_bits(matrix_accumulator_5);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_fragment_80 = unpack_fp8_bits(matrix_accumulator_4);

    matrix_accumulator_96 = unpack_fp8_bits(matrix_accumulator_36);

    matrix_accumulator_98 = unpack_fp8_bits(matrix_accumulator_36 >> 16);

    matrix_accumulator_97 = unpack_fp8_bits(matrix_accumulator_37);

    matrix_accumulator_99 = unpack_fp8_bits(matrix_accumulator_37 >> 16);

    matrix_accumulator_100 = unpack_fp8_bits(matrix_accumulator_38);

    matrix_accumulator_108 = unpack_fp8_bits(matrix_accumulator_38 >> 16);

    matrix_accumulator_101 = unpack_fp8_bits(matrix_accumulator_39);

    matrix_accumulator_109 = unpack_fp8_bits(matrix_accumulator_39 >> 16);

    matrix_accumulator_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_14),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_4 = unpack_fp8_bits(matrix_accumulator_4 >> 16);

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_6 = unpack_fp8_bits(matrix_accumulator_8);

    matrix_accumulator_7 = unpack_fp8_bits(address_word_9);

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_84 = unpack_fp8_bits(matrix_accumulator_10);

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(address_word_3));

    matrix_accumulator_104 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

    matrix_accumulator_85 = unpack_fp8_bits(address_word_11);

    matrix_accumulator_105 = unpack_fp8_bits(address_word_11 >> 16);

    matrix_accumulator_5 = unpack_fp8_bits(matrix_accumulator_5 >> 16);

    matrix_fragment_79 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_fragment_72);

    matrix_accumulator_37 = unpack_fp8_bits(matrix_fragment_73);

    matrix_accumulator_38 = unpack_fp8_bits(matrix_fragment_72 >> 16);

    matrix_accumulator_39 = unpack_fp8_bits(matrix_fragment_73 >> 16);

    matrix_accumulator_86 = unpack_fp8_bits(matrix_fragment_74);

    matrix_accumulator_88 = unpack_fp8_bits(matrix_fragment_74 >> 16);

    matrix_fragment_72 = unpack_fp8_bits(matrix_fragment_75);

    matrix_accumulator_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_6),
                            select_half_operand<2, false, false>(address_word_2));

    address_word_3 = unpack_fp8_bits(address_word_9 >> 16);

    address_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                                         select_half_operand<2, false, false>(address_word_2));

    matrix_fragment_75 = unpack_fp8_bits(matrix_fragment_75 >> 16);

    matrix_accumulator_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(address_word_2));

    address_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                          select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_4),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_5),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    address_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                         select_half_operand<2, false, false>(matrix_fragment_76));

    address_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_3),
                                         select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_accumulator_89));
    do {
        index_word_121 = uint32_t(uint32_t(index_word_116) >> 0x6u);

        index_word_117 = 0u;

        index_word_120 = 0x400u;

        __syncwarp(__activemask());

        matrix_fragment_72 = permute_signed(index_word_121, 0u, 0x9910u);

        warp_lane_119 = threadIdx.x & 31u;

        condition_0 = uint32_t(index_word_116) > uint32_t(0x1bfu);

        matrix_fragment_72 = matrix_fragment_72 * 0xabu;

        matrix_fragment_72 = matrix_fragment_72 & 0xffffu;

        matrix_fragment_72 = uint32_t(uint32_t(matrix_fragment_72) >> 0x9u);

        matrix_fragment_72 = permute_signed(matrix_fragment_72, 0u, 0x9910u);

        matrix_fragment_72 = matrix_fragment_72 * 0x3u;

        matrix_fragment_73 = (index_word_117 << 0x18u) + index_word_120;

        matrix_fragment_72 = 0u - matrix_fragment_72;

        matrix_fragment_72 = permute_signed(matrix_fragment_72, 0u, 0x7710u);

        index_word_121 = index_word_121 + matrix_fragment_72;

        index_word_118 = index_word_121 & 0xffu;

        index_word_118 = index_word_118 * 0x1000u + matrix_fragment_73;

        tile_word_122 = warp_lane_119 * 0x10u + index_word_118;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122, matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
            matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0x400u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0x800u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_10, address_word_11, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_10,
                                 address_word_11);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0x200u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_10, address_word_11, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_10,
                                 address_word_11);

        multiply_accumulate_half(matrix_accumulator_4, matrix_accumulator_5, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_4,
                                 matrix_accumulator_5);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0x600u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_8, address_word_9, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_8,
                                 address_word_9);

        multiply_accumulate_half(address_word_2, address_word_3, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_50, matrix_fragment_51, address_word_2,
                                 address_word_3);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0xa00u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_90,
                                 matrix_accumulator_91);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0xc00u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_8, address_word_9, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_8,
                                 address_word_9);

        multiply_accumulate_half(address_word_2, address_word_3, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, address_word_2,
                                 address_word_3);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_104,
                                 matrix_accumulator_105);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_122 + 0xe00u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_88,
                                 matrix_accumulator_89);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_90,
                                 matrix_accumulator_91);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_6, matrix_accumulator_7, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_6,
                                 matrix_accumulator_7);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_88,
                                 matrix_accumulator_89);
        if (!condition_0) {
            matrix_fragment_43 = blockIdx.z;

            matrix_fragment_74 = index_word_116 + 0x40u;

            matrix_fragment_40 = parameters[parameter_trained_weights_low_6];
            matrix_fragment_41 = parameters[parameter_trained_weights_high_7];

            matrix_fragment_43 = matrix_fragment_43 * 0x200u + matrix_fragment_74;

            matrix_fragment_74 = uint32_t(uint32_t(matrix_fragment_74) >> 0x6u);

            matrix_fragment_43 = matrix_fragment_43 * 0x10u + reused_word_110;

            matrix_fragment_75 = permute_signed(matrix_fragment_74, 0u, 0x9910u);

            matrix_fragment_43 = matrix_fragment_43 * 0x8u;

            matrix_fragment_75 = matrix_fragment_75 * 0xabu;

            split_address_words(matrix_fragment_40, matrix_fragment_41,
                                int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_40, matrix_fragment_41));

            matrix_fragment_75 = matrix_fragment_75 & 0xffffu;

            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(warp_lane_119)) * int64_t(int32_t(0x10u)) +
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

            index_word_120 = index_word_120 + 0x3000u;

            matrix_fragment_73 = permute_signed(matrix_fragment_75, 0u, 0x7710u);

            matrix_fragment_75 = (index_word_117 << 0x18u) + index_word_120;

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
            condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_75,
                                            async_tokens[(((matrix_fragment_75) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
            if (!condition_0) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_0 =
                        wait_copy_barrier(shared_tile, matrix_fragment_75,
                                          async_tokens[(((matrix_fragment_75) >> 3) & 1)]);
#else
                    __syncthreads();
                    condition_0 = true;
#endif
                } while (!condition_0);
            }
        }
        matrix_fragment_76 = blockIdx.y;

        condition_0 = uint32_t(index_word_116) > uint32_t(0x13fu);

        index_word_121 = index_word_121 & 0xffu;

        matrix_fragment_76 = matrix_fragment_76 * 0x8u;
        if (!condition_0) {
            matrix_fragment_72 = index_word_116 + 0xc0u;

            condition_0 = (matrix_fragment_72 & 0x40u) != 0;
            if (!condition_0) {
                matrix_fragment_74 = threadIdx.x;

                uniform_address_word_4 = parameters[parameter_scalar_control_17];

                matrix_fragment_73 = threadIdx.z;

                matrix_fragment_72 = matrix_fragment_76 + (0u - uniform_address_word_4);

                matrix_fragment_72 = max(int32_t(0u), int32_t(matrix_fragment_72));

                condition_0 = (matrix_fragment_73 | matrix_fragment_74 | thread_row_113) != 0;

                matrix_fragment_74 = uint32_t(uint32_t(matrix_fragment_72) >> 0x3u);

                condition_0 = int32_t(matrix_fragment_74) >= int32_t(index_word_115) || condition_0;
                if (!condition_0) {
                    do {
                        matrix_fragment_72 = parameters[parameter_optional_control_16];

                        matrix_fragment_72 = index_word_111 * 0x8u + (0u - matrix_fragment_72);

                        matrix_fragment_75 = matrix_fragment_72 + 0x8u;

                        matrix_fragment_73 = matrix_fragment_72 + 0x7u;

                        matrix_fragment_72 = max(int32_t(0u), int32_t(matrix_fragment_72));

                        condition_0 = int32_t(matrix_fragment_75) > int32_t(0u);

                        matrix_fragment_73 =
                            uint32_t(combine_words(matrix_fragment_73, 0u) << 0x1du >> 32) + 0x1u;

                        matrix_fragment_75 = uint32_t(uint32_t(matrix_fragment_72) >> 0x3u);

                        matrix_fragment_73 = condition_0 ? matrix_fragment_73 : 0u;

                        matrix_fragment_80 =
                            min(int32_t(matrix_fragment_73), int32_t(index_word_114));

                        condition_0 = int32_t(matrix_fragment_75) >= int32_t(matrix_fragment_80);
                        if (!condition_0) {
                            matrix_fragment_73 = blockIdx.z;

                            matrix_fragment_72 = matrix_fragment_73 * 0x200u + index_word_116;

                            matrix_fragment_72 = matrix_fragment_72 + 0xc0u;

                            matrix_fragment_73 = uint32_t(uint32_t(matrix_fragment_72) >> 0x7u);

                            matrix_fragment_73 = matrix_fragment_74 * 0x4u + matrix_fragment_73;

                            matrix_fragment_78 = matrix_fragment_73 * index_word_114;
                            do {
                                matrix_fragment_72 = parameters[parameter_wait_low_14];
                                matrix_fragment_73 = parameters[parameter_wait_high_15];

                                matrix_fragment_77 = matrix_fragment_78 + matrix_fragment_75;

                                split_address_words(
                                    matrix_fragment_72, matrix_fragment_73,
                                    int64_t(int32_t(matrix_fragment_77)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_72, matrix_fragment_73));

                                matrix_fragment_77 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_72, matrix_fragment_73),
                                    memory_ranges, error_counts);

                                matrix_fragment_75 = matrix_fragment_75 + 0x1u;

                                condition_1 =
                                    uint32_t(matrix_fragment_75) != uint32_t(matrix_fragment_80);

                                condition_0 = int32_t(matrix_fragment_77) > -1;
                                if (!condition_0) {
                                    do {
                                        __nanosleep(0x40u);

                                        matrix_fragment_77 = pipeline_read<CheckBounds>(
                                            combine_words(matrix_fragment_72, matrix_fragment_73),
                                            memory_ranges, error_counts);

                                        condition_0 = int32_t(matrix_fragment_77) > -1;
                                    } while (!condition_0);
                                }
                            } while (condition_1);
                        }
                        matrix_fragment_74 = matrix_fragment_74 + 0x1u;

                        condition_0 = uint32_t(matrix_fragment_74) != uint32_t(index_word_115);
                    } while (condition_0);
                }
                __syncwarp(__activemask());

                __syncthreads();
            }
            matrix_fragment_80 = blockIdx.y;

            matrix_fragment_75 = parameters[parameter_tensor_height_8];

            thread_row_113 = threadIdx.y;

            matrix_fragment_74 = parameters[parameter_tensor_width_9];

            matrix_fragment_72 = parameters[parameter_tensor_height_8];
            matrix_fragment_73 = parameters[parameter_tensor_width_9];

            matrix_fragment_78 = uint32_t(int32_t(matrix_fragment_75) >> 0x1fu);

            matrix_fragment_78 =
                uint32_t(combine_words(matrix_fragment_78, 0u) << 0x2u >> 32) + matrix_fragment_75;

            matrix_fragment_80 = matrix_fragment_80 * 0x2u;

            matrix_fragment_78 = uint32_t(int32_t(matrix_fragment_78) >> 0x2u);

            matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_74) >> 0x1fu);

            matrix_fragment_81 =
                uint32_t(combine_words(thread_row_113, 0u) << 0x1eu >> 32) + matrix_fragment_80;

            matrix_fragment_75 =
                uint32_t(combine_words(matrix_fragment_75, 0u) << 0x2u >> 32) + matrix_fragment_74;

            matrix_fragment_72 = matrix_fragment_72 & 0xfffffffcu;

            matrix_fragment_79 = matrix_fragment_73 & 0xfffffffcu;

            condition_0 = int32_t(matrix_fragment_81) >= int32_t(matrix_fragment_78) && condition_6;

            condition_2 = uint32_t(matrix_fragment_72) != uint32_t(0x4u);

            matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_75) >> 0x2u);

            split_address_words(matrix_fragment_74, matrix_fragment_75, 0ull);

            condition_1 = uint32_t(matrix_fragment_79) == uint32_t(0x4u) || condition_0;

            condition_0 = int32_t(matrix_fragment_81) < int32_t(matrix_fragment_78) || !condition_2;

            condition_1 = int32_t(index_word_0) < int32_t(matrix_fragment_77) || condition_1;

            condition_0 = condition_1 & condition_0;

            if (condition_0) {
                matrix_fragment_72 = parameters[parameter_input_low_0];
                matrix_fragment_73 = parameters[parameter_input_high_1];

                matrix_fragment_83 = blockIdx.z;

                condition_1 = int32_t(matrix_fragment_81) >= int32_t(matrix_fragment_78);

                matrix_fragment_82 = thread_row_113 & 0x1u;

                matrix_fragment_74 = condition_1 ? index_word_0 : 0u;

                condition_1 = uint32_t(matrix_fragment_79) != uint32_t(0x4u);

                matrix_fragment_75 = condition_6 ? matrix_fragment_74 : 0u;

                matrix_fragment_74 = !condition_1 ? matrix_fragment_75 : index_word_0;

                if (condition_2) {
                    matrix_fragment_74 =
                        matrix_fragment_81 * matrix_fragment_77 + matrix_fragment_74;
                }

                matrix_fragment_83 = matrix_fragment_83 * 0x200u + index_word_116;

                matrix_fragment_75 = matrix_fragment_83 + 0xc0u;

                matrix_fragment_75 =
                    uint32_t(combine_words(matrix_fragment_75, 0u) << 0x1bu >> 32) +
                    matrix_fragment_82;

                matrix_fragment_74 = matrix_fragment_74 * 0x10u + matrix_fragment_75;

                matrix_fragment_74 = matrix_fragment_74 * 0x80u;

                matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_74) >> 0x1fu);
            }
            uniform_index_word_5 = 0u;

            uniform_address_word_4 = 0x400u;

            uint64_t byte_address_6;
            if (condition_0) {
                byte_address_6 =
                    scaled_word_address(combine_words(matrix_fragment_72, matrix_fragment_73),
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
                matrix_fragment_81 = uint32_t(byte_address_6);
            }

            uniform_address_word_4 = uniform_address_word_4 + 0x3000u;

            if (condition_0) {
                matrix_fragment_74 = uint32_t(byte_address_6 >> 32);
            }

            split_address_words(matrix_fragment_72, matrix_fragment_73, 0ull);

            if (condition_0) {
                matrix_fragment_72 = matrix_fragment_81;

                matrix_fragment_73 = matrix_fragment_74;
            }

            uniform_address_word_4 = (uniform_index_word_5 << 0x18u) + uniform_address_word_4;

            index_word_121 = (index_word_121 << 0x3u) + uniform_address_word_4;
            if (!condition_0) {
                matrix_fragment_81 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_72 = thread_row_113 * 0x200u + index_word_118;

                matrix_fragment_73 = uniform_address_word_4;

                matrix_fragment_74 = uniform_address_word_4;

                matrix_fragment_81 = matrix_fragment_81 * 0x10u + matrix_fragment_72;

                matrix_fragment_72 =
                    permute_signed(matrix_fragment_73, matrix_fragment_74, 0x5410u);

                matrix_fragment_73 = matrix_fragment_72;

                matrix_fragment_74 = matrix_fragment_72;

                matrix_fragment_75 = matrix_fragment_72;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_81,
                                                     matrix_fragment_72, matrix_fragment_73,
                                                     matrix_fragment_74, matrix_fragment_75,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    uniform_address_word_4 = broadcast_active_lane(thread_row_113);

                    uniform_address_word_6 = broadcast_active_lane(index_word_118);

                    uniform_address_word_12 = broadcast_active_lane(matrix_fragment_72);

                    uniform_address_word_13 = broadcast_active_lane(matrix_fragment_73);

                    uniform_index_word_5 = broadcast_active_lane(index_word_121);

                    matrix_fragment_72 = 0x200u;

                    uniform_address_word_4 =
                        (uniform_address_word_4 << 0x9u) + uniform_address_word_6;

                    uniform_address_word_6 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_12, uniform_address_word_13),
                        uniform_address_word_6 * 16u, index_word_121, shared_bytes, memory_ranges,
                        error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_12, uniform_address_word_13),
                        uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            matrix_fragment_73 = thread_row_113 + 0x4u;

            split_address_words(matrix_fragment_74, matrix_fragment_75, 0ull);

            matrix_fragment_73 =
                uint32_t(combine_words(matrix_fragment_73, 0u) << 0x1eu >> 32) + matrix_fragment_80;

            condition_3 = int32_t(matrix_fragment_73) >= int32_t(matrix_fragment_78) && condition_6;

            condition_0 = int32_t(matrix_fragment_73) < int32_t(matrix_fragment_78) || !condition_2;

            condition_1 = uint32_t(matrix_fragment_79) == uint32_t(0x4u) || condition_3;

            condition_1 = int32_t(index_word_0) < int32_t(matrix_fragment_77) || condition_1;

            condition_0 = condition_1 & condition_0;

            condition_1 = uint32_t(matrix_fragment_79) != uint32_t(0x4u);

            matrix_fragment_79 = condition_3 ? index_word_0 : 0u;

            matrix_fragment_72 = !condition_1 ? matrix_fragment_79 : index_word_0;
            if (condition_0) {
                matrix_fragment_75 = blockIdx.z;

                matrix_fragment_74 = thread_row_113 & 0x1u;

                if (condition_2) {
                    matrix_fragment_72 =
                        matrix_fragment_73 * matrix_fragment_77 + matrix_fragment_72;
                }

                matrix_fragment_75 = matrix_fragment_75 * 0x200u + index_word_116;

                matrix_fragment_75 = matrix_fragment_75 + 0xc0u;

                matrix_fragment_75 =
                    uint32_t(combine_words(matrix_fragment_75, 0u) << 0x1bu >> 32) +
                    matrix_fragment_74;

                matrix_fragment_74 = matrix_fragment_72 * 0x10u + matrix_fragment_75;

                matrix_fragment_74 = matrix_fragment_74 * 0x80u;

                matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_74) >> 0x1fu);
            }
            if (!condition_0) {
                matrix_fragment_77 = threadIdx.x & 31u;

                matrix_fragment_72 = thread_row_113 * 0x200u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_73 = matrix_fragment_72 + 0x800u;

                matrix_fragment_75 = uniform_address_word_4;

                matrix_fragment_73 = matrix_fragment_73 & 0xff800u;

                matrix_fragment_73 = matrix_fragment_73 | 0x400u & matrix_fragment_72;

                matrix_fragment_73 = matrix_fragment_73 | 0x200u & matrix_fragment_72;

                matrix_fragment_72 = uniform_address_word_4;

                index_word_118 = index_word_118 + matrix_fragment_73;

                matrix_fragment_72 =
                    permute_signed(matrix_fragment_72, matrix_fragment_75, 0x5410u);

                matrix_fragment_73 = matrix_fragment_72;

                matrix_fragment_74 = matrix_fragment_72;

                matrix_fragment_77 = matrix_fragment_77 * 0x10u + index_word_118;

                matrix_fragment_75 = matrix_fragment_72;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_77,
                                                     matrix_fragment_72, matrix_fragment_73,
                                                     matrix_fragment_74, matrix_fragment_75,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (condition_0) {
                    matrix_fragment_78 = parameters[parameter_input_low_0];
                    matrix_fragment_79 = parameters[parameter_input_high_1];
                }

                uniform_address_word_4 = 0xffffffffu;

                split_address_words(matrix_fragment_72, matrix_fragment_73, 0ull);

                if (condition_0) {
                    const uint64_t byte_address_7 =
                        scaled_word_address(combine_words(matrix_fragment_78, matrix_fragment_79),
                                            combine_words(matrix_fragment_74, matrix_fragment_75));
                    matrix_fragment_77 = uint32_t(byte_address_7);

                    matrix_fragment_73 = uint32_t(byte_address_7 >> 32);

                    matrix_fragment_72 = matrix_fragment_77;
                }

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    uniform_address_word_4 = broadcast_active_lane(thread_row_113);

                    uniform_address_word_6 = broadcast_active_lane(index_word_118);

                    uniform_address_word_12 = broadcast_active_lane(matrix_fragment_72);

                    uniform_address_word_13 = broadcast_active_lane(matrix_fragment_73);

                    uniform_address_word_7 = broadcast_active_lane(index_word_121);

                    matrix_fragment_72 = 0x200u;

                    uniform_address_word_4 = uniform_address_word_4 << (0x9u & 31u);

                    uniform_index_word_5 = uniform_address_word_4 + 0x800u;

                    uniform_index_word_5 = uniform_index_word_5 & 0xff800u;

                    uniform_index_word_5 = uniform_index_word_5 | 0x400u & uniform_address_word_4;

                    uniform_index_word_5 = uniform_index_word_5 | 0x200u & uniform_address_word_4;

                    uniform_address_word_4 = 0x20u;

                    uniform_address_word_6 = uniform_address_word_6 + uniform_index_word_5;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_12, uniform_address_word_13),
                        uniform_address_word_4 * 16u, index_word_121, shared_bytes, memory_ranges,
                        error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_12, uniform_address_word_13),
                        uniform_address_word_4 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_0 = uint32_t(index_word_116) >= uint32_t(0x1c0u);

        index_word_116 = index_word_116 + 0x40u;
    } while (!condition_0);
    index_word_0 = threadIdx.x & 31u;

    matrix_fragment_40 = parameters[parameter_tensor_height_8];
    matrix_fragment_41 = parameters[parameter_tensor_width_9];

    matrix_fragment_42 = index_word_0;

    matrix_fragment_45 = index_word_0;

    matrix_fragment_43 = uint32_t(int32_t(matrix_fragment_42) >> 0x2u);

    matrix_fragment_45 =
        uint32_t(int64_t(int32_t(matrix_fragment_45)) * (1ll << 0x1cu) >> 32) + matrix_fragment_76;

    matrix_fragment_44 =
        uint32_t(combine_words(matrix_fragment_43, 0u) << 0x2u >> 32) + matrix_fragment_43;

    condition_0 = int32_t(matrix_fragment_45) >= int32_t(matrix_fragment_40);

    matrix_fragment_44 = matrix_fragment_44 & 0xfffffffcu;

    matrix_fragment_47 = matrix_fragment_45 + 0x2u;

    condition_0 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_0;

    matrix_fragment_44 = matrix_fragment_43 + (0u - matrix_fragment_44);

    matrix_fragment_43 = address_word_112 * 0x4u + thread_row_113;

    condition_1 = int32_t(matrix_fragment_47) >= int32_t(matrix_fragment_40);

    matrix_fragment_46 = index_word_111 * 0x8u + matrix_fragment_44;

    condition_1 = int32_t(matrix_fragment_47) < int32_t(0u) || condition_1;

    matrix_fragment_43 = matrix_fragment_43 * 0x4u;

    condition_6 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_3 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_4 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    matrix_fragment_44 = matrix_fragment_41 * 0x4u;

    condition_6 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_6;

    condition_5 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_5;

    condition_3 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_3;

    condition_4 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_4;

    condition_2 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_6 = condition_0 | condition_6;

    condition_5 = condition_1 | condition_5;

    condition_3 = condition_0 | condition_3;

    matrix_fragment_52 = uint32_t(condition_5) << 5;

    condition_4 = condition_1 | condition_4;

    condition_5 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);
    if (!condition_6) {
        matrix_fragment_48 = parameters[parameter_output_low_4];
        matrix_fragment_49 = parameters[parameter_output_high_5];

        matrix_fragment_51 = matrix_fragment_46 * 0x4u;

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = matrix_fragment_43 * matrix_fragment_40 + matrix_fragment_45;

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_50 = matrix_fragment_44 * matrix_fragment_50 + matrix_fragment_51;

        matrix_fragment_51 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_51 = matrix_fragment_50 + index_word_0 + (0u - matrix_fragment_51);

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_48, matrix_fragment_49),
                                    matrix_fragment_53, memory_ranges, error_counts);
    }
    condition_6 = int32_t(matrix_fragment_52) != int32_t(0u);

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_6) {
        matrix_accumulator_28 = parameters[parameter_output_low_4];
        matrix_accumulator_29 = parameters[parameter_output_high_5];

        matrix_accumulator_35 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_34 = matrix_fragment_43 * matrix_fragment_40 + matrix_fragment_47;

        matrix_accumulator_35 = matrix_fragment_44 * matrix_accumulator_34 + matrix_accumulator_35;

        matrix_accumulator_34 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_35 = matrix_accumulator_35 + index_word_0 + (0u - matrix_accumulator_34);

        split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                            int64_t(int32_t(matrix_accumulator_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_28, matrix_accumulator_29));

        matrix_accumulator_35 = permute_signed(matrix_fragment_48, matrix_fragment_49, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_28, matrix_accumulator_29),
                                    matrix_accumulator_35, memory_ranges, error_counts);
    }
    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_29 = matrix_fragment_43 + 0x1u;
    if (!condition_3) {
        matrix_accumulator_34 = parameters[parameter_output_low_4];
        matrix_accumulator_35 = parameters[parameter_output_high_5];

        matrix_fragment_49 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_16 = matrix_accumulator_29 * matrix_fragment_40 + matrix_fragment_45;

        matrix_fragment_49 = matrix_fragment_44 * matrix_accumulator_16 + matrix_fragment_49;

        matrix_accumulator_16 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_49 = matrix_fragment_49 + index_word_0 + (0u - matrix_accumulator_16);

        split_address_words(matrix_accumulator_34, matrix_accumulator_35,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_34, matrix_accumulator_35));

        matrix_fragment_49 = permute_signed(matrix_accumulator_22, matrix_fragment_51, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_34, matrix_accumulator_35),
                                    matrix_fragment_49, memory_ranges, error_counts);
    }
    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        matrix_accumulator_16 = parameters[parameter_output_low_4];
        matrix_accumulator_17 = parameters[parameter_output_high_5];

        matrix_accumulator_23 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_22 = matrix_accumulator_29 * matrix_fragment_40 + matrix_fragment_47;

        matrix_accumulator_23 = matrix_fragment_44 * matrix_accumulator_22 + matrix_accumulator_23;

        matrix_accumulator_22 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_23 = matrix_accumulator_23 + index_word_0 + (0u - matrix_accumulator_22);

        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            int64_t(int32_t(matrix_accumulator_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, matrix_accumulator_17));

        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_28, matrix_accumulator_35, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_16, matrix_accumulator_17),
                                    matrix_accumulator_23, memory_ranges, error_counts);
    }
    condition_2 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_2;

    matrix_accumulator_22 = matrix_accumulator_22 & 0xfffffffdu;

    condition_4 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_3 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_2 = condition_0 | condition_2;

    condition_5 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_5;

    condition_4 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_4;

    condition_3 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_3;

    matrix_accumulator_16 = matrix_fragment_46 + 0x4u;

    matrix_accumulator_17 = matrix_fragment_45 + 0x4u;

    matrix_accumulator_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                      select_half_operand<2, false, false>(0u));

    condition_4 = condition_1 | condition_4;

    condition_3 = condition_1 | condition_3;

    if (condition_2) {
        matrix_accumulator_22 = 0x2u | matrix_accumulator_22;
    }

    condition_2 = condition_0 | condition_5;

    condition_5 = (matrix_accumulator_22 & 0x2u) != 0;

    condition_6 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_0 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_1 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    matrix_accumulator_22 = matrix_accumulator_22 & 0xfffffffeu;

    condition_6 = int32_t(matrix_fragment_45) >= int32_t(matrix_fragment_40) || condition_6;

    condition_0 = int32_t(matrix_fragment_47) >= int32_t(matrix_fragment_40) || condition_0;

    condition_1 = int32_t(matrix_fragment_45) >= int32_t(matrix_fragment_40) || condition_1;

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_4),
                                       select_half_operand<2, false, false>(0u));

    if (condition_2) {
        matrix_accumulator_22 = 0x1u | matrix_accumulator_22;
    }

    matrix_accumulator_23 = matrix_fragment_43 + 0x2u;

    condition_6 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_6;

    condition_0 = int32_t(matrix_fragment_47) < int32_t(0u) || condition_0;

    condition_1 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_1;

    condition_2 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_40);
    if (!condition_5) {
        matrix_accumulator_34 = parameters[parameter_output_low_4];
        matrix_accumulator_35 = parameters[parameter_output_high_5];

        matrix_fragment_49 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_23 * matrix_fragment_40 + matrix_fragment_45;

        matrix_fragment_49 = matrix_fragment_44 * matrix_accumulator_4 + matrix_fragment_49;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_49 = matrix_fragment_49 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(matrix_accumulator_34, matrix_accumulator_35,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_34, matrix_accumulator_35));

        matrix_fragment_49 = permute_signed(matrix_accumulator_10, matrix_fragment_51, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_34, matrix_accumulator_35),
                                    matrix_fragment_49, memory_ranges, error_counts);
    }
    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_11),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_5),
                      select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        matrix_accumulator_4 = parameters[parameter_output_low_4];
        matrix_accumulator_5 = parameters[parameter_output_high_5];

        address_word_11 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_10 = matrix_accumulator_23 * matrix_fragment_40 + matrix_fragment_47;

        address_word_11 = matrix_fragment_44 * matrix_accumulator_10 + address_word_11;

        matrix_accumulator_10 = matrix_fragment_42 & 0xfffffffcu;

        address_word_11 = address_word_11 + index_word_0 + (0u - matrix_accumulator_10);

        split_address_words(matrix_accumulator_4, matrix_accumulator_5,
                            int64_t(int32_t(address_word_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_4, matrix_accumulator_5));

        address_word_11 = permute_signed(matrix_accumulator_28, matrix_accumulator_35, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_4, matrix_accumulator_5),
                                    address_word_11, memory_ranges, error_counts);
    }
    condition_4 = (matrix_accumulator_22 & 0x1u) != 0;

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_5 = matrix_fragment_43 + 0x3u;
    if (!condition_4) {
        matrix_accumulator_10 = parameters[parameter_output_low_4];
        address_word_11 = parameters[parameter_output_high_5];

        matrix_accumulator_35 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_5 * matrix_fragment_40 + matrix_fragment_45;

        matrix_accumulator_35 = matrix_fragment_44 * matrix_accumulator_4 + matrix_accumulator_35;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_35 = matrix_accumulator_35 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));

        matrix_accumulator_35 = permute_signed(matrix_accumulator_38, matrix_fragment_49, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_10, address_word_11),
                                    matrix_accumulator_35, memory_ranges, error_counts);
    }
    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        matrix_accumulator_10 = parameters[parameter_output_low_4];
        address_word_11 = parameters[parameter_output_high_5];

        matrix_accumulator_35 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_5 * matrix_fragment_40 + matrix_fragment_47;

        matrix_accumulator_35 = matrix_fragment_44 * matrix_accumulator_4 + matrix_accumulator_35;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_35 = matrix_accumulator_35 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));

        matrix_accumulator_35 =
            permute_signed(matrix_accumulator_39, matrix_accumulator_28, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_10, address_word_11),
                                    matrix_accumulator_35, memory_ranges, error_counts);
    }
    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(0u));
    if (!condition_6) {
        matrix_accumulator_10 = parameters[parameter_output_low_4];
        address_word_11 = parameters[parameter_output_high_5];

        matrix_accumulator_35 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_43 * matrix_fragment_40 + matrix_fragment_45;

        matrix_accumulator_35 = matrix_fragment_44 * matrix_accumulator_4 + matrix_accumulator_35;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_35 = matrix_accumulator_35 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));

        matrix_accumulator_35 =
            permute_signed(matrix_accumulator_32, matrix_accumulator_39, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_10, address_word_11),
                                    matrix_accumulator_35, memory_ranges, error_counts);
    }
    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_4 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        matrix_accumulator_10 = parameters[parameter_output_low_4];
        address_word_11 = parameters[parameter_output_high_5];

        matrix_accumulator_27 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_26 = matrix_fragment_43 * matrix_fragment_40 + matrix_fragment_47;

        matrix_accumulator_27 = matrix_fragment_44 * matrix_accumulator_26 + matrix_accumulator_27;

        matrix_accumulator_26 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_27 = matrix_accumulator_27 + index_word_0 + (0u - matrix_accumulator_26);

        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));

        matrix_accumulator_27 =
            permute_signed(matrix_accumulator_33, matrix_accumulator_4, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_10, address_word_11),
                                    matrix_accumulator_27, memory_ranges, error_counts);
    }
    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_4 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_14),
                      select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        matrix_accumulator_10 = parameters[parameter_output_low_4];
        address_word_11 = parameters[parameter_output_high_5];

        matrix_accumulator_21 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_14 = matrix_accumulator_29 * matrix_fragment_40 + matrix_fragment_45;

        matrix_accumulator_21 = matrix_fragment_44 * matrix_accumulator_14 + matrix_accumulator_21;

        matrix_accumulator_14 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_21 = matrix_accumulator_21 + index_word_0 + (0u - matrix_accumulator_14);

        split_address_words(matrix_accumulator_10, address_word_11,
                            int64_t(int32_t(matrix_accumulator_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_10, address_word_11));

        matrix_accumulator_21 =
            permute_signed(matrix_accumulator_20, matrix_accumulator_27, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_10, address_word_11),
                                    matrix_accumulator_21, memory_ranges, error_counts);
    }
    condition_6 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_0 = int32_t(matrix_accumulator_17) < int32_t(0u) || condition_2;

    condition_1 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_2 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_6 = int32_t(matrix_fragment_47) >= int32_t(matrix_fragment_40) || condition_6;

    condition_5 = int32_t(matrix_fragment_45) >= int32_t(matrix_fragment_40) || condition_5;

    condition_3 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_4 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    address_word_11 = matrix_fragment_45 + 0x6u;

    condition_6 = int32_t(matrix_fragment_47) < int32_t(0u) || condition_6;

    condition_5 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_5;

    condition_1 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_1;

    condition_2 = int32_t(matrix_fragment_47) >= int32_t(matrix_fragment_40) || condition_2;

    condition_3 = int32_t(matrix_fragment_45) >= int32_t(matrix_fragment_40) || condition_3;

    condition_4 = int32_t(matrix_fragment_47) >= int32_t(matrix_fragment_40) || condition_4;

    matrix_accumulator_20 = uint32_t(condition_5) << 5;

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                      select_half_operand<2, false, false>(0u));

    condition_1 = condition_0 | condition_1;

    condition_2 = int32_t(matrix_fragment_47) < int32_t(0u) || condition_2;

    condition_3 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_3;

    condition_4 = int32_t(matrix_fragment_47) < int32_t(0u) || condition_4;

    condition_5 = int32_t(address_word_11) >= int32_t(matrix_fragment_40);
    if (!condition_6) {
        matrix_accumulator_14 = parameters[parameter_output_low_4];
        matrix_accumulator_15 = parameters[parameter_output_high_5];

        matrix_accumulator_21 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_10 = matrix_accumulator_29 * matrix_fragment_40 + matrix_fragment_47;

        matrix_accumulator_21 = matrix_fragment_44 * matrix_accumulator_10 + matrix_accumulator_21;

        matrix_accumulator_10 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_21 = matrix_accumulator_21 + index_word_0 + (0u - matrix_accumulator_10);

        split_address_words(matrix_accumulator_14, matrix_accumulator_15,
                            int64_t(int32_t(matrix_accumulator_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));

        matrix_accumulator_21 =
            permute_signed(matrix_accumulator_4, matrix_accumulator_27, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_14, matrix_accumulator_15),
                                    matrix_accumulator_21, memory_ranges, error_counts);
    }
    condition_6 = int32_t(matrix_accumulator_20) != int32_t(0u);

    matrix_accumulator_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_2),
                                          select_half_operand<2, false, false>(0u));
    if (!condition_6) {
        matrix_accumulator_14 = parameters[parameter_output_low_4];
        matrix_accumulator_15 = parameters[parameter_output_high_5];

        matrix_accumulator_21 = matrix_accumulator_16 * 0x4u;

        address_word_2 = matrix_accumulator_23 * matrix_fragment_40 + matrix_fragment_45;

        matrix_accumulator_21 = matrix_fragment_44 * address_word_2 + matrix_accumulator_21;

        address_word_2 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_21 = matrix_accumulator_21 + index_word_0 + (0u - address_word_2);

        split_address_words(matrix_accumulator_14, matrix_accumulator_15,
                            int64_t(int32_t(matrix_accumulator_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));

        matrix_accumulator_21 =
            permute_signed(matrix_accumulator_8, matrix_accumulator_27, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_14, matrix_accumulator_15),
                                    matrix_accumulator_21, memory_ranges, error_counts);
    }
    matrix_accumulator_4 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_9),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_3),
                                          select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_8 = matrix_accumulator_23 * matrix_fragment_40 + matrix_fragment_47;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_8 + address_word_9;

        matrix_accumulator_8 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_8);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_4, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_45 = matrix_accumulator_5 * matrix_fragment_40 + matrix_fragment_45;

        address_word_9 = matrix_fragment_44 * matrix_fragment_45 + address_word_9;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_84, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_4 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_8 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_47 = matrix_accumulator_5 * matrix_fragment_40 + matrix_fragment_47;

        address_word_9 = matrix_fragment_44 * matrix_fragment_47 + address_word_9;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_8);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_85, matrix_accumulator_4, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_43 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_102, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    condition_6 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_4 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(address_word_11) < int32_t(0u) || condition_5;

    condition_1 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_3 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_6 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_6;

    condition_4 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_4;

    condition_1 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_1;

    condition_3 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_3;

    condition_6 = condition_0 | condition_6;

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_10 = condition_6 ? 0x40u : 0u;

    condition_6 = condition_5 | condition_4;

    condition_1 = condition_0 | condition_1;

    condition_3 = condition_0 | condition_3;

    condition_2 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_0 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);

    condition_4 = int32_t(matrix_fragment_46) >= int32_t(matrix_fragment_41);
    if (!condition_6) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_43 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_103, matrix_accumulator_8, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    condition_2 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_2;

    condition_0 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_0;

    condition_4 = int32_t(matrix_fragment_46) < int32_t(0u) || condition_4;

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    condition_2 = condition_5 | condition_2;

    condition_0 = condition_5 | condition_0;

    condition_4 = condition_5 | condition_4;

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                      select_half_operand<2, false, false>(0u));

    condition_6 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);
    if (!condition_1) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_29 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_92, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_93 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_29 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_93, matrix_accumulator_8, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    condition_1 = int32_t(matrix_accumulator_10) != int32_t(0u);

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                      select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_23 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_96, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                      select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_23 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_97, matrix_accumulator_8, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_5 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_100, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_fragment_46 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_5 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_101, matrix_accumulator_8, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    condition_6 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_40) || condition_6;

    condition_5 = int32_t(address_word_11) >= int32_t(matrix_fragment_40) || condition_5;

    matrix_accumulator_22 = matrix_accumulator_22 & 0xfffffffdu;

    condition_0 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    address_word_3 = condition_6 ? 0x40u : 0u;

    address_word_2 = uint32_t(condition_5) << 5;

    condition_1 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(address_word_3) != int32_t(0u);

    condition_2 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_3 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_4 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_6 = int32_t(matrix_accumulator_16) >= int32_t(matrix_fragment_41);

    condition_5 = int32_t(matrix_accumulator_17) < int32_t(0u) || condition_5;

    condition_0 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_40) || condition_0;

    condition_1 = int32_t(address_word_11) >= int32_t(matrix_fragment_40) || condition_1;

    condition_2 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_40) || condition_2;

    condition_3 = int32_t(address_word_11) >= int32_t(matrix_fragment_40) || condition_3;

    condition_4 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_40) || condition_4;

    condition_6 = int32_t(address_word_11) >= int32_t(matrix_fragment_40) || condition_6;

    condition_0 = int32_t(matrix_accumulator_17) < int32_t(0u) || condition_0;

    condition_1 = int32_t(address_word_11) < int32_t(0u) || condition_1;

    condition_2 = int32_t(matrix_accumulator_17) < int32_t(0u) || condition_2;

    condition_3 = int32_t(address_word_11) < int32_t(0u) || condition_3;

    condition_4 = int32_t(matrix_accumulator_17) < int32_t(0u) || condition_4;

    condition_6 = int32_t(address_word_11) < int32_t(0u) || condition_6;

    if (condition_5) {
        matrix_accumulator_22 = 0x2u | matrix_accumulator_22;
    }

    condition_5 = int32_t(address_word_2) != int32_t(0u);

    matrix_accumulator_30 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = matrix_accumulator_22 & 0xfffffffeu;

    condition_5 = int32_t(address_word_11) < int32_t(0u) || condition_5;

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(0u));

    if (condition_5) {
        matrix_accumulator_22 = 0x1u | matrix_accumulator_22;
    }

    condition_5 = (matrix_accumulator_22 & 0x2u) != 0;
    if (!condition_5) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_43 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_30, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    condition_5 = (matrix_accumulator_22 & 0x1u) != 0;

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_4 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));
    if (!condition_5) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_8 = matrix_fragment_42 & 0xfffffffcu;

        matrix_fragment_43 = matrix_fragment_43 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_fragment_43 + address_word_9;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_8);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_31, matrix_accumulator_4, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                      select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_29 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_18, matrix_accumulator_15, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_4 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                      select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_8 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_29 = matrix_accumulator_29 * matrix_fragment_40 + address_word_11;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_29 + address_word_9;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_8);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_19, matrix_accumulator_4, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_6),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                      select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        address_word_9 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_accumulator_23 * matrix_fragment_40 + matrix_accumulator_17;

        address_word_9 = matrix_fragment_44 * matrix_accumulator_4 + address_word_9;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        address_word_9 = address_word_9 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_9 = permute_signed(matrix_accumulator_6, matrix_accumulator_13, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), address_word_9,
                                    memory_ranges, error_counts);
    }
    matrix_accumulator_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_37 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                      select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        matrix_accumulator_7 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_6 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_23 = matrix_accumulator_23 * matrix_fragment_40 + address_word_11;

        matrix_accumulator_7 = matrix_fragment_44 * matrix_accumulator_23 + matrix_accumulator_7;

        matrix_accumulator_7 = matrix_accumulator_7 + index_word_0 + (0u - matrix_accumulator_6);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_7 = permute_signed(matrix_accumulator_4, matrix_accumulator_37, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3),
                                    matrix_accumulator_7, memory_ranges, error_counts);
    }
    matrix_accumulator_86 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                      select_half_operand<2, false, false>(0u));

    address_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                   select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        address_word_2 = parameters[parameter_output_low_4];
        address_word_3 = parameters[parameter_output_high_5];

        matrix_accumulator_7 = matrix_accumulator_16 * 0x4u;

        matrix_accumulator_4 = matrix_fragment_42 & 0xfffffffcu;

        matrix_accumulator_17 = matrix_accumulator_5 * matrix_fragment_40 + matrix_accumulator_17;

        matrix_accumulator_7 = matrix_fragment_44 * matrix_accumulator_17 + matrix_accumulator_7;

        matrix_accumulator_7 = matrix_accumulator_7 + index_word_0 + (0u - matrix_accumulator_4);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_7 = permute_signed(matrix_accumulator_86, address_word_9, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3),
                                    matrix_accumulator_7, memory_ranges, error_counts);
    }
    if (condition_6) {
        return;
    }

    address_word_2 = parameters[parameter_output_low_4];
    address_word_3 = parameters[parameter_output_high_5];

    matrix_accumulator_7 = matrix_accumulator_16 * 0x4u;

    matrix_fragment_42 = matrix_fragment_42 & 0xfffffffcu;

    matrix_accumulator_5 = matrix_accumulator_5 * matrix_fragment_40 + address_word_11;

    matrix_accumulator_7 = matrix_fragment_44 * matrix_accumulator_5 + matrix_accumulator_7;

    matrix_accumulator_5 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 = matrix_accumulator_7 + index_word_0 + (0u - matrix_fragment_42);

    index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                 select_half_operand<2, false, false>(0u));

    matrix_accumulator_5 = permute_signed(matrix_accumulator_5, index_word_0, 0x5410u);

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(matrix_accumulator_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), matrix_accumulator_5,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(212) void phoenix_global_attention_split_heads_512_projection_output_view_wait(
    KernelArguments parameters) {
    run_global_attention_split_heads_512_projection_output_view_wait<false>(parameters.words,
                                                                            nullptr, nullptr);
}
