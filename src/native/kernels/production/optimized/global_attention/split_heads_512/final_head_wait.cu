// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/final_head_wait.cu
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
    parameter_output_low_2 = 2,
    parameter_output_high_3 = 3,
    parameter_trained_weights_low_4 = 4,
    parameter_trained_weights_high_5 = 5,
    parameter_wait_low_6 = 6,
    parameter_wait_high_7 = 7,
    parameter_LH_8 = 8,
    parameter_LW_9 = 9,
};

struct alignas(8) KernelArguments {
    uint32_t words[10];
};
static_assert(sizeof(KernelArguments) == 40, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_final_head_wait(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t address_word_4 = 0;
    uint32_t address_word_5 = 0;
    uint32_t packed_half_6 = 0;
    uint32_t packed_half_7 = 0;
    uint32_t index_word_8 = 0;
    uint32_t index_word_9 = 0;
    uint32_t address_word_10 = 0;
    uint32_t block_index_11 = 0;
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
    uint32_t index_word_88 = 0;
    uint32_t index_word_89 = 0;
    uint32_t reused_word_90 = 0;
    uint32_t index_word_91 = 0;
    uint32_t reused_word_92 = 0;
    uint32_t index_word_93 = 0;
    uint32_t index_word_94 = 0;
    uint32_t index_word_95 = 0;
    uint32_t index_word_96 = 0;
    uint32_t index_word_97 = 0;
    uint32_t warp_lane_98 = 0;
    uint32_t index_word_99 = 0;
    uint32_t index_word_100 = 0;
    uint32_t tile_word_101 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_reused_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_address_word_16 = 0;
    uint32_t uniform_address_word_17 = 0;
    uint32_t uniform_index_word_79 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    bool uniform_condition_2 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    matrix_accumulator_19 = parameters[parameter_LW_9];

    matrix_accumulator_13 = blockIdx.x;

    uniform_address_word_7 = parameters[parameter_LH_8];

    matrix_accumulator_15 = threadIdx.x;

    uniform_reused_word_12 = 0u;
    uniform_reused_word_13 = 0u;

    packed_half_6 = threadIdx.y;

    matrix_accumulator_17 = threadIdx.x & 31u;

    packed_half_7 = matrix_accumulator_19 + 0xffffffffu;

    address_word_10 = uint32_t(int32_t(packed_half_7) >> 0x1fu);

    packed_half_0 = uint32_t(combine_words(address_word_10, 0u) << 0x3u >> 32) + packed_half_7;

    index_word_9 = uint32_t(int32_t(packed_half_0) >> 0x3u);

    condition_1 = (matrix_accumulator_15 | packed_half_6) != 0;

    index_word_8 = index_word_9 + 0x1u;

    block_index_11 = int32_t(index_word_8) < 0 ? 0u - index_word_8 : index_word_8;

    matrix_accumulator_12 = int32_t(index_word_8) < 0 ? 0u - index_word_8 : index_word_8;

    packed_half_0 = __float_as_uint(__int2float_ru(block_index_11));

    // All three decisions use the same vote over the current active lanes.
    uniform_condition_0 = vote_warp_predicate<true>(condition_1);
    uniform_condition_1 = uniform_condition_0;
    uniform_condition_2 = uniform_condition_0;

    packed_half_0 = reciprocal_bits(packed_half_0);

    address_word_2 = packed_half_0 + 0xffffffeu;

    packed_half_0 =
        int32_t(matrix_accumulator_13) < 0 ? 0u - matrix_accumulator_13 : matrix_accumulator_13;

    address_word_3 = __float2uint_rz(__uint_as_float(address_word_2));

    address_word_2 = 0u;

    address_word_4 = 0u - address_word_3;

    address_word_5 = address_word_4 * block_index_11;

    address_word_3 = uint32_t(uint64_t(address_word_3) * address_word_5 +
                                  combine_words(address_word_2, address_word_3) >>
                              32);

    address_word_2 = 0u - matrix_accumulator_12;

    if (!condition_1) {
        matrix_accumulator_12 = 0u;
    }

    address_word_3 = uint32_t(uint64_t(address_word_3) * packed_half_0 >> 32);

    packed_half_0 = address_word_3 * address_word_2 + packed_half_0;

    if (!condition_1) {
        address_word_2 = blockDim.x;
    }

    condition_2 = uint32_t(block_index_11) > uint32_t(packed_half_0);

    if (!condition_1) {
        address_word_5 = blockDim.y;
    }

    if (!condition_2) {
        packed_half_0 = packed_half_0 + (0u - block_index_11);

        address_word_3 = address_word_3 + 0x1u;
    }

    condition_0 = uint32_t(packed_half_0) >= uint32_t(block_index_11);

    packed_half_0 = index_word_8 ^ matrix_accumulator_13;

    condition_2 = int32_t(packed_half_0) >= int32_t(0u);

    packed_half_0 = blockIdx.z;

    if (!condition_1) {
        block_index_11 = address_word_2 * address_word_5;

        address_word_2 = 0x400u;
    }

    if (condition_0) {
        address_word_3 = address_word_3 + 0x1u;
    }

    condition_0 = int32_t(index_word_8) != int32_t(0u);

    address_word_4 = parameters[parameter_trained_weights_low_4];
    address_word_5 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_4 = broadcast_warp_word(block_index_11, !condition_1, uniform_index_word_4);

    matrix_accumulator_14 = address_word_3;

    address_word_3 = packed_half_6 * 0x40u;

    if (!condition_2) {
        matrix_accumulator_14 = 0u - matrix_accumulator_14;
    }

    if (!condition_0) {
        matrix_accumulator_14 = ~index_word_8;
    }

    index_word_8 = address_word_3 & 0xc0u;

    if (!uniform_condition_0) {
        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_address_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);
    }

    if (!condition_1) {
        address_word_3 = address_word_2 + 0x3000u;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_1);

    address_word_2 = matrix_accumulator_14 * 0x100u + index_word_8;

    if (!condition_1) {
        matrix_accumulator_12 = (matrix_accumulator_12 << 0x18u) + address_word_3;
    }

    address_word_3 = address_word_2 * 0x8u;

    uniform_address_word_6 =
        broadcast_warp_word(matrix_accumulator_12, !condition_1, uniform_address_word_6);

    block_index_11 = packed_half_0 * 0x20000u + address_word_3;

    index_word_8 = packed_half_6 & 0x3u;

    split_address_words(address_word_4, address_word_5,
                        int64_t(int32_t(block_index_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_4, address_word_5));

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    uniform_condition_0 = uint32_t(uniform_address_word_7) != 0u;

    block_index_11 = blockIdx.y;

    index_word_8 = index_word_8 | matrix_accumulator_15;
    condition_0 = index_word_8 != 0;

    split_address_words(address_word_4, address_word_5,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_4, address_word_5));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x10u);
#endif
    }

    __syncthreads();

    index_word_9 = index_word_9 * matrix_accumulator_14 + matrix_accumulator_14;

    matrix_accumulator_12 = uint32_t(int32_t(matrix_accumulator_19) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_index_word_4 = uniform_address_word_7 + 0xffffffffu;
    }

    uniform_address_word_6 = uint32_t(int32_t(uniform_address_word_7) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_address_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);
    }

    uniform_address_word_6 =
        uint32_t(combine_words(uniform_address_word_6, 0u) << 0x2u >> 32) + uniform_address_word_7;

    if (uniform_condition_0) {
        uniform_address_word_5 = uint32_t(combine_words(uniform_address_word_5, 0u) << 0x2u >> 32) +
                                 uniform_index_word_4;
    }

    uniform_index_word_8 = uint32_t(int32_t(uniform_address_word_6) >> 0x2u);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5),
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    uniform_index_word_4 = 0u;

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x200ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x400ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x600ull,
                                        matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
                                        matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x8000ull,
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x8200ull,
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x8400ull,
                                        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38,
                                        matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_4, address_word_5) + 0x8600ull,
                                        matrix_fragment_32, matrix_fragment_33, matrix_fragment_34,
                                        matrix_fragment_35, memory_ranges, error_counts);

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_address_word_5)) * (1ll << 0x1eu) >> 32) + 0x1u;
    }

    matrix_accumulator_81 = 0u;

    address_word_5 = uint32_t(combine_words(address_word_10, 0u) << 0x2u >> 32) + packed_half_7;

    address_word_10 = 0u - index_word_9 + matrix_accumulator_13;

    address_word_4 =
        uint32_t(combine_words(matrix_accumulator_12, 0u) << 0x2u >> 32) + matrix_accumulator_19;

    address_word_5 = uint32_t(int64_t(int32_t(address_word_5)) * (1ll << 0x1eu) >> 32) + 0x1u;

    address_word_4 = uint32_t(int32_t(address_word_4) >> 0x2u);

    packed_half_7 = address_word_10 * 0x2u;
    if (!condition_0) {
        index_word_9 = block_index_11 * 0x8u + packed_half_6;

        matrix_accumulator_12 = (address_word_10 << 0x3u) + 0x4u;

        index_word_9 = uint32_t(uint32_t(index_word_9) >> 0x2u);

        matrix_accumulator_12 =
            uint32_t(combine_words(matrix_accumulator_12, 0u) << 0x1eu >> 32) + 0x1u;

        condition_0 = int32_t(index_word_9) < int32_t(uniform_index_word_4);
        if (condition_0) {
            matrix_accumulator_13 = index_word_9 + 0x1u;

            matrix_accumulator_16 = min(int32_t(matrix_accumulator_12), int32_t(address_word_5));

            matrix_accumulator_14 = address_word_5 * 0x2u;

            matrix_accumulator_18 =
                min(int32_t(matrix_accumulator_13), int32_t(uniform_index_word_4));
            do {
                condition_0 = int32_t(packed_half_7) >= int32_t(matrix_accumulator_16);
                if (!condition_0) {
                    matrix_accumulator_13 = packed_half_0 + index_word_9;

                    matrix_accumulator_15 = packed_half_7;

                    matrix_accumulator_20 = matrix_accumulator_14 * matrix_accumulator_13;
                    do {
                        matrix_accumulator_12 = parameters[parameter_wait_low_6];
                        matrix_accumulator_13 = parameters[parameter_wait_high_7];

                        matrix_accumulator_17 = matrix_accumulator_20 + matrix_accumulator_15;

                        split_address_words(
                            matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));

                        matrix_accumulator_17 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_12, matrix_accumulator_13),
                            memory_ranges, error_counts);

                        matrix_accumulator_15 = matrix_accumulator_15 + 0x1u;

                        condition_1 =
                            uint32_t(matrix_accumulator_15) != uint32_t(matrix_accumulator_16);

                        condition_0 = int32_t(matrix_accumulator_17) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                matrix_accumulator_17 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13),
                                    memory_ranges, error_counts);

                                condition_0 = int32_t(matrix_accumulator_17) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                index_word_9 = index_word_9 + 0x1u;

                condition_0 = uint32_t(index_word_9) != uint32_t(matrix_accumulator_18);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    index_word_91 = parameters[parameter_LH_8];

    index_word_94 = uint32_t(uint32_t(packed_half_6) >> 0x2u);

    index_word_9 = uint32_t(uint32_t(packed_half_6) >> 0x1u);

    __syncthreads();

    condition_0 = false;

    index_word_95 = block_index_11 * 0x2u + index_word_94;

    matrix_accumulator_12 = index_word_9 & 0x1u;

    reused_word_92 = index_word_95 * address_word_4;

    uniform_index_word_14 = 0u;

    index_word_9 = packed_half_7 + matrix_accumulator_12;

    matrix_accumulator_12 = packed_half_6 * 0x20u;

    matrix_accumulator_13 = index_word_9;

    index_word_91 = index_word_91 & 0xfffffffcu;

    condition_2 = uint32_t(index_word_91) != uint32_t(0x4u);

    condition_1 = int32_t(index_word_95) < int32_t(uniform_index_word_8) || !condition_2;

    reused_word_90 = condition_2 ? reused_word_92 : 0u;
    if (condition_1) {
        matrix_accumulator_13 = parameters[parameter_LW_9];

        matrix_accumulator_13 = matrix_accumulator_13 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_13) != uint32_t(0x4u);

        matrix_accumulator_13 = 0u;

        condition_0 = int32_t(index_word_9) < int32_t(address_word_4) || !condition_2;

        if (condition_2) {
            matrix_accumulator_13 = index_word_9;
        }
    }
    if (condition_0) {
        matrix_accumulator_18 = parameters[parameter_input_low_0];
        matrix_accumulator_19 = parameters[parameter_input_high_1];
    }

    index_word_93 = matrix_accumulator_12 & 0x20u;

    matrix_accumulator_14 = packed_half_6 * 0x80u;

    if (condition_0) {
        matrix_accumulator_12 = reused_word_90 + matrix_accumulator_13;
    }

    uniform_index_word_11 = 0x400u;

    index_word_93 = packed_half_0 * 0x200u + index_word_93;

    matrix_accumulator_15 = matrix_accumulator_14 & 0x100u;

    uniform_address_word_5 = (uniform_index_word_14 << 0x18u) + uniform_index_word_11;

    matrix_accumulator_14 = matrix_accumulator_14 & 0x80u;

    if (condition_0) {
        matrix_accumulator_16 = matrix_accumulator_12 * 0x200u + index_word_93;
    }

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    matrix_accumulator_15 = index_word_94 * 0x200u + matrix_accumulator_15;

    if (condition_0) {
        matrix_accumulator_16 = matrix_accumulator_16 * 0x4u;
    }

    matrix_accumulator_14 = matrix_accumulator_15 + matrix_accumulator_14;

    if (condition_0) {
        matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_16) >> 0x1fu);

        matrix_accumulator_12 = matrix_accumulator_16;
    }

    index_word_89 = (matrix_accumulator_14 << 0x2u) + uniform_address_word_5;

    if (condition_0) {
        matrix_accumulator_13 = matrix_accumulator_17;
    }

    split_address_words(matrix_accumulator_16, matrix_accumulator_17, 0ull);

    if (condition_0) {
        const uint64_t byte_address_0 =
            scaled_word_address(combine_words(matrix_accumulator_18, matrix_accumulator_19),
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));
        matrix_accumulator_15 = uint32_t(byte_address_0);

        matrix_accumulator_12 = uint32_t(byte_address_0 >> 32);

        matrix_accumulator_16 = matrix_accumulator_15;

        matrix_accumulator_17 = matrix_accumulator_12;
    }
    if (!condition_0) {
        matrix_accumulator_16 = threadIdx.x & 31u;

        uniform_address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_5;

        matrix_accumulator_13 = uniform_address_word_5;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_13, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_16 = matrix_accumulator_16 * 0x10u + index_word_89;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_16,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_5 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_16);

            uniform_address_word_5 = uniform_index_word_11 + 0x3000u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_17);

            uniform_address_word_6 = broadcast_active_lane(index_word_89);

            uniform_address_word_7 = (uniform_index_word_14 << 0x18u) + uniform_address_word_5;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_5 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_5 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_5 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_13 = index_word_9;
    if (condition_1) {
        matrix_accumulator_12 = parameters[parameter_LW_9];

        matrix_accumulator_13 = 0u;

        matrix_accumulator_12 = matrix_accumulator_12 & 0xfffffffcu;

        condition_2 = uint32_t(matrix_accumulator_12) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_9) < int32_t(address_word_4) || !condition_2;

        if (condition_2) {
            matrix_accumulator_13 = index_word_9;
        }
    }
    if (!condition_0) {
        matrix_accumulator_12 = threadIdx.x & 31u;

        uniform_address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_13 = uniform_address_word_5;

        matrix_accumulator_14 = uniform_address_word_5;

        matrix_accumulator_16 = matrix_accumulator_12 * 0x10u + index_word_89;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_13, matrix_accumulator_14, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_16 + 0x1000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];

            matrix_accumulator_12 = reused_word_90 + matrix_accumulator_13;
        }

        uniform_address_word_5 = 0xffffffffu;

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            matrix_accumulator_12 = matrix_accumulator_12 * 0x200u + index_word_93;

            matrix_accumulator_14 = (matrix_accumulator_12 << 0x2u) + 0x100u;
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        if (condition_0) {
            matrix_accumulator_15 = uint32_t(int32_t(matrix_accumulator_14) >> 0x1fu);

            matrix_accumulator_12 = matrix_accumulator_14;

            matrix_accumulator_13 = matrix_accumulator_15;

            const uint64_t byte_address_1 =
                scaled_word_address(combine_words(matrix_accumulator_16, matrix_accumulator_17),
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));
            matrix_accumulator_14 = uint32_t(byte_address_1);

            matrix_accumulator_13 = uint32_t(byte_address_1 >> 32);

            matrix_accumulator_18 = matrix_accumulator_14;

            matrix_accumulator_19 = matrix_accumulator_13;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_6 = broadcast_active_lane(index_word_89);

            uniform_address_word_5 = uniform_index_word_11 + 0x3000u;

            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_18);

            uniform_address_word_9 = 0x20u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_19);

            uniform_address_word_5 = (uniform_index_word_14 << 0x18u) + uniform_address_word_5;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_7 = uniform_address_word_5 + 0x8u;

            uniform_address_word_6 = uniform_address_word_6 + 0x1000u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_9 * 16u, uniform_address_word_5 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_9 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_13 = index_word_9;
    if (condition_1) {
        matrix_accumulator_12 = parameters[parameter_LW_9];

        matrix_accumulator_13 = 0u;

        matrix_accumulator_12 = matrix_accumulator_12 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_12) != uint32_t(0x4u);

        condition_0 = int32_t(index_word_9) < int32_t(address_word_4) || !condition_1;

        if (condition_1) {
            matrix_accumulator_13 = index_word_9;
        }
    }
    if (!condition_0) {
        matrix_accumulator_12 = threadIdx.x & 31u;

        uniform_address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_13 = uniform_address_word_5;

        matrix_accumulator_14 = uniform_address_word_5;

        matrix_accumulator_16 = matrix_accumulator_12 * 0x10u + index_word_89;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_13, matrix_accumulator_14, 0x5410u);

        matrix_accumulator_13 = matrix_accumulator_12;

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_16 + 0x2000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];

            matrix_accumulator_12 = reused_word_90 + matrix_accumulator_13;
        }

        uniform_address_word_5 = 0xffffffffu;

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            matrix_accumulator_12 = matrix_accumulator_12 * 0x200u + index_word_93;

            matrix_accumulator_14 = (matrix_accumulator_12 << 0x2u) + 0x200u;
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        if (condition_0) {
            matrix_accumulator_15 = uint32_t(int32_t(matrix_accumulator_14) >> 0x1fu);

            matrix_accumulator_12 = matrix_accumulator_14;

            matrix_accumulator_13 = matrix_accumulator_15;

            // Later control flow also consumes this address-overflow predicate.
            condition_1 =
                (uint64_t(matrix_accumulator_12) << 2) + matrix_accumulator_16 > 0xffffffffull;
            const uint64_t byte_address_2 =
                scaled_word_address(combine_words(matrix_accumulator_16, matrix_accumulator_17),
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));
            matrix_accumulator_14 = uint32_t(byte_address_2);

            matrix_accumulator_13 = uint32_t(byte_address_2 >> 32);

            matrix_accumulator_18 = matrix_accumulator_14;

            matrix_accumulator_19 = matrix_accumulator_13;
        }

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_6 = broadcast_active_lane(index_word_89);

            uniform_address_word_5 = uniform_index_word_11 + 0x3000u;

            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_18);

            uniform_address_word_9 = 0x20u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_19);

            uniform_address_word_5 = (uniform_index_word_14 << 0x18u) + uniform_address_word_5;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_7 = uniform_address_word_5 + 0x10u;

            uniform_address_word_6 = uniform_address_word_6 + 0x2000u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_9 * 16u, uniform_address_word_5 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_9 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_15 = 0u;

    uniform_address_word_5 = uniform_index_word_11 + 0x3000u;

    packed_half_6 = block_index_11 * 0x8u + packed_half_6;

    address_word_10 = (address_word_10 << 0x3u) + 0x4u;

    matrix_accumulator_12 = uniform_address_word_5;

    packed_half_6 = uint32_t(uint32_t(packed_half_6) >> 0x2u);

    condition_0 = int32_t(packed_half_6) < int32_t(uniform_index_word_4);

    condition_0 = uint32_t(index_word_8) != 0u || !condition_0;

    matrix_accumulator_15 = (matrix_accumulator_15 << 0x18u) + matrix_accumulator_12;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_accumulator_15;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_12 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 = wait_copy_barrier(shared_tile, matrix_accumulator_15,
                                    async_tokens[(((matrix_accumulator_15) >> 3) & 1)]);
#else
    __syncthreads();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_accumulator_15,
                                            async_tokens[(((matrix_accumulator_15) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    index_word_8 = parameters[parameter_LW_9];

    address_word_10 = uint32_t(combine_words(address_word_10, 0u) << 0x1eu >> 32) + 0x1u;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    block_index_11 = packed_half_6 + 0x1u;

    index_word_88 = 0u;

    matrix_accumulator_80 = 0u;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    matrix_accumulator_86 = 0;
    matrix_accumulator_87 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_24 = 0;
    matrix_accumulator_25 = 0;

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_accumulator_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_accumulator_76 = 0;
    matrix_accumulator_77 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_18 = 0;
    matrix_accumulator_19 = 0;

    matrix_accumulator_12 = 0;
    matrix_accumulator_13 = 0;

    matrix_accumulator_14 = 0;
    matrix_accumulator_15 = 0;

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    address_word_10 = min(int32_t(address_word_10), int32_t(address_word_5));

    block_index_11 = min(int32_t(block_index_11), int32_t(uniform_index_word_4));

    index_word_8 = index_word_8 & 0xfffffffcu;
    do {
        index_word_100 = uint32_t(uint32_t(index_word_88) >> 0x6u);

        index_word_96 = 0u;

        index_word_97 = 0x400u;

        __syncwarp(__activemask());

        matrix_fragment_64 = permute_signed(index_word_100, 0u, 0x9910u);

        warp_lane_98 = threadIdx.x & 31u;

        condition_1 = uint32_t(index_word_88) > uint32_t(0x1bfu);

        condition_2 = uint32_t(index_word_88) > uint32_t(0x13fu);

        matrix_fragment_64 = matrix_fragment_64 * 0xabu;

        matrix_fragment_64 = matrix_fragment_64 & 0xffffu;

        matrix_fragment_64 = uint32_t(uint32_t(matrix_fragment_64) >> 0x9u);

        matrix_fragment_64 = permute_signed(matrix_fragment_64, 0u, 0x9910u);

        matrix_fragment_64 = matrix_fragment_64 * 0x3u;

        matrix_fragment_64 = 0u - matrix_fragment_64;

        matrix_fragment_65 = permute_signed(matrix_fragment_64, 0u, 0x7710u);

        index_word_100 = index_word_100 + matrix_fragment_65;

        matrix_fragment_65 = (index_word_96 << 0x18u) + index_word_97;

        index_word_99 = index_word_100 * 0x1000u;

        index_word_99 = 0xff000u & index_word_99;

        matrix_fragment_64 = index_word_94 * 0x800u + index_word_99;

        matrix_fragment_65 = matrix_fragment_65 + matrix_fragment_64;

        tile_word_101 = warp_lane_98 * 0x10u + matrix_fragment_65;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_101, matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
            matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_101 + 0x400u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_101 + 0x600u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_76,
                                 matrix_accumulator_77);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_86,
                                 matrix_accumulator_87);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_101 + 0x200u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_76,
                                 matrix_accumulator_77);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_16,
                                 matrix_accumulator_17);
        if (!condition_1) {
            matrix_fragment_67 = index_word_88 + 0x40u;

            matrix_fragment_32 = parameters[parameter_trained_weights_low_4];
            matrix_fragment_33 = parameters[parameter_trained_weights_high_5];

            matrix_fragment_34 = packed_half_0 * 0x200u + matrix_fragment_67;

            matrix_fragment_67 = uint32_t(uint32_t(matrix_fragment_67) >> 0x6u);

            matrix_fragment_35 = matrix_fragment_34 * 0x100u + address_word_3;

            matrix_fragment_66 = permute_signed(matrix_fragment_67, 0u, 0x9910u);

            matrix_fragment_66 = matrix_fragment_66 * 0xabu;

            matrix_fragment_66 = matrix_fragment_66 & 0xffffu;

            matrix_fragment_66 = uint32_t(uint32_t(matrix_fragment_66) >> 0x9u);

            split_address_words(matrix_fragment_32, matrix_fragment_33,
                                int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_32, matrix_fragment_33));

            matrix_fragment_66 = permute_signed(matrix_fragment_66, 0u, 0x9910u);

            split_address_words(matrix_fragment_64, matrix_fragment_65,
                                int64_t(int32_t(warp_lane_98)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_32, matrix_fragment_33));

            matrix_fragment_66 = matrix_fragment_66 * 0x3u;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65), matrix_fragment_60,
                matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
                error_counts);

            matrix_fragment_66 = 0u - matrix_fragment_66;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x200ull,
                matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x400ull,
                matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x600ull,
                matrix_fragment_48, matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8000ull,
                matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8200ull,
                matrix_fragment_40, matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8400ull,
                matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8600ull,
                matrix_fragment_32, matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                memory_ranges, error_counts);

            index_word_97 = index_word_97 + 0x3000u;

            index_word_96 = (index_word_96 << 0x18u) + index_word_97;

            matrix_fragment_64 = permute_signed(matrix_fragment_66, 0u, 0x7710u);

            matrix_fragment_67 = matrix_fragment_67 + matrix_fragment_64;

            matrix_fragment_67 = matrix_fragment_67 & 0xffu;

            matrix_fragment_67 = matrix_fragment_67 * 0x8u + index_word_96;

#if DLSS5_ENABLE_ASYNC_COPIES
            {
                const uint32_t operand_a = matrix_fragment_67;
                async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
                split_address_words(matrix_fragment_64, matrix_fragment_65,
                                    async_tokens[operand_a >> 3 & 1]);
            }
#else
            matrix_fragment_64 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_fragment_67,
                                            async_tokens[(((matrix_fragment_67) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
            if (!condition_1) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_1 =
                        wait_copy_barrier(shared_tile, matrix_fragment_67,
                                          async_tokens[(((matrix_fragment_67) >> 3) & 1)]);
#else
                    __syncthreads();
                    condition_1 = true;
#endif
                } while (!condition_1);
            }
        }
        matrix_fragment_68 = index_word_88 + 0xc0u;

        index_word_100 = index_word_100 & 0xffu;
        if (!condition_2) {
            condition_1 = (matrix_fragment_68 & 0xc0u) != 0;
            if (!condition_1) {
                if (!condition_0) {
                    matrix_fragment_64 = packed_half_0 * 0x200u + matrix_fragment_68;

                    matrix_fragment_66 = packed_half_6;

                    matrix_fragment_71 = uint32_t(uint32_t(matrix_fragment_64) >> 0x8u);
                    do {
                        condition_1 = int32_t(packed_half_7) >= int32_t(address_word_10);
                        if (!condition_1) {
                            matrix_fragment_64 = matrix_fragment_66 * 0x2u + matrix_fragment_71;

                            matrix_fragment_67 = packed_half_7;

                            matrix_fragment_70 = matrix_fragment_64 * address_word_5;
                            do {
                                matrix_fragment_64 = parameters[parameter_wait_low_6];
                                matrix_fragment_65 = parameters[parameter_wait_high_7];

                                matrix_fragment_69 = matrix_fragment_70 + matrix_fragment_67;

                                split_address_words(
                                    matrix_fragment_64, matrix_fragment_65,
                                    int64_t(int32_t(matrix_fragment_69)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_64, matrix_fragment_65));

                                matrix_fragment_69 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_64, matrix_fragment_65),
                                    memory_ranges, error_counts);

                                matrix_fragment_67 = matrix_fragment_67 + 0x1u;

                                condition_2 =
                                    uint32_t(matrix_fragment_67) != uint32_t(address_word_10);

                                condition_1 = int32_t(matrix_fragment_69) > -1;
                                if (!condition_1) {
                                    do {
                                        __nanosleep(0x40u);

                                        matrix_fragment_69 = pipeline_read<CheckBounds>(
                                            combine_words(matrix_fragment_64, matrix_fragment_65),
                                            memory_ranges, error_counts);

                                        condition_1 = int32_t(matrix_fragment_69) > -1;
                                    } while (!condition_1);
                                }
                            } while (condition_2);
                        }
                        matrix_fragment_66 = matrix_fragment_66 + 0x1u;

                        condition_1 = uint32_t(matrix_fragment_66) != uint32_t(block_index_11);
                    } while (condition_1);
                }
                __syncwarp(__activemask());

                __syncthreads();
            }
            condition_1 = int32_t(index_word_95) >= int32_t(uniform_index_word_8);

            condition_2 = int32_t(index_word_95) >= int32_t(uniform_index_word_8);

            matrix_fragment_65 = index_word_93 + matrix_fragment_68;

            condition_1 = uint32_t(index_word_91) != uint32_t(0x4u) && condition_1;

            condition_2 = uint32_t(index_word_91) == uint32_t(0x4u) || !condition_2;

            condition_3 = uint32_t(index_word_8) == uint32_t(0x4u) || condition_1;

            matrix_fragment_64 = condition_1 ? index_word_9 : 0u;

            condition_4 = int32_t(index_word_9) < int32_t(address_word_4) || condition_3;

            condition_3 = uint32_t(index_word_8) != uint32_t(0x4u);

            condition_2 = condition_4 & condition_2;

            matrix_fragment_67 = !condition_3 ? matrix_fragment_64 : index_word_9;
            if (!condition_2) {
                matrix_fragment_68 = threadIdx.x & 31u;

                uniform_address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                index_word_99 = index_word_89 + index_word_99;

                matrix_fragment_64 = uniform_address_word_5;

                matrix_fragment_65 = uniform_address_word_5;

                matrix_fragment_64 =
                    permute_signed(matrix_fragment_64, matrix_fragment_65, 0x5410u);

                matrix_fragment_65 = matrix_fragment_64;

                matrix_fragment_66 = matrix_fragment_64;

                matrix_fragment_67 = matrix_fragment_64;

                matrix_fragment_68 = matrix_fragment_68 * 0x10u + index_word_99;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_68,
                                                     matrix_fragment_64, matrix_fragment_65,
                                                     matrix_fragment_66, matrix_fragment_67,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (condition_2) {
                    matrix_fragment_70 = parameters[parameter_input_low_0];
                    matrix_fragment_71 = parameters[parameter_input_high_1];

                    matrix_fragment_64 = reused_word_90 + matrix_fragment_67;
                }

                __syncwarp(__activemask());

                if (condition_2) {
                    matrix_fragment_64 = matrix_fragment_64 * 0x200u + matrix_fragment_65;
                }

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;

                if (condition_2) {
                    matrix_fragment_67 = matrix_fragment_64 * 0x4u;
                }

                split_address_words(matrix_fragment_64, matrix_fragment_65, 0ull);

                if (condition_2) {
                    matrix_fragment_64 = matrix_fragment_67;

                    matrix_fragment_65 = uint32_t(int32_t(matrix_fragment_67) >> 0x1fu);
                }

                split_address_words(matrix_fragment_66, matrix_fragment_67, 0ull);

                if (condition_2) {
                    const uint64_t byte_address_3 =
                        scaled_word_address(combine_words(matrix_fragment_70, matrix_fragment_71),
                                            combine_words(matrix_fragment_64, matrix_fragment_65));
                    matrix_fragment_69 = uint32_t(byte_address_3);

                    matrix_fragment_67 = uint32_t(byte_address_3 >> 32);

                    matrix_fragment_66 = matrix_fragment_69;
                }
                if (condition_1) {
                    uniform_address_word_6 = 0u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_9 = index_word_99;

                    uniform_address_word_5 = 0x400u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_7 = index_word_89;

                    uniform_address_word_5 = uniform_address_word_5 + 0x3000u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_index_word_10 = index_word_100;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_16 = matrix_fragment_66;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_17 = matrix_fragment_67;

                    matrix_fragment_64 = 0x200u;

                    uniform_address_word_5 =
                        (uniform_address_word_6 << 0x18u) + uniform_address_word_5;

                    uniform_address_word_6 = uniform_address_word_7 + uniform_address_word_9;

                    uniform_address_word_7 =
                        (uniform_index_word_10 << 0x3u) + uniform_address_word_5;

                    uniform_address_word_5 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_16, uniform_address_word_17),
                        uniform_address_word_5 * 16u, uniform_address_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_16, uniform_address_word_17),
                        uniform_address_word_5 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_1 = uint32_t(index_word_88) >= uint32_t(0x1c0u);

        index_word_88 = index_word_88 + 0x40u;
    } while (!condition_1);
    index_word_8 = parameters[parameter_output_low_2];
    index_word_9 = parameters[parameter_output_high_3];

    condition_0 = int32_t(index_word_95) >= int32_t(uniform_index_word_8);

    address_word_3 = packed_half_7 + reused_word_92;

    address_word_5 = packed_half_7 + 0x1u;

    condition_1 = int32_t(packed_half_7) >= int32_t(address_word_4) || condition_0;

    address_word_3 = address_word_3 * 0x400u + address_word_2;

    condition_0 = int32_t(address_word_5) >= int32_t(address_word_4) || condition_0;

    address_word_3 = address_word_3 * 0x4u;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_8, index_word_9));
    if (!condition_1) {
        matrix_fragment_33 = threadIdx.x & 31u;

        packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                      select_half_operand<2, false, false>(0u));

        index_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                     select_half_operand<2, false, false>(0u));

        matrix_accumulator_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                          select_half_operand<2, false, false>(packed_half_6));

        address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(index_word_8));

        address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                      select_half_operand<2, false, false>(0u));

        packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(0u));

        matrix_accumulator_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                          select_half_operand<2, false, false>(0u));

        block_index_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       select_half_operand<2, false, false>(address_word_4));

        index_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                     select_half_operand<2, false, false>(address_word_5));

        matrix_accumulator_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                          select_half_operand<2, false, false>(matrix_accumulator_27));

        index_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                     select_half_operand<2, false, false>(packed_half_0));

        address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                        select_half_operand<2, false, false>(packed_half_7));

        matrix_accumulator_26 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(matrix_accumulator_29));

        split_address_words(address_word_4, address_word_5,
                            int64_t(int32_t(matrix_fragment_33)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_27 = matrix_accumulator_23;

        store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                index_word_8, index_word_9, address_word_10,
                                                block_index_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_4, address_word_5) + 0x200ull, matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);
    }
    if (condition_0) {
        return;
    }

    matrix_accumulator_23 = threadIdx.x & 31u;

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                  select_half_operand<2, false, false>(0u));

    packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                  select_half_operand<2, false, false>(0u));

    index_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                 select_half_operand<2, false, false>(0u));

    address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                   select_half_operand<2, false, false>(0u));

    address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                   select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                  select_half_operand<2, false, false>(0u));

    packed_half_7 = permute_signed(packed_half_7, address_word_4, 0x5410u);

    packed_half_6 = permute_signed(packed_half_6, index_word_9, 0x5410u);

    address_word_5 = permute_signed(address_word_5, packed_half_0, 0x5410u);

    address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                   select_half_operand<2, false, false>(0u));

    index_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                 select_half_operand<2, false, false>(0u));

    address_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_14),
                                    select_half_operand<2, false, false>(0u));

    block_index_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                                   select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                                  select_half_operand<2, false, false>(0u));

    index_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                                 select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(matrix_accumulator_23)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_2, address_word_3));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                      select_half_operand<2, false, false>(0u));

    address_word_4 = permute_signed(address_word_4, index_word_9, 0x5410u);

    block_index_11 = permute_signed(block_index_11, matrix_accumulator_14, 0x5410u);

    address_word_10 = permute_signed(address_word_10, matrix_accumulator_15, 0x5410u);

    store_four_words_streaming<CheckBounds>(
        combine_words(address_word_2, address_word_3) + 0x4000ull, address_word_4, address_word_5,
        packed_half_6, packed_half_7, memory_ranges, error_counts);

    index_word_9 = permute_signed(matrix_accumulator_19, packed_half_0, 0x5410u);

    index_word_8 = permute_signed(index_word_8, matrix_accumulator_13, 0x5410u);

    store_four_words_streaming<CheckBounds>(
        combine_words(address_word_2, address_word_3) + 0x4200ull, index_word_8, index_word_9,
        address_word_10, block_index_11, memory_ranges, error_counts);
    return;
}
extern "C" __global__ __maxnreg__(
    146) void phoenix_global_attention_split_heads_512_final_head_wait(KernelArguments parameters) {
    run_global_attention_split_heads_512_final_head_wait<false>(parameters.words, nullptr, nullptr);
}
