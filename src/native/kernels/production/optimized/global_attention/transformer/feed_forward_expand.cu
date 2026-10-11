// DLSS5 Phoenix optimized kernel: global_attention/transformer/feed_forward_expand.cu
// Coarsest 1024-channel global-transformer stage. Attention spans the full latent token set.
// Projection and feed-forward modules retain their explicit publication and synchronization boundaries.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 25624
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

// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_output_low_4 = 4,
    parameter_output_high_5 = 5,
    parameter_trained_weights_low_6 = 6,
    parameter_trained_weights_high_7 = 7,
    parameter_wait_optional_low_12 = 12,
    parameter_wait_optional_high_13 = 13,
    parameter_publication_optional_low_14 = 14,
    parameter_publication_optional_high_15 = 15,
    parameter_LH_16 = 16,
    parameter_LW_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[18];
};
static_assert(sizeof(KernelArguments) == 72, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_transformer_feed_forward_expand(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t address_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t packed_half_3 = 0;
    uint32_t packed_half_4 = 0;
    uint32_t index_word_5 = 0;
    uint32_t packed_half_6 = 0;
    uint32_t packed_half_7 = 0;
    uint32_t packed_half_8 = 0;
    uint32_t packed_half_9 = 0;
    uint32_t packed_half_10 = 0;
    uint32_t packed_half_11 = 0;
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t matrix_accumulator_14 = 0;
    uint32_t matrix_accumulator_15 = 0;
    uint32_t matrix_accumulator_16 = 0;
    uint32_t warp_lane_17 = 0;
    uint32_t matrix_accumulator_18 = 0;
    uint32_t address_word_19 = 0;
    uint32_t address_word_20 = 0;
    uint32_t address_word_21 = 0;
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
    uint32_t matrix_accumulator_40 = 0;
    uint32_t matrix_accumulator_41 = 0;
    uint32_t matrix_accumulator_42 = 0;
    uint32_t matrix_accumulator_43 = 0;
    uint32_t matrix_accumulator_44 = 0;
    uint32_t matrix_accumulator_45 = 0;
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
    uint32_t packed_half_124 = 0;
    uint32_t packed_half_125 = 0;
    uint32_t packed_half_126 = 0;
    uint32_t packed_half_127 = 0;
    uint32_t packed_half_128 = 0;
    uint32_t packed_half_129 = 0;
    uint32_t packed_half_130 = 0;
    uint32_t packed_half_131 = 0;
    uint32_t packed_half_132 = 0;
    uint32_t packed_half_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t warp_lane_135 = 0;
    uint32_t packed_half_136 = 0;
    uint32_t index_word_137 = 0;
    uint32_t address_word_138 = 0;
    uint32_t index_word_139 = 0;
    uint32_t tile_word_140 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_reused_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_address_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_address_word_16 = 0;
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

    packed_half_8 = parameters[parameter_LH_16];
    packed_half_9 = parameters[parameter_LW_17];

    matrix_accumulator_12 = blockIdx.x;

    uniform_reused_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    packed_half_124 = threadIdx.y;

    matrix_accumulator_13 = threadIdx.x;

    packed_half_125 = blockIdx.z;

    matrix_accumulator_15 = threadIdx.x & 31u;

    packed_half_9 = packed_half_9 * packed_half_8 + 0xffffffffu;

    packed_half_10 = uint32_t(int32_t(packed_half_9) >> 0x1fu);

    address_word_0 = uint32_t(combine_words(packed_half_10, 0u) << 0x7u >> 32) + packed_half_9;

    packed_half_11 = uint32_t(int32_t(address_word_0) >> 0x7u);

    packed_half_133 = matrix_accumulator_13 | packed_half_124;
    condition_1 = packed_half_133 != 0;

    packed_half_8 = packed_half_11 + 0x1u;

    index_word_5 = int32_t(packed_half_8) < 0 ? 0u - packed_half_8 : packed_half_8;

    packed_half_6 = int32_t(packed_half_8) < 0 ? 0u - packed_half_8 : packed_half_8;

    address_word_0 = __float_as_uint(__int2float_ru(index_word_5));

    // All three decisions use the same vote over the current active lanes.
    uniform_condition_0 = vote_warp_predicate<true>(condition_1);
    uniform_condition_1 = uniform_condition_0;
    uniform_condition_2 = uniform_condition_0;

    address_word_0 = reciprocal_bits(address_word_0);

    address_word_2 = address_word_0 + 0xffffffeu;

    packed_half_3 = __float2uint_rz(__uint_as_float(address_word_2));

    address_word_2 = 0u;

    packed_half_4 = 0u - packed_half_3;

    packed_half_7 = packed_half_4 * index_word_5;

    packed_half_4 =
        int32_t(matrix_accumulator_12) < 0 ? 0u - matrix_accumulator_12 : matrix_accumulator_12;

    packed_half_3 = uint32_t(uint64_t(packed_half_3) * packed_half_7 +
                                 combine_words(address_word_2, packed_half_3) >>
                             32);

    packed_half_7 = 0u - packed_half_6;

    address_word_2 = packed_half_4;

    if (!condition_1) {
        packed_half_6 = 0u;
    }

    address_word_0 = uint32_t(uint64_t(packed_half_3) * address_word_2 >> 32);

    if (!condition_1) {
        packed_half_3 = blockDim.x;
    }

    address_word_2 = address_word_0 * packed_half_7 + address_word_2;

    condition_2 = uint32_t(index_word_5) > uint32_t(address_word_2);

    if (!condition_1) {
        packed_half_4 = blockDim.y;
    }

    if (!condition_2) {
        address_word_2 = address_word_2 + (0u - index_word_5);

        address_word_0 = address_word_0 + 0x1u;
    }

    condition_0 = uint32_t(address_word_2) >= uint32_t(index_word_5);

    address_word_2 = matrix_accumulator_12 ^ packed_half_8;

    condition_2 = int32_t(address_word_2) >= int32_t(0u);

    if (!condition_1) {
        packed_half_7 = packed_half_3 * packed_half_4;

        address_word_2 = 0x400u;
    }

    packed_half_3 = packed_half_124 * 0x40u;

    packed_half_4 = parameters[parameter_trained_weights_low_6];
    index_word_5 = parameters[parameter_trained_weights_high_7];

    uniform_address_word_4 =
        broadcast_warp_word(packed_half_7, !condition_1, uniform_address_word_4);

    packed_half_7 = packed_half_3 & 0x40u;

    if (!condition_1) {
        packed_half_3 = address_word_2 + 0x6000u;
    }

    if (condition_0) {
        address_word_0 = address_word_0 + 0x1u;
    }

    condition_0 = int32_t(packed_half_8) != int32_t(0u);

    if (!condition_1) {
        packed_half_6 = (packed_half_6 << 0x18u) + packed_half_3;
    }

    if (!condition_2) {
        address_word_0 = 0u - address_word_0;
    }

    uniform_address_word_6 =
        broadcast_warp_word(packed_half_6, !condition_1, uniform_address_word_6);

    if (!uniform_condition_0) {
        uniform_address_word_4 = 0u - uniform_address_word_4 + 0x100000u;

        uniform_address_word_5 = uniform_address_word_4 << (0xbu & 31u);

        uniform_address_word_4 = uniform_address_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_1);

    if (!condition_0) {
        address_word_0 = ~packed_half_8;
    }

    address_word_2 = address_word_0 * 0x80u + packed_half_7;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    packed_half_3 = address_word_2 * 0x8u;

    packed_half_7 = packed_half_125 * 0x100000u + packed_half_3;

    split_address_words(packed_half_4, index_word_5,
                        int64_t(int32_t(packed_half_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_4, index_word_5));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    split_address_words(packed_half_6, packed_half_7,
                        int64_t(int32_t(matrix_accumulator_15)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_4, index_word_5));

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x10u);
#endif
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7),
                                        matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
                                        matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x200ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x400ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x600ull,
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x20000ull,
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x20200ull,
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x20400ull,
                                        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
                                        matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_6, packed_half_7) + 0x20600ull,
                                        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                        matrix_fragment_79, memory_ranges, error_counts);

    condition_0 = (packed_half_124 & 0x1u) != 0;

    index_word_5 = packed_half_11 * address_word_0 + address_word_0;

    packed_half_9 = uint32_t(combine_words(packed_half_10, 0u) << 0x5u >> 32) + packed_half_9;

    matrix_accumulator_15 = 0u;

    condition_0 = uint32_t(matrix_accumulator_13) != 0u || condition_0;

    packed_half_9 = packed_half_9 & 0xffffffe0u;

    index_word_5 = 0u - index_word_5 + matrix_accumulator_12;

    packed_half_9 = packed_half_9 + 0x20u;

    packed_half_10 = uint32_t(int32_t(packed_half_9) >> 0x4u);

    packed_half_9 = index_word_5 * 0x80u;
    if (!condition_0) {
        packed_half_4 = packed_half_124 * 0x20u + packed_half_9;

        packed_half_6 = packed_half_4 + 0x20u;

        packed_half_4 = uint32_t(uint32_t(packed_half_4) >> 0x7u);

        packed_half_7 = uint32_t(combine_words(packed_half_6, 0u) << 0x19u >> 32) + 0x1u;

        matrix_accumulator_13 = min(int32_t(packed_half_8), int32_t(packed_half_7));

        condition_1 = int32_t(packed_half_4) >= int32_t(matrix_accumulator_13);
        if (!condition_1) {
            do {
                packed_half_6 = parameters[parameter_wait_optional_low_12];
                packed_half_7 = parameters[parameter_wait_optional_high_13];

                packed_half_11 = packed_half_125 + packed_half_4;

                packed_half_11 = packed_half_11 * 0x8u;

                split_address_words(packed_half_6, packed_half_7,
                                    uint64_t(uint32_t(packed_half_11)) * uint64_t(uint32_t(0x4u)) +
                                        combine_words(packed_half_6, packed_half_7));

                packed_half_11 = pipeline_read<CheckBounds>(
                    combine_words(packed_half_6, packed_half_7), memory_ranges, error_counts);

                packed_half_4 = packed_half_4 + 0x1u;

                condition_2 = uint32_t(packed_half_4) != uint32_t(matrix_accumulator_13);

                condition_1 = int32_t(packed_half_11) > -1;
                if (!condition_1) {
                    do {
                        __nanosleep(0x40u);

                        packed_half_11 =
                            pipeline_read<CheckBounds>(combine_words(packed_half_6, packed_half_7),
                                                       memory_ranges, error_counts);

                        condition_1 = int32_t(packed_half_11) > -1;
                    } while (!condition_1);
                }
            } while (condition_2);
        }
    }
    __syncwarp(__activemask());

    packed_half_4 = index_word_5 * 0x8u + packed_half_124;

    matrix_accumulator_12 = 0x80u;

    packed_half_132 = packed_half_124 * 0x100u;

    matrix_accumulator_16 = 0x1u;

    packed_half_6 = packed_half_4 * 0x1000u;

    condition_3 = int32_t(packed_half_4) >= int32_t(packed_half_10);

    condition_2 = int32_t(packed_half_4) >= int32_t(packed_half_10);

    packed_half_7 = permute_signed(0u, packed_half_7, 0x7610u);

    packed_half_11 = packed_half_125 * 0x1000u + packed_half_6;

    matrix_accumulator_13 = 0x80u;

    uniform_index_word_10 = 0u;

    uniform_index_word_11 = 0x400u;

    __syncthreads();

    address_word_19 = !condition_2 ? packed_half_11 : 0u;

    split_address_words(address_word_20, address_word_21, 0ull);

    if (!condition_3) {
        matrix_accumulator_14 = packed_half_4 + packed_half_125;

        matrix_accumulator_12 = address_word_19 + 0x80u;

        matrix_accumulator_13 = packed_half_132 + 0x80u;

        packed_half_7 = permute_signed(matrix_accumulator_16, packed_half_7, 0x7610u);
    }

    uniform_address_word_6 = 0u;

    if (!condition_3) {
        warp_lane_17 = (matrix_accumulator_14 << 0xcu) + 0x80u;

        matrix_accumulator_14 = packed_half_132 + 0x80u;

        condition_4 = uint32_t(warp_lane_17) != uint32_t(matrix_accumulator_12);

        condition_1 = uint32_t(matrix_accumulator_14) != uint32_t(matrix_accumulator_13);

        matrix_accumulator_12 = parameters[parameter_input_low_0];
        matrix_accumulator_13 = parameters[parameter_input_high_1];

        matrix_accumulator_14 = !condition_4 ? packed_half_7 : 0u;
    }

    uniform_index_word_12 = (uniform_index_word_10 << 0x18u) + uniform_index_word_11;

    if (!condition_3) {
        matrix_accumulator_14 = !condition_1 ? matrix_accumulator_14 : 0u;

        packed_half_7 = permute_signed(matrix_accumulator_14, packed_half_7, 0x7610u);
    }

    uniform_address_word_4 = uniform_index_word_12;

    uniform_address_word_5 = uniform_address_word_6;

    packed_half_7 = packed_half_7 & 0xffffu;

    split_address_words(matrix_accumulator_16, warp_lane_17,
                        combine_words(uniform_address_word_4, uniform_address_word_5));

    condition_1 = uint32_t(packed_half_7) != 0u;

    if (!condition_3) {
        split_address_words(address_word_20, address_word_21,
                            int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));
    }

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        uint64_t(uint32_t(packed_half_132)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_16, warp_lane_17));
    if (!condition_1) {
        if (!condition_3) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            warp_lane_17 = parameters[parameter_input_high_1];
        }

        split_address_words(address_word_20, address_word_21, 0ull);

        packed_half_7 = (packed_half_124 << 0xau) + uniform_index_word_12;

        if (!condition_3) {
            split_address_words(address_word_20, address_word_21,
                                int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_16, warp_lane_17));
        }
        if (condition_3) {
            matrix_accumulator_14 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            warp_lane_17 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, warp_lane_17, 0x5410u);

            warp_lane_17 = matrix_accumulator_16;

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + packed_half_7;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_14 = address_word_20;

                uniform_address_word_4 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_15 = address_word_21;

                uniform_address_word_6 = packed_half_7;

                uniform_address_word_7 = (uniform_index_word_10 << 0x18u) + uniform_address_word_4;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_4 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_4 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_3) {
            matrix_accumulator_14 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            warp_lane_17 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, warp_lane_17, 0x5410u);

            warp_lane_17 = matrix_accumulator_16;

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + packed_half_7;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14 + 0x200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_3) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            matrix_accumulator_14 = packed_half_4 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_20, address_word_21, 0ull);

            matrix_accumulator_14 = (matrix_accumulator_14 << 0xcu) + 0x80u;

            address_word_19 = !condition_2 ? matrix_accumulator_14 : 0u;

            if (!condition_3) {
                split_address_words(address_word_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = packed_half_7;

                uniform_address_word_5 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_6 = address_word_20;

                uniform_address_word_13 = 0x20u;

                uniform_address_word_7 = address_word_21;

                uniform_address_word_5 = (uniform_index_word_10 << 0x18u) + uniform_address_word_5;

                packed_half_7 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_4 = matrix_accumulator_12;

            uniform_address_word_7 = uniform_index_word_11 + 0x6000u;

            uniform_address_word_5 = matrix_accumulator_13;

            uniform_address_word_13 = 0x40u;

            uniform_address_word_14 = address_word_20;

            uniform_address_word_7 = (uniform_index_word_10 << 0x18u) + uniform_address_word_7;

            uniform_address_word_15 = address_word_21;

            packed_half_7 = 0x400u;

            if (!condition_3) {
                split_address_words(matrix_accumulator_16, warp_lane_17,
                                    combine_words(uniform_address_word_4, uniform_address_word_5));
            }

            split_address_words(uniform_address_word_4, uniform_address_word_5, 0ull);

            uniform_address_word_4 =
                broadcast_warp_word(matrix_accumulator_16, !condition_3, uniform_address_word_4);

            uniform_address_word_5 =
                broadcast_warp_word(warp_lane_17, !condition_3, uniform_address_word_5);

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_13 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    packed_half_7 = packed_half_4 + 0x4u;

    matrix_accumulator_14 = packed_half_11 + 0x4000u;

    matrix_accumulator_16 = 0x80u;

    address_word_20 = 0x1u;

    warp_lane_17 = 0x80u;

    condition_5 = int32_t(packed_half_7) >= int32_t(packed_half_10);

    condition_4 = int32_t(packed_half_7) >= int32_t(packed_half_10);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    packed_half_131 = packed_half_132 + 0x400u;

    address_word_21 = !condition_4 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_14 = permute_signed(0u, matrix_accumulator_14, 0x7610u);

    if (!condition_5) {
        matrix_accumulator_18 = packed_half_7 + packed_half_125;

        matrix_accumulator_16 = address_word_21 + 0x80u;

        warp_lane_17 = packed_half_132 + 0x480u;

        matrix_accumulator_14 = permute_signed(address_word_20, matrix_accumulator_14, 0x7610u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x1000u);

        address_word_19 = (matrix_accumulator_18 << 0xcu) + 0x80u;

        matrix_accumulator_18 = packed_half_132 + 0x480u;

        condition_6 = uint32_t(address_word_19) != uint32_t(matrix_accumulator_16);

        condition_1 = uint32_t(matrix_accumulator_18) != uint32_t(warp_lane_17);

        matrix_accumulator_16 = parameters[parameter_input_low_0];
        warp_lane_17 = parameters[parameter_input_high_1];

        matrix_accumulator_18 = !condition_6 ? matrix_accumulator_14 : 0u;

        matrix_accumulator_18 = !condition_1 ? matrix_accumulator_18 : 0u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(matrix_accumulator_18, address_word_19, 0ull);

    matrix_accumulator_14 = matrix_accumulator_14 & 0xffffu;

    condition_1 = uint32_t(matrix_accumulator_14) != 0u;

    if (!condition_5) {
        split_address_words(matrix_accumulator_18, address_word_19,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, warp_lane_17));
    }
    if (!condition_1) {
        if (condition_5) {
            matrix_accumulator_14 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            warp_lane_17 = (packed_half_131 << 0x2u) + uniform_index_word_12;

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + warp_lane_17;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_5) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_18, address_word_19, 0ull);

            if (!condition_5) {
                split_address_words(matrix_accumulator_18, address_word_19,
                                    int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = packed_half_131;

                uniform_address_word_5 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_6 = matrix_accumulator_18;

                uniform_address_word_13 = 0x20u;

                uniform_address_word_7 = address_word_19;

                uniform_address_word_5 = (uniform_index_word_10 << 0x18u) + uniform_address_word_5;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = (uniform_address_word_4 << 0x2u) + uniform_index_word_12;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_5) {
            matrix_accumulator_14 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            warp_lane_17 = (packed_half_131 << 0x2u) + uniform_index_word_12;

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = matrix_accumulator_14 * 0x10u + warp_lane_17;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14 + 0x200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_5) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            matrix_accumulator_14 = packed_half_7 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_20, address_word_21, 0ull);

            matrix_accumulator_14 = (matrix_accumulator_14 << 0xcu) + 0x80u;

            address_word_19 = !condition_4 ? matrix_accumulator_14 : 0u;

            if (!condition_5) {
                split_address_words(address_word_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = packed_half_131;

                uniform_address_word_5 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_6 = address_word_20;

                uniform_address_word_13 = 0x20u;

                uniform_address_word_7 = address_word_21;

                uniform_address_word_5 = (uniform_index_word_10 << 0x18u) + uniform_address_word_5;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = (uniform_address_word_4 << 0x2u) + uniform_index_word_12;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_13 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_4 = matrix_accumulator_22;

            uniform_address_word_7 = uniform_index_word_11 + 0x6000u;

            uniform_address_word_5 = matrix_accumulator_23;

            uniform_address_word_13 = 0x40u;

            uniform_address_word_14 = matrix_accumulator_18;

            uniform_address_word_7 = (uniform_index_word_10 << 0x18u) + uniform_address_word_7;

            uniform_address_word_15 = address_word_19;

            matrix_accumulator_14 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_13 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_14 = packed_half_11 + 0x100u;

    if (!condition_3) {
        address_word_19 = packed_half_4 + packed_half_125;
    }

    matrix_accumulator_18 = 0x1u;

    matrix_accumulator_16 = 0x80u;

    warp_lane_17 = 0x80u;

    address_word_21 = !condition_2 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_14 = permute_signed(0u, matrix_accumulator_14, 0x7610u);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_3) {
        warp_lane_17 = packed_half_132 + 0x80u;

        address_word_19 = (address_word_19 << 0xcu) + 0x180u;

        matrix_accumulator_16 = address_word_21 + 0x80u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x2000u);

        matrix_accumulator_18 = packed_half_132 + 0x80u;

        condition_6 = uint32_t(address_word_19) != uint32_t(matrix_accumulator_16);
    }

    packed_half_129 = (packed_half_125 << 0xau) + 0x40u;

    if (!condition_3) {
        condition_1 = uint32_t(matrix_accumulator_18) != uint32_t(warp_lane_17);

        matrix_accumulator_18 = !condition_6 ? matrix_accumulator_14 : 0u;

        matrix_accumulator_16 = parameters[parameter_input_low_0];
        warp_lane_17 = parameters[parameter_input_high_1];

        matrix_accumulator_18 = !condition_1 ? matrix_accumulator_18 : 0u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(matrix_accumulator_18, address_word_19, 0ull);

    matrix_accumulator_14 = matrix_accumulator_14 & 0xffffu;

    condition_1 = uint32_t(matrix_accumulator_14) != 0u;

    if (!condition_3) {
        split_address_words(matrix_accumulator_18, address_word_19,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, warp_lane_17));
    }
    if (!condition_1) {
        if (!condition_3) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            warp_lane_17 = parameters[parameter_input_high_1];
        }

        split_address_words(matrix_accumulator_18, address_word_19, 0ull);

        matrix_accumulator_14 = (packed_half_124 << 0xau) + uniform_index_word_12;

        if (!condition_3) {
            split_address_words(matrix_accumulator_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_16, warp_lane_17));
        }
        if (condition_3) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            address_word_20 = warp_lane_17 * 0x10u + matrix_accumulator_14;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, address_word_20 + 0x2000u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_5 = matrix_accumulator_14;

                uniform_address_word_4 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_14 = matrix_accumulator_18;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_15 = address_word_19;

                uniform_address_word_6 = (uniform_index_word_10 << 0x18u) + uniform_address_word_4;

                matrix_accumulator_16 = 0x200u;

                uniform_address_word_4 = uniform_address_word_5 + 0x2000u;

                uniform_address_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_3) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = warp_lane_17 * 0x10u + matrix_accumulator_14;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14 + 0x2200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_3) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            matrix_accumulator_18 = packed_half_4 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_20, address_word_21, 0ull);

            matrix_accumulator_18 = (matrix_accumulator_18 << 0xcu) + 0x180u;

            address_word_19 = !condition_2 ? matrix_accumulator_18 : 0u;

            if (!condition_3) {
                split_address_words(address_word_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_5 = matrix_accumulator_14;

                uniform_address_word_4 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_14 = address_word_20;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_15 = address_word_21;

                uniform_address_word_6 = (uniform_index_word_10 << 0x18u) + uniform_address_word_4;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = uniform_address_word_5 + 0x2200u;

                uniform_address_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_6 = uniform_index_word_11 + 0x6000u;

            uniform_address_word_4 = matrix_accumulator_22;

            uniform_address_word_16 = 0x40u;

            uniform_address_word_5 = matrix_accumulator_23;

            uniform_address_word_13 = (uniform_index_word_10 << 0x18u) + uniform_address_word_6;

            uniform_address_word_14 = matrix_accumulator_18;

            uniform_address_word_15 = address_word_19;

            uniform_address_word_7 = uniform_address_word_13 + 0x8u;

            matrix_accumulator_14 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_13 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_14 = packed_half_11 + 0x4100u;

    if (!condition_5) {
        address_word_19 = packed_half_7 + packed_half_125;
    }

    matrix_accumulator_18 = 0x1u;

    matrix_accumulator_16 = 0x80u;

    warp_lane_17 = 0x80u;

    address_word_21 = !condition_4 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_14 = permute_signed(0u, matrix_accumulator_14, 0x7610u);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_5) {
        warp_lane_17 = packed_half_132 + 0x480u;

        address_word_19 = (address_word_19 << 0xcu) + 0x180u;

        matrix_accumulator_16 = address_word_21 + 0x80u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x3000u);

        matrix_accumulator_18 = packed_half_132 + 0x480u;

        condition_2 = uint32_t(address_word_19) != uint32_t(matrix_accumulator_16);

        condition_1 = uint32_t(matrix_accumulator_18) != uint32_t(warp_lane_17);

        matrix_accumulator_18 = !condition_2 ? matrix_accumulator_14 : 0u;

        matrix_accumulator_16 = parameters[parameter_input_low_0];
        warp_lane_17 = parameters[parameter_input_high_1];

        matrix_accumulator_18 = !condition_1 ? matrix_accumulator_18 : 0u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(matrix_accumulator_18, address_word_19, 0ull);

    matrix_accumulator_14 = matrix_accumulator_14 & 0xffffu;

    condition_1 = uint32_t(matrix_accumulator_14) != 0u;

    if (!condition_5) {
        split_address_words(matrix_accumulator_18, address_word_19,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, warp_lane_17));
    }
    if (!condition_1) {
        if (!condition_5) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            warp_lane_17 = parameters[parameter_input_high_1];
        }

        split_address_words(matrix_accumulator_18, address_word_19, 0ull);

        matrix_accumulator_14 = (packed_half_131 << 0x2u) + uniform_index_word_12;

        if (!condition_5) {
            split_address_words(matrix_accumulator_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_16, warp_lane_17));
        }
        if (condition_5) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            address_word_20 = warp_lane_17 * 0x10u + matrix_accumulator_14;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, address_word_20 + 0x2000u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_5 = matrix_accumulator_14;

                uniform_address_word_4 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_14 = matrix_accumulator_18;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_15 = address_word_19;

                uniform_address_word_6 = (uniform_index_word_10 << 0x18u) + uniform_address_word_4;

                matrix_accumulator_16 = 0x200u;

                uniform_address_word_4 = uniform_address_word_5 + 0x2000u;

                uniform_address_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_5) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = warp_lane_17 * 0x10u + matrix_accumulator_14;

            warp_lane_17 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14 + 0x2200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_5) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            matrix_accumulator_18 = packed_half_7 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_20, address_word_21, 0ull);

            matrix_accumulator_18 = (matrix_accumulator_18 << 0xcu) + 0x180u;

            address_word_19 = !condition_4 ? matrix_accumulator_18 : 0u;

            if (!condition_5) {
                split_address_words(address_word_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_5 = matrix_accumulator_14;

                uniform_address_word_4 = uniform_index_word_11 + 0x6000u;

                uniform_address_word_14 = address_word_20;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_15 = address_word_21;

                uniform_address_word_6 = (uniform_index_word_10 << 0x18u) + uniform_address_word_4;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = uniform_address_word_5 + 0x2200u;

                uniform_address_word_5 = uniform_address_word_6 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_6 = uniform_index_word_11 + 0x6000u;

            uniform_address_word_4 = matrix_accumulator_22;

            uniform_address_word_16 = 0x40u;

            uniform_address_word_5 = matrix_accumulator_23;

            uniform_address_word_13 = (uniform_index_word_10 << 0x18u) + uniform_address_word_6;

            uniform_address_word_14 = matrix_accumulator_18;

            uniform_address_word_15 = address_word_19;

            uniform_address_word_7 = uniform_address_word_13 + 0x8u;

            matrix_accumulator_14 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_13 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    if (!condition_0) {
        matrix_accumulator_14 = packed_half_124 * 0x20u + packed_half_9;

        matrix_accumulator_16 = matrix_accumulator_14 + 0x20u;

        matrix_accumulator_14 = uint32_t(uint32_t(matrix_accumulator_14) >> 0x7u);

        warp_lane_17 = uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x19u >> 32) + 0x1u;

        matrix_accumulator_23 = min(int32_t(packed_half_8), int32_t(warp_lane_17));

        condition_1 = int32_t(matrix_accumulator_14) >= int32_t(matrix_accumulator_23);
        if (!condition_1) {
            matrix_accumulator_16 = packed_half_129 + 0x40u;

            address_word_19 = uint32_t(uint32_t(matrix_accumulator_16) >> 0x7u);
            do {
                matrix_accumulator_16 = parameters[parameter_wait_optional_low_12];
                warp_lane_17 = parameters[parameter_wait_optional_high_13];

                address_word_21 = matrix_accumulator_14 * 0x8u + address_word_19;

                split_address_words(matrix_accumulator_16, warp_lane_17,
                                    uint64_t(uint32_t(address_word_21)) * uint64_t(uint32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));

                matrix_accumulator_18 =
                    pipeline_read<CheckBounds>(combine_words(matrix_accumulator_16, warp_lane_17),
                                               memory_ranges, error_counts);

                matrix_accumulator_14 = matrix_accumulator_14 + 0x1u;

                condition_2 = uint32_t(matrix_accumulator_14) != uint32_t(matrix_accumulator_23);

                condition_1 = int32_t(matrix_accumulator_18) > -1;
                if (!condition_1) {
                    do {
                        __nanosleep(0x40u);

                        matrix_accumulator_18 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_16, warp_lane_17), memory_ranges,
                            error_counts);

                        condition_1 = int32_t(matrix_accumulator_18) > -1;
                    } while (!condition_1);
                }
            } while (condition_2);
        }
    }
    condition_2 = int32_t(packed_half_4) >= int32_t(packed_half_10);

    condition_3 = int32_t(packed_half_4) >= int32_t(packed_half_10);

    matrix_accumulator_14 = packed_half_11 + 0x200u;

    matrix_accumulator_16 = 0x80u;

    matrix_accumulator_18 = 0x1u;

    warp_lane_17 = 0x80u;

    __syncwarp(__activemask());

    address_word_21 = !condition_3 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_14 = permute_signed(0u, matrix_accumulator_14, 0x7610u);

    __syncthreads();

    if (!condition_2) {
        address_word_19 = packed_half_4 + packed_half_125;

        warp_lane_17 = packed_half_132 + 0x80u;

        matrix_accumulator_16 = address_word_21 + 0x80u;

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_2) {
        matrix_accumulator_18 = packed_half_132 + 0x80u;

        address_word_19 = (address_word_19 << 0xcu) + 0x280u;

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x4000u);

        condition_1 = uint32_t(matrix_accumulator_18) != uint32_t(warp_lane_17);

        condition_4 = uint32_t(address_word_19) != uint32_t(matrix_accumulator_16);

        matrix_accumulator_16 = !condition_4 ? matrix_accumulator_14 : 0u;

        matrix_accumulator_18 = !condition_1 ? matrix_accumulator_16 : 0u;

        matrix_accumulator_16 = parameters[parameter_input_low_0];
        warp_lane_17 = parameters[parameter_input_high_1];

        matrix_accumulator_14 =
            permute_signed(matrix_accumulator_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(matrix_accumulator_18, address_word_19, 0ull);

    matrix_accumulator_14 = matrix_accumulator_14 & 0xffffu;

    condition_1 = uint32_t(matrix_accumulator_14) != 0u;

    if (!condition_2) {
        split_address_words(matrix_accumulator_18, address_word_19,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, warp_lane_17));
    }
    if (!condition_1) {
        matrix_accumulator_23 = 0u;

        if (!condition_2) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            warp_lane_17 = parameters[parameter_input_high_1];
        }

        matrix_accumulator_14 = 0x400u;

        split_address_words(matrix_accumulator_24, matrix_accumulator_25, 0ull);

        if (!condition_2) {
            split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_16, warp_lane_17));
        }

        address_word_19 = (matrix_accumulator_23 << 0x18u) + matrix_accumulator_14;

        matrix_accumulator_18 = packed_half_124 * 0x400u + address_word_19;
        if (condition_2) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_24 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_24 = permute_signed(matrix_accumulator_24, address_word_19, 0x5410u);

            matrix_accumulator_25 = matrix_accumulator_24;

            matrix_accumulator_26 = matrix_accumulator_24;

            matrix_accumulator_27 = matrix_accumulator_24;

            warp_lane_17 = warp_lane_17 * 0x10u + matrix_accumulator_18;

            store_four_shared_words<CheckBounds>(shared_tile, warp_lane_17 + 0x4000u,
                                                 matrix_accumulator_24, matrix_accumulator_25,
                                                 matrix_accumulator_26, matrix_accumulator_27,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                matrix_accumulator_16 = matrix_accumulator_14 + 0x6000u;

                uniform_address_word_5 = matrix_accumulator_23;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_4 = matrix_accumulator_18;

                uniform_address_word_6 = matrix_accumulator_16;

                uniform_address_word_14 = matrix_accumulator_24;

                uniform_address_word_15 = matrix_accumulator_25;

                matrix_accumulator_16 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4000u;

                uniform_address_word_6 = (uniform_address_word_5 << 0x18u) + uniform_address_word_6;

                uniform_address_word_5 = uniform_address_word_6 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            warp_lane_17 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_16 = permute_signed(matrix_accumulator_16, address_word_19, 0x5410u);

            address_word_19 = matrix_accumulator_16;

            matrix_accumulator_14 = warp_lane_17 * 0x10u + matrix_accumulator_18;

            warp_lane_17 = matrix_accumulator_16;

            matrix_accumulator_18 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_14 + 0x4200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                matrix_accumulator_16 = parameters[parameter_input_low_0];
                warp_lane_17 = parameters[parameter_input_high_1];
            }

            address_word_19 = packed_half_4 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_20, address_word_21, 0ull);

            address_word_19 = (address_word_19 << 0xcu) + 0x280u;

            address_word_19 = !condition_3 ? address_word_19 : 0u;

            if (!condition_2) {
                split_address_words(address_word_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_16, warp_lane_17));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                matrix_accumulator_14 = matrix_accumulator_14 + 0x6000u;

                uniform_address_word_5 = matrix_accumulator_23;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_4 = matrix_accumulator_18;

                uniform_address_word_6 = matrix_accumulator_14;

                uniform_address_word_14 = address_word_20;

                uniform_address_word_15 = address_word_21;

                matrix_accumulator_14 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4200u;

                uniform_address_word_6 = (uniform_address_word_5 << 0x18u) + uniform_address_word_6;

                uniform_address_word_5 = uniform_address_word_6 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_7 = 0u;

            uniform_address_word_4 = 0x400u;

            uniform_address_word_5 = matrix_accumulator_23;

            uniform_address_word_6 = uniform_address_word_4 + 0x6000u;

            uniform_address_word_4 = matrix_accumulator_22;

            uniform_address_word_16 = 0x40u;

            uniform_address_word_14 = matrix_accumulator_18;

            uniform_address_word_15 = address_word_19;

            matrix_accumulator_14 = 0x400u;

            uniform_address_word_13 = (uniform_address_word_7 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_13 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_13 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_2 = int32_t(packed_half_7) >= int32_t(packed_half_10);

    condition_3 = int32_t(packed_half_7) >= int32_t(packed_half_10);

    packed_half_11 = packed_half_11 + 0x4200u;

    matrix_accumulator_14 = 0x80u;

    matrix_accumulator_18 = 0x1u;

    matrix_accumulator_16 = 0x80u;

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    address_word_19 = !condition_3 ? packed_half_11 : 0u;

    packed_half_11 = permute_signed(0u, packed_half_11, 0x7610u);

    if (!condition_2) {
        warp_lane_17 = packed_half_7 + packed_half_125;

        matrix_accumulator_14 = address_word_19 + 0x80u;

        packed_half_11 = permute_signed(matrix_accumulator_18, packed_half_11, 0x7610u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x5000u);

        matrix_accumulator_16 = packed_half_132 + 0x480u;

        warp_lane_17 = (warp_lane_17 << 0xcu) + 0x280u;

        address_word_21 = packed_half_132 + 0x480u;

        condition_4 = uint32_t(warp_lane_17) != uint32_t(matrix_accumulator_14);

        condition_1 = uint32_t(address_word_21) != uint32_t(matrix_accumulator_16);
    }

    split_address_words(address_word_20, address_word_21, 0ull);

    if (!condition_2) {
        matrix_accumulator_14 = !condition_4 ? packed_half_11 : 0u;

        matrix_accumulator_16 = parameters[parameter_input_low_0];
        warp_lane_17 = parameters[parameter_input_high_1];

        matrix_accumulator_14 = !condition_1 ? matrix_accumulator_14 : 0u;

        packed_half_11 = permute_signed(matrix_accumulator_14, packed_half_11, 0x7610u);
    }

    packed_half_11 = packed_half_11 & 0xffffu;

    condition_1 = uint32_t(packed_half_11) != 0u;

    if (!condition_2) {
        split_address_words(address_word_20, address_word_21,
                            int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_16, warp_lane_17));
    }
    if (!condition_1) {
        warp_lane_17 = 0u;

        if (!condition_2) {
            matrix_accumulator_12 = parameters[parameter_input_low_0];
            matrix_accumulator_13 = parameters[parameter_input_high_1];
        }

        matrix_accumulator_14 = 0x400u;

        split_address_words(address_word_20, address_word_21, 0ull);

        if (!condition_2) {
            split_address_words(address_word_20, address_word_21,
                                int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));
        }

        matrix_accumulator_16 = (warp_lane_17 << 0x18u) + matrix_accumulator_14;

        matrix_accumulator_16 = packed_half_131 * 0x4u + matrix_accumulator_16;
        if (condition_2) {
            packed_half_11 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            address_word_20 = uniform_address_word_4;

            matrix_accumulator_13 = uniform_address_word_4;

            address_word_20 = permute_signed(address_word_20, matrix_accumulator_13, 0x5410u);

            address_word_21 = address_word_20;

            matrix_accumulator_22 = address_word_20;

            matrix_accumulator_23 = address_word_20;

            packed_half_11 = packed_half_11 * 0x10u + matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(shared_tile, packed_half_11 + 0x4000u,
                                                 address_word_20, address_word_21,
                                                 matrix_accumulator_22, matrix_accumulator_23,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                packed_half_11 = matrix_accumulator_14 + 0x6000u;

                uniform_address_word_5 = warp_lane_17;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_4 = matrix_accumulator_16;

                uniform_address_word_6 = packed_half_11;

                uniform_address_word_14 = address_word_20;

                uniform_address_word_15 = address_word_21;

                packed_half_11 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4000u;

                uniform_address_word_6 = (uniform_address_word_5 << 0x18u) + uniform_address_word_6;

                uniform_address_word_5 = uniform_address_word_6 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            packed_half_11 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 = uniform_address_word_4;

            matrix_accumulator_13 = uniform_address_word_4;

            packed_half_11 = packed_half_11 * 0x10u + matrix_accumulator_16;

            matrix_accumulator_16 =
                permute_signed(matrix_accumulator_12, matrix_accumulator_13, 0x5410u);

            warp_lane_17 = matrix_accumulator_16;

            matrix_accumulator_18 = matrix_accumulator_16;

            address_word_19 = matrix_accumulator_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, packed_half_11 + 0x4200u, matrix_accumulator_16, warp_lane_17,
                matrix_accumulator_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                matrix_accumulator_12 = parameters[parameter_input_low_0];
                matrix_accumulator_13 = parameters[parameter_input_high_1];
            }

            packed_half_11 = packed_half_7 + packed_half_125;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_18, address_word_19, 0ull);

            packed_half_11 = (packed_half_11 << 0xcu) + 0x280u;

            packed_half_11 = !condition_3 ? packed_half_11 : 0u;

            if (!condition_2) {
                split_address_words(
                    matrix_accumulator_18, address_word_19,
                    int64_t(int32_t(packed_half_11)) * int64_t(int32_t(0x4u)) +
                        combine_words(matrix_accumulator_12, matrix_accumulator_13));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                matrix_accumulator_14 = matrix_accumulator_14 + 0x6000u;

                uniform_address_word_5 = warp_lane_17;

                uniform_address_word_7 = 0x20u;

                uniform_address_word_4 = matrix_accumulator_16;

                uniform_address_word_6 = matrix_accumulator_14;

                uniform_address_word_14 = matrix_accumulator_18;

                uniform_address_word_15 = address_word_19;

                packed_half_11 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4200u;

                uniform_address_word_6 = (uniform_address_word_5 << 0x18u) + uniform_address_word_6;

                uniform_address_word_5 = uniform_address_word_6 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, uniform_address_word_6 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_7 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_7 = 0u;

            uniform_address_word_4 = 0x400u;

            uniform_address_word_5 = matrix_accumulator_23;

            uniform_address_word_6 = uniform_address_word_4 + 0x6000u;

            uniform_address_word_4 = matrix_accumulator_22;

            uniform_address_word_16 = 0x40u;

            uniform_address_word_14 = address_word_20;

            uniform_address_word_15 = address_word_21;

            packed_half_11 = 0x400u;

            uniform_address_word_13 = (uniform_address_word_7 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_13 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_16 * 16u, uniform_address_word_13 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_16 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_12 = 0u;

    packed_half_11 = 0x400u;

    matrix_accumulator_42 = packed_half_124 * 0x20u + packed_half_9;

    packed_half_130 = packed_half_124 * 0x2u;

    packed_half_11 = packed_half_11 + 0x6000u;

    packed_half_9 = matrix_accumulator_42 + 0x20u;

    packed_half_9 = uint32_t(combine_words(packed_half_9, 0u) << 0x19u >> 32) + 0x1u;

    packed_half_8 = min(int32_t(packed_half_8), int32_t(packed_half_9));

    packed_half_11 = (matrix_accumulator_12 << 0x18u) + packed_half_11;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = packed_half_11;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_12 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 =
        wait_copy_barrier(shared_tile, packed_half_11, async_tokens[(((packed_half_11) >> 3) & 1)]);
#else
    __syncthreads();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, packed_half_11,
                                            async_tokens[(((packed_half_11) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    packed_half_128 = packed_half_124 * 0x800u;

    packed_half_11 = uint32_t(uint32_t(matrix_accumulator_42) >> 0x7u);

    packed_half_127 = packed_half_124 * 0x200u;

    packed_half_9 = packed_half_132 + 0x80u;

    packed_half_126 = 0u;

    matrix_accumulator_14 = 0u;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_accumulator_16 = 0;
    warp_lane_17 = 0;

    matrix_accumulator_18 = 0;
    address_word_19 = 0;

    address_word_20 = 0;
    address_word_21 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_24 = 0;
    matrix_accumulator_25 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    matrix_accumulator_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_accumulator_34 = 0;
    matrix_accumulator_35 = 0;

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

    matrix_accumulator_38 = 0;
    matrix_accumulator_39 = 0;

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_12 = 0;
    matrix_accumulator_13 = 0;

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    condition_0 = int32_t(packed_half_11) >= int32_t(packed_half_8) || condition_0;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    packed_half_130 = packed_half_130 & 0x7fcu;

    packed_half_128 = packed_half_128 & 0x1ff000u;

    packed_half_127 = packed_half_127 & 0x7fc00u;

#pragma unroll 2
    do {
        index_word_139 = uint32_t(uint32_t(packed_half_126) >> 0x6u);

        packed_half_136 = 0u;

        index_word_137 = 0x400u;

        matrix_fragment_80 = permute_signed(index_word_139, 0u, 0x9910u);

        warp_lane_135 = threadIdx.x & 31u;

        __syncwarp(__activemask());

        matrix_fragment_80 = matrix_fragment_80 * 0xabu;

        condition_1 = uint32_t(packed_half_126) > uint32_t(0x3bfu);

        condition_2 = uint32_t(packed_half_126) > uint32_t(0x33fu);

        matrix_fragment_80 = matrix_fragment_80 & 0xffffu;

        matrix_fragment_80 = uint32_t(uint32_t(matrix_fragment_80) >> 0x9u);

        matrix_fragment_80 = permute_signed(matrix_fragment_80, 0u, 0x9910u);

        matrix_fragment_80 = matrix_fragment_80 * 0x3u;

        matrix_fragment_81 = (packed_half_136 << 0x18u) + index_word_137;

        matrix_fragment_80 = 0u - matrix_fragment_80;

        matrix_fragment_80 = permute_signed(matrix_fragment_80, 0u, 0x7710u);

        index_word_139 = index_word_139 + matrix_fragment_80;

        address_word_138 = index_word_139 * 0x2000u;

        address_word_138 = 0x1fe000u & address_word_138;

        packed_half_134 = matrix_fragment_81 + address_word_138;

        matrix_fragment_80 = packed_half_128 + packed_half_134;

        tile_word_140 = packed_half_127 * 0x4u + packed_half_134;

        matrix_fragment_80 = warp_lane_135 * 0x10u + matrix_fragment_80;

        tile_word_140 = warp_lane_135 * 0x10u + tile_word_140;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_80, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0x400u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0x600u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_88,
                                 matrix_fragment_89);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0x200u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0x800u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_30,
                                 matrix_accumulator_31);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0xc00u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_30,
                                 matrix_accumulator_31);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0xe00u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(address_word_20, address_word_21, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_56, matrix_fragment_57, address_word_20,
                                 address_word_21);

        multiply_accumulate_half(matrix_accumulator_18, address_word_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_18,
                                 address_word_19);

        multiply_accumulate_half(matrix_accumulator_16, warp_lane_17, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_16,
                                 warp_lane_17);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_140 + 0xa00u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(address_word_20, address_word_21, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_72, matrix_fragment_73, address_word_20,
                                 address_word_21);

        multiply_accumulate_half(matrix_accumulator_18, address_word_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_18,
                                 address_word_19);

        multiply_accumulate_half(matrix_accumulator_16, warp_lane_17, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_16,
                                 warp_lane_17);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_44,
                                 matrix_accumulator_45);
        if (!condition_1) {
            matrix_fragment_82 = packed_half_126 + 0x40u;

            matrix_fragment_48 = parameters[parameter_trained_weights_low_6];
            matrix_fragment_49 = parameters[parameter_trained_weights_high_7];

            matrix_fragment_50 = packed_half_129 + packed_half_126;

            matrix_fragment_82 = uint32_t(uint32_t(matrix_fragment_82) >> 0x6u);

            matrix_fragment_51 = matrix_fragment_50 * 0x400u + packed_half_3;

            matrix_fragment_83 = permute_signed(matrix_fragment_82, 0u, 0x9910u);

            matrix_fragment_83 = matrix_fragment_83 * 0xabu;

            matrix_fragment_83 = matrix_fragment_83 & 0xffffu;

            matrix_fragment_83 = uint32_t(uint32_t(matrix_fragment_83) >> 0x9u);

            split_address_words(matrix_fragment_48, matrix_fragment_49,
                                int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_48, matrix_fragment_49));

            matrix_fragment_83 = permute_signed(matrix_fragment_83, 0u, 0x9910u);

            split_address_words(matrix_fragment_80, matrix_fragment_81,
                                int64_t(int32_t(warp_lane_135)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_48, matrix_fragment_49));

            matrix_fragment_83 = matrix_fragment_83 * 0x3u;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81), matrix_fragment_48,
                matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x200ull,
                matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x400ull,
                matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x600ull,
                matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x20000ull,
                matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x20200ull,
                matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x20400ull,
                matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x20600ull,
                matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                memory_ranges, error_counts);

            index_word_137 = index_word_137 + 0x6000u;

            matrix_fragment_81 = 0u - matrix_fragment_83;

            matrix_fragment_83 = (packed_half_136 << 0x18u) + index_word_137;

            matrix_fragment_81 = permute_signed(matrix_fragment_81, 0u, 0x7710u);

            matrix_fragment_82 = matrix_fragment_82 + matrix_fragment_81;

            matrix_fragment_82 = matrix_fragment_82 & 0xffu;

            matrix_fragment_83 = matrix_fragment_82 * 0x8u + matrix_fragment_83;

#if DLSS5_ENABLE_ASYNC_COPIES
            {
                const uint32_t operand_a = matrix_fragment_83;
                async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
                split_address_words(matrix_fragment_80, matrix_fragment_81,
                                    async_tokens[operand_a >> 3 & 1]);
            }
#else
            matrix_fragment_80 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, matrix_fragment_83,
                                            async_tokens[(((matrix_fragment_83) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
            if (!condition_1) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_1 =
                        wait_copy_barrier(shared_tile, matrix_fragment_83,
                                          async_tokens[(((matrix_fragment_83) >> 3) & 1)]);
#else
                    __syncthreads();
                    condition_1 = true;
#endif
                } while (!condition_1);
            }
        }
        matrix_fragment_87 = index_word_139 & 0xffu;

        matrix_fragment_82 = packed_half_126 + 0xc0u;

        index_word_139 = 0u;
        if (!condition_2) {
            condition_1 = (matrix_fragment_82 & 0x40u) != 0;
            if (!condition_1) {
                if (!condition_0) {
                    matrix_fragment_80 = packed_half_125 * 0x400u + matrix_fragment_82;

                    matrix_fragment_83 = packed_half_11;

                    matrix_fragment_84 = uint32_t(uint32_t(matrix_fragment_80) >> 0x7u);
                    do {
                        matrix_fragment_80 = parameters[parameter_wait_optional_low_12];
                        matrix_fragment_81 = parameters[parameter_wait_optional_high_13];

                        matrix_fragment_85 = matrix_fragment_83 * 0x8u + matrix_fragment_84;

                        split_address_words(
                            matrix_fragment_80, matrix_fragment_81,
                            uint64_t(uint32_t(matrix_fragment_85)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));

                        matrix_fragment_85 = pipeline_read<CheckBounds>(
                            combine_words(matrix_fragment_80, matrix_fragment_81), memory_ranges,
                            error_counts);

                        matrix_fragment_83 = matrix_fragment_83 + 0x1u;

                        condition_2 = uint32_t(matrix_fragment_83) != uint32_t(packed_half_8);

                        condition_1 = int32_t(matrix_fragment_85) > -1;
                        if (!condition_1) {
                            do {
                                __nanosleep(0x40u);

                                matrix_fragment_85 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_80, matrix_fragment_81),
                                    memory_ranges, error_counts);

                                condition_1 = int32_t(matrix_fragment_85) > -1;
                            } while (!condition_1);
                        }
                    } while (condition_2);
                }
                __syncwarp(__activemask());

                __syncthreads();
            }
            matrix_fragment_81 = packed_half_125 * 0x400u + matrix_fragment_82;

            condition_2 = int32_t(packed_half_4) >= int32_t(packed_half_10);

            condition_3 = int32_t(packed_half_4) >= int32_t(packed_half_10);

            uniform_address_word_7 = 0u;

            matrix_fragment_86 = matrix_fragment_81 * 0x4u + packed_half_6;

            matrix_fragment_80 = 0x80u;

            matrix_fragment_83 = 0x1u;

            matrix_fragment_82 = permute_signed(0u, matrix_fragment_82, 0x7610u);

            uniform_address_word_6 = 0x400u;

            matrix_fragment_93 = !condition_3 ? matrix_fragment_86 : 0u;

            matrix_fragment_84 = 0x80u;

            uniform_address_word_13 = 0u;

            split_address_words(matrix_fragment_94, matrix_fragment_95, 0ull);

            if (!condition_2) {
                matrix_fragment_81 = matrix_fragment_86 + 0x80u;

                matrix_fragment_80 = matrix_fragment_93 + 0x80u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_83, matrix_fragment_82, 0x7610u);

                matrix_fragment_84 = packed_half_9;

                condition_4 = uint32_t(matrix_fragment_81) != uint32_t(matrix_fragment_80);

                matrix_fragment_80 = parameters[parameter_input_low_0];
                matrix_fragment_81 = parameters[parameter_input_high_1];

                condition_1 = uint32_t(packed_half_9) != uint32_t(matrix_fragment_84);

                matrix_fragment_83 = !condition_4 ? matrix_fragment_82 : 0u;
            }

            uniform_address_word_4 = (uniform_address_word_7 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = uniform_address_word_6 + 0x6000u;

            if (!condition_2) {
                matrix_fragment_83 = !condition_1 ? matrix_fragment_83 : 0u;
            }

            uniform_address_word_6 = (uniform_address_word_7 << 0x18u) + uniform_address_word_6;

            if (!condition_2) {
                matrix_fragment_82 =
                    permute_signed(matrix_fragment_83, matrix_fragment_82, 0x7610u);
            }

            uniform_address_word_5 = uniform_address_word_13;

            split_address_words(matrix_fragment_84, matrix_fragment_85,
                                combine_words(address_word_138, index_word_139) +
                                    combine_words(uniform_address_word_4, uniform_address_word_5));

            matrix_fragment_82 = matrix_fragment_82 & 0xffffu;

            matrix_fragment_87 = (matrix_fragment_87 << 0x3u) + uniform_address_word_6;

            split_address_words(matrix_fragment_84, matrix_fragment_85,
                                uint64_t(uint32_t(packed_half_132)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_84, matrix_fragment_85));

            condition_1 = uint32_t(matrix_fragment_82) != 0u;

            split_address_words(matrix_fragment_82, matrix_fragment_83, 0ull);

            if (!condition_2) {
                split_address_words(matrix_fragment_94, matrix_fragment_95,
                                    combine_words(matrix_fragment_84, matrix_fragment_85));

                split_address_words(matrix_fragment_82, matrix_fragment_83,
                                    int64_t(int32_t(matrix_fragment_93)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_80, matrix_fragment_81));
            }
            if (!condition_1) {
                if (!condition_2) {
                    matrix_fragment_80 = parameters[parameter_input_low_0];
                    matrix_fragment_81 = parameters[parameter_input_high_1];
                }

                split_address_words(matrix_fragment_82, matrix_fragment_83, 0ull);

                matrix_fragment_94 = packed_half_124 * 0x400u + packed_half_134;

                if (!condition_2) {
                    split_address_words(matrix_fragment_82, matrix_fragment_83,
                                        int64_t(int32_t(matrix_fragment_93)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_80, matrix_fragment_81));
                }
                if (condition_2) {
                    matrix_fragment_81 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_83 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_83, 0x5410u);

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    matrix_fragment_92 = matrix_fragment_81 * 0x10u + matrix_fragment_94;

                    matrix_fragment_81 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_92,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    uniform_address_word_4 = 0xffffffffu;

                    condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_1) {
                        uniform_address_word_6 = matrix_fragment_82;

                        uniform_address_word_13 = 0x20u;

                        uniform_address_word_7 = matrix_fragment_83;

                        uniform_address_word_4 = matrix_fragment_94;

                        uniform_address_word_5 = matrix_fragment_87;

                        matrix_fragment_80 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
                if (condition_2) {
                    matrix_fragment_81 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_83 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_83, 0x5410u);

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    matrix_fragment_94 = matrix_fragment_81 * 0x10u + matrix_fragment_94;

                    matrix_fragment_81 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_94 + 0x200u,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    if (!condition_2) {
                        matrix_fragment_80 = parameters[parameter_input_low_0];
                        matrix_fragment_81 = parameters[parameter_input_high_1];
                    }

                    matrix_fragment_83 = matrix_fragment_86 + 0x80u;

                    uniform_address_word_4 = 0xffffffffu;

                    split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                    matrix_fragment_83 = !condition_3 ? matrix_fragment_83 : 0u;

                    if (!condition_2) {
                        split_address_words(
                            matrix_fragment_92, matrix_fragment_93,
                            int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));
                    }

                    condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_1) {
                        uniform_address_word_4 = matrix_fragment_94;

                        uniform_address_word_13 = 0x20u;

                        uniform_address_word_6 = matrix_fragment_92;

                        uniform_address_word_7 = matrix_fragment_93;

                        uniform_address_word_5 = matrix_fragment_87;

                        matrix_fragment_80 = 0x200u;

                        uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_4 = matrix_fragment_94;

                    uniform_address_word_13 = 0x40u;

                    uniform_address_word_5 = matrix_fragment_95;

                    uniform_address_word_14 = matrix_fragment_82;

                    uniform_address_word_15 = matrix_fragment_83;

                    uniform_address_word_7 = matrix_fragment_87;

                    matrix_fragment_80 = 0x400u;

                    uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_13 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            condition_2 = int32_t(packed_half_7) >= int32_t(packed_half_10);

            condition_3 = int32_t(packed_half_7) >= int32_t(packed_half_10);

            matrix_fragment_80 = matrix_fragment_86 + 0x4000u;

            matrix_fragment_81 = 0x80u;

            matrix_fragment_92 = 0x1u;

            matrix_fragment_82 = permute_signed(0u, matrix_fragment_82, 0x7610u);

            split_address_words(matrix_fragment_94, matrix_fragment_95, 0ull);

            matrix_fragment_83 = !condition_3 ? matrix_fragment_80 : 0u;

            if (!condition_2) {
                matrix_fragment_80 = matrix_fragment_86 + 0x4080u;

                matrix_fragment_81 = matrix_fragment_83 + 0x80u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_92, matrix_fragment_82, 0x7610u);

                split_address_words(matrix_fragment_94, matrix_fragment_95,
                                    combine_words(matrix_fragment_84, matrix_fragment_85) +
                                        0x1000u);

                condition_1 = uint32_t(matrix_fragment_80) != uint32_t(matrix_fragment_81);

                matrix_fragment_80 = parameters[parameter_input_low_0];
                matrix_fragment_81 = parameters[parameter_input_high_1];

                matrix_fragment_92 = !condition_1 ? matrix_fragment_82 : 0u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_92, matrix_fragment_82, 0x7610u);
            }

            split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

            matrix_fragment_82 = matrix_fragment_82 & 0xffffu;

            condition_1 = uint32_t(matrix_fragment_82) != 0u;

            if (!condition_2) {
                split_address_words(matrix_fragment_92, matrix_fragment_93,
                                    int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_80, matrix_fragment_81));
            }
            if (!condition_1) {
                if (!condition_2) {
                    matrix_fragment_80 = parameters[parameter_input_low_0];
                    matrix_fragment_81 = parameters[parameter_input_high_1];
                }

                split_address_words(matrix_fragment_84, matrix_fragment_85, 0ull);

                packed_half_134 = packed_half_131 * 0x4u + packed_half_134;

                if (!condition_2) {
                    split_address_words(matrix_fragment_84, matrix_fragment_85,
                                        int64_t(int32_t(matrix_fragment_83)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_80, matrix_fragment_81));
                }
                if (condition_2) {
                    matrix_fragment_81 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_83 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_83, 0x5410u);

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    matrix_fragment_84 = matrix_fragment_81 * 0x10u + packed_half_134;

                    matrix_fragment_81 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_84,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    uniform_address_word_4 = 0xffffffffu;

                    condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_1) {
                        uniform_address_word_6 = matrix_fragment_84;

                        uniform_address_word_13 = 0x20u;

                        uniform_address_word_7 = matrix_fragment_85;

                        uniform_address_word_4 = packed_half_134;

                        uniform_address_word_5 = matrix_fragment_87;

                        matrix_fragment_80 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
                if (condition_2) {
                    matrix_fragment_81 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_83 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_83, 0x5410u);

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    packed_half_134 = matrix_fragment_81 * 0x10u + packed_half_134;

                    matrix_fragment_81 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, packed_half_134 + 0x200u,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    if (!condition_2) {
                        matrix_fragment_80 = parameters[parameter_input_low_0];
                        matrix_fragment_81 = parameters[parameter_input_high_1];
                    }

                    matrix_fragment_83 = matrix_fragment_86 + 0x4080u;

                    uniform_address_word_4 = 0xffffffffu;

                    split_address_words(matrix_fragment_84, matrix_fragment_85, 0ull);

                    matrix_fragment_83 = !condition_3 ? matrix_fragment_83 : 0u;

                    if (!condition_2) {
                        split_address_words(
                            matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));
                    }

                    condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_1) {
                        uniform_address_word_4 = packed_half_134;

                        uniform_address_word_13 = 0x20u;

                        uniform_address_word_6 = matrix_fragment_84;

                        uniform_address_word_7 = matrix_fragment_85;

                        uniform_address_word_5 = matrix_fragment_87;

                        matrix_fragment_80 = 0x200u;

                        uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_4 = matrix_fragment_94;

                    uniform_address_word_13 = 0x40u;

                    uniform_address_word_5 = matrix_fragment_95;

                    uniform_address_word_14 = matrix_fragment_92;

                    uniform_address_word_15 = matrix_fragment_93;

                    uniform_address_word_7 = matrix_fragment_87;

                    matrix_fragment_80 = 0x400u;

                    uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_13 * 16u, matrix_fragment_87, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_13 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_1 = uint32_t(packed_half_126) >= uint32_t(0x3c0u);

        packed_half_126 = packed_half_126 + 0x40u;
    } while (!condition_1);
    packed_half_4 = threadIdx.z;

    packed_half_3 = 0x2b28u;

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                       0x44004400u);

    packed_half_11 = index_word_5 * 0x8u + packed_half_130;

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       0x44004400u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    address_word_2 = packed_half_11 * 0x1000u + address_word_2;

    packed_half_7 = packed_half_11 + 0x1u;

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_9 = packed_half_11 + 0x2u;

    condition_1 = int32_t(packed_half_11) >= int32_t(packed_half_10);

    condition_2 = int32_t(packed_half_7) >= int32_t(packed_half_10);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                       0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    condition_3 = int32_t(packed_half_9) >= int32_t(packed_half_10);

    packed_half_9 = packed_half_11 + 0x3u;

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                                       0x44004400u);

    condition_4 = int32_t(packed_half_9) >= int32_t(packed_half_10);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                                       0x44004400u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                                       0x44004400u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_accumulator_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    condition_0 = (packed_half_133 | packed_half_4) != 0;

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_74), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       0x44004400u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_accumulator_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_78), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       0x44004400u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_76), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x44004400u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                                       0x44004400u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       0x44004400u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_80), 0x3b283b28u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_81), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                       0x44004400u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       0x44004400u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_75), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                                       0x44004400u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(matrix_fragment_77), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       0x44004400u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       0x44004400u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                       0x44004400u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                       0x44004400u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_86), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       0x44004400u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       0x44004400u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       0x44004400u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_94), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_50 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x44004400u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       0x44004400u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       0x44004400u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_accumulator_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3b283b28u);

    packed_half_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_accumulator_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_93), 0x3b283b28u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_51 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x44004400u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_accumulator_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

    packed_half_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3b283b28u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    packed_half_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_124), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    packed_half_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       0x44004400u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       0x44004400u);

    packed_half_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(packed_half_126), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       0x44004400u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    packed_half_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       0x44004400u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       0x44004400u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(packed_half_128), 0x3b283b28u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_129), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_127 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                                       0x44004400u);

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                                       0x44004400u);

    packed_half_127 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(packed_half_127), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    packed_half_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                                       0x44004400u);

    packed_half_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(packed_half_131), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       0x44004400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       0x44004400u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_20), 0x44004400u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

    packed_half_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_21), 0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       0x44004400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                                       0x44004400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_19), 0x44004400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       0x44004400u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

    matrix_fragment_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_10),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

    packed_half_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                       0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_10),
        select_half_operand<2, false, false>(packed_half_6), 0x3b283b28u);

    packed_half_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(warp_lane_17), 0x44004400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_10 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_20),
                            select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    packed_half_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    packed_half_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0xc400c400u);

    packed_half_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_8), 0x3b283b28u);

    matrix_fragment_50 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_15), 0x44004400u);

    packed_half_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    packed_half_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_12), 0x44004400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_11),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    packed_half_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(packed_half_7), 0x3b283b28u);

    packed_half_132 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x44004400u);

    packed_half_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_53),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x44004400u);

    packed_half_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_11),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    packed_half_11 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107), 0x44004400u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

    packed_half_10 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x44004400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_11), 0xc400c400u);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_21),
                            select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x44004400u);

    address_word_20 = parameters[parameter_output_low_4];
    address_word_21 = parameters[parameter_output_high_5];

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0xc400c400u);

    packed_half_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_11),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_60),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_133 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_132), 0xc400c400u);

    packed_half_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_11),
        select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

    warp_lane_135 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_134), 0xc400c400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(packed_half_11), 0x3b283b28u);

    matrix_fragment_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_14), 0x44004400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_133),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    packed_half_136 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(warp_lane_135),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_13), 0x44004400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_42), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(packed_half_9));

    matrix_fragment_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_43), 0x44004400u);

    packed_half_9 = address_word_2 * 0x4u;

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    split_address_words(address_word_20, address_word_21,
                        int64_t(int32_t(packed_half_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_20, address_word_21));

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_60),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_132 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_61),
        select_half_operand<0, false, true>(packed_half_3), 0x37283728u);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<2, false, false>(packed_half_132), 0x3b283b28u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_133),
        select_half_operand<2, false, false>(packed_half_134), 0x3b283b28u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(warp_lane_135),
        select_half_operand<2, false, false>(packed_half_136), 0x3b283b28u);

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    packed_half_124 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                          select_half_operand<2, false, false>(packed_half_124));

    packed_half_125 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                          select_half_operand<2, false, false>(packed_half_125));

    packed_half_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(packed_half_126));

    packed_half_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(packed_half_127));

    packed_half_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(packed_half_128));

    packed_half_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(packed_half_129));

    packed_half_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(packed_half_130));

    packed_half_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(packed_half_131));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    packed_half_4 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_19),
                                        select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(packed_half_8));

    packed_half_6 = multiply_half_pairs(select_half_operand<2, false, false>(warp_lane_17),
                                        select_half_operand<2, false, false>(packed_half_6));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_7));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(packed_half_11));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_14),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_fragment_61));
    if (!condition_1) {
        matrix_accumulator_25 = threadIdx.x & 31u;

        packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                                      select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

        address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                                       select_half_operand<2, false, false>(0u));

        packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                      select_half_operand<2, false, false>(0u));

        warp_lane_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                                     select_half_operand<2, false, false>(0u));

        address_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                                       select_half_operand<2, false, false>(packed_half_8));

        matrix_accumulator_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                          select_half_operand<2, false, false>(matrix_accumulator_12));

        matrix_accumulator_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                          select_half_operand<2, false, false>(packed_half_10));

        packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(address_word_2));

        packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(packed_half_3));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                                       select_half_operand<2, false, false>(warp_lane_17));

        matrix_accumulator_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                          select_half_operand<2, false, false>(address_word_19));

        matrix_accumulator_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(matrix_accumulator_23));

        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_20, address_word_21));

        store_four_words_streaming<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                                packed_half_8, packed_half_9, packed_half_10,
                                                packed_half_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, packed_half_3) + 0x200ull, matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }
    split_address_words(address_word_2, packed_half_3,
                        combine_words(address_word_20, address_word_21) + 0x10000u);

    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                  select_half_operand<2, false, false>(0u));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                   select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(0u));

    address_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_125),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_126),
                                          select_half_operand<2, false, false>(0u));

    warp_lane_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_127),
                                 select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        matrix_accumulator_27 = threadIdx.x & 31u;

        packed_half_9 = permute_signed(packed_half_9, matrix_accumulator_12, 0x5410u);

        packed_half_8 = permute_signed(packed_half_8, matrix_accumulator_13, 0x5410u);

        packed_half_11 = permute_signed(packed_half_11, matrix_accumulator_14, 0x5410u);

        packed_half_10 = permute_signed(packed_half_10, matrix_accumulator_15, 0x5410u);

        warp_lane_17 = permute_signed(warp_lane_17, matrix_accumulator_28, 0x5410u);

        matrix_accumulator_22 =
            permute_signed(matrix_accumulator_22, matrix_accumulator_25, 0x5410u);

        address_word_19 = permute_signed(address_word_19, matrix_accumulator_26, 0x5410u);

        matrix_accumulator_24 =
            permute_signed(matrix_accumulator_24, matrix_accumulator_23, 0x5410u);

        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, packed_half_3));

        split_address_words(matrix_accumulator_14, matrix_accumulator_15,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_20, address_word_21));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_12, matrix_accumulator_13), packed_half_8,
            packed_half_9, packed_half_10, packed_half_11, memory_ranges, error_counts);

        packed_half_8 = matrix_accumulator_24;

        packed_half_9 = address_word_19;

        packed_half_10 = matrix_accumulator_22;

        packed_half_11 = warp_lane_17;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_14, matrix_accumulator_15) + 0x10200ull, packed_half_8,
            packed_half_9, packed_half_10, packed_half_11, memory_ranges, error_counts);
    }
    split_address_words(address_word_20, address_word_21,
                        combine_words(address_word_20, address_word_21) + 0x20000u);

    matrix_accumulator_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_4),
                                          select_half_operand<2, false, false>(0u));

    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                  select_half_operand<2, false, false>(0u));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                   select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_6),
                                  select_half_operand<2, false, false>(0u));

    warp_lane_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                 select_half_operand<2, false, false>(0u));

    packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_7),
                                  select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        address_word_19 = threadIdx.x & 31u;

        packed_half_4 = permute_signed(packed_half_4, packed_half_7, 0x5410u);

        packed_half_11 = permute_signed(packed_half_11, matrix_accumulator_12, 0x5410u);

        packed_half_10 = permute_signed(packed_half_10, matrix_accumulator_13, 0x5410u);

        packed_half_9 = permute_signed(packed_half_9, matrix_fragment_68, 0x5410u);

        packed_half_8 = permute_signed(packed_half_8, matrix_fragment_69, 0x5410u);

        matrix_accumulator_13 = permute_signed(matrix_accumulator_16, warp_lane_17, 0x5410u);

        matrix_accumulator_12 = permute_signed(matrix_fragment_65, matrix_accumulator_14, 0x5410u);

        matrix_accumulator_15 =
            permute_signed(matrix_accumulator_15, matrix_accumulator_18, 0x5410u);

        split_address_words(packed_half_6, packed_half_7,
                            int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_20, address_word_21));

        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, packed_half_3));

        store_four_words_streaming<CheckBounds>(combine_words(packed_half_6, packed_half_7),
                                                packed_half_8, packed_half_9, packed_half_10,
                                                packed_half_11, memory_ranges, error_counts);

        packed_half_8 = matrix_accumulator_15;

        packed_half_9 = matrix_accumulator_12;

        packed_half_10 = matrix_accumulator_13;

        packed_half_11 = packed_half_4;

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, packed_half_3) + 0x10200ull, packed_half_8, packed_half_9,
            packed_half_10, packed_half_11, memory_ranges, error_counts);
    }
    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                  select_half_operand<2, false, false>(0u));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                  select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                  select_half_operand<2, false, false>(0u));

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                                   select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                   select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(0u));

    packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                  select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                  select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        warp_lane_17 = threadIdx.x & 31u;

        packed_half_11 = permute_signed(packed_half_11, matrix_fragment_50, 0x5410u);

        packed_half_10 = permute_signed(packed_half_10, matrix_fragment_49, 0x5410u);

        packed_half_9 = permute_signed(packed_half_9, address_word_2, 0x5410u);

        packed_half_8 = permute_signed(packed_half_8, packed_half_3, 0x5410u);

        matrix_accumulator_15 = permute_signed(matrix_accumulator_15, packed_half_4, 0x5410u);

        matrix_accumulator_14 = permute_signed(matrix_accumulator_14, packed_half_7, 0x5410u);

        matrix_accumulator_13 = permute_signed(matrix_accumulator_13, matrix_fragment_58, 0x5410u);

        matrix_accumulator_12 = permute_signed(matrix_accumulator_12, matrix_fragment_57, 0x5410u);

        split_address_words(address_word_20, address_word_21,
                            int64_t(int32_t(warp_lane_17)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_20, address_word_21));

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_20, address_word_21) + 0x10000ull, packed_half_8,
            packed_half_9, packed_half_10, packed_half_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_20, address_word_21) + 0x10200ull, matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }
    __syncwarp(__activemask());

    __syncthreads();

    if (condition_0) {
        return;
    }

    address_word_2 = parameters[parameter_publication_optional_low_14];
    packed_half_3 = parameters[parameter_publication_optional_high_15];

    index_word_5 = index_word_5 * 0x20u + address_word_0;

    split_address_words(address_word_2, packed_half_3,
                        int64_t(int32_t(index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, packed_half_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__ __launch_bounds__(
    128) void phoenix_global_attention_transformer_feed_forward_expand(KernelArguments parameters) {
    run_global_attention_transformer_feed_forward_expand<false>(parameters.words, nullptr, nullptr);
}
