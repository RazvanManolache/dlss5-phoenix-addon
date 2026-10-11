// DLSS5 Phoenix optimized kernel: global_attention/transformer/feed_forward_expand_publish.cu
// Coarsest 1024-channel global-transformer stage. Attention spans the full latent token set.
// Projection and feed-forward modules retain their explicit publication and synchronization boundaries.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 25624
#include "../../../common/device_memory_scalar.cuh"
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_output_low_4 = 4,
    parameter_output_high_5 = 5,
    parameter_trained_weights_low_6 = 6,
    parameter_trained_weights_high_7 = 7,
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
__device__ __forceinline__ void run_global_attention_transformer_feed_forward_expand_publish(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t address_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t packed_half_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t packed_half_4 = 0;
    uint32_t address_word_5 = 0;
    uint32_t packed_half_6 = 0;
    uint32_t packed_half_7 = 0;
    uint32_t packed_half_8 = 0;
    uint32_t packed_half_9 = 0;
    uint32_t address_word_10 = 0;
    uint32_t packed_half_11 = 0;
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t matrix_accumulator_14 = 0;
    uint32_t matrix_accumulator_15 = 0;
    uint32_t address_word_16 = 0;
    uint32_t matrix_accumulator_17 = 0;
    uint32_t address_word_18 = 0;
    uint32_t address_word_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
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
    uint32_t index_word_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t packed_half_136 = 0;
    uint32_t packed_half_137 = 0;
    uint32_t tile_word_138 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_reused_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
    uint32_t uniform_address_word_10 = 0;
    uint32_t uniform_address_word_11 = 0;
    uint32_t uniform_address_word_12 = 0;
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
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    packed_half_4 = parameters[parameter_LH_16];
    address_word_5 = parameters[parameter_LW_17];

    matrix_accumulator_14 = blockIdx.x;

    uniform_reused_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    packed_half_124 = threadIdx.y;

    packed_half_131 = threadIdx.x;

    matrix_accumulator_13 = blockIdx.z;

    matrix_accumulator_12 = threadIdx.x & 31u;

    address_word_5 = address_word_5 * packed_half_4 + 0xffffffffu;

    packed_half_8 = uint32_t(int32_t(address_word_5) >> 0x1fu);

    address_word_0 = uint32_t(combine_words(packed_half_8, 0u) << 0x7u >> 32) + address_word_5;

    matrix_accumulator_15 = uint32_t(int32_t(address_word_0) >> 0x7u);

    packed_half_131 = packed_half_124 | packed_half_131;
    condition_1 = packed_half_131 != 0;

    packed_half_4 = matrix_accumulator_15 + 0x1u;

    packed_half_7 = int32_t(packed_half_4) < 0 ? 0u - packed_half_4 : packed_half_4;

    address_word_10 = int32_t(packed_half_4) < 0 ? 0u - packed_half_4 : packed_half_4;

    address_word_0 = __float_as_uint(__int2float_ru(packed_half_7));

    uniform_condition_0 = vote_warp_predicate<true>(condition_1);

    address_word_5 = uint32_t(combine_words(packed_half_8, 0u) << 0x5u >> 32) + address_word_5;

    uniform_condition_1 = vote_warp_predicate<true>(condition_1);

    uniform_condition_2 = vote_warp_predicate<true>(condition_1);

    address_word_0 = reciprocal_bits(address_word_0);

    packed_half_2 = address_word_0 + 0xffffffeu;

    address_word_3 = __float2uint_rz(__uint_as_float(packed_half_2));

    packed_half_2 = 0u;

    packed_half_6 = 0u - address_word_3;

    packed_half_9 = packed_half_6 * packed_half_7;

    packed_half_6 =
        int32_t(matrix_accumulator_14) < 0 ? 0u - matrix_accumulator_14 : matrix_accumulator_14;

    address_word_3 = uint32_t(uint64_t(address_word_3) * packed_half_9 +
                                  combine_words(packed_half_2, address_word_3) >>
                              32);

    packed_half_9 = 0u - address_word_10;

    packed_half_2 = packed_half_6;

    address_word_0 = uint32_t(uint64_t(address_word_3) * packed_half_2 >> 32);

    if (!condition_1) {
        address_word_3 = blockDim.x;
    }

    packed_half_2 = address_word_0 * packed_half_9 + packed_half_2;

    packed_half_9 = 0u;

    condition_2 = uint32_t(packed_half_7) > uint32_t(packed_half_2);

    if (!condition_1) {
        packed_half_6 = blockDim.y;
    }

    if (!condition_2) {
        packed_half_2 = packed_half_2 + (0u - packed_half_7);

        address_word_0 = address_word_0 + 0x1u;
    }

    condition_0 = uint32_t(packed_half_2) >= uint32_t(packed_half_7);

    packed_half_2 = matrix_accumulator_14 ^ packed_half_4;

    condition_2 = int32_t(packed_half_2) >= int32_t(0u);

    if (!condition_1) {
        address_word_10 = address_word_3 * packed_half_6;
    }

    packed_half_2 = 0x400u;

    packed_half_6 = parameters[parameter_trained_weights_low_6];
    packed_half_7 = parameters[parameter_trained_weights_high_7];

    address_word_3 = packed_half_124 * 0x40u;

    uniform_address_word_4 =
        broadcast_warp_word(address_word_10, !condition_1, uniform_address_word_4);

    if (condition_0) {
        address_word_0 = address_word_0 + 0x1u;
    }

    condition_0 = int32_t(packed_half_4) != int32_t(0u);

    address_word_3 = address_word_3 & 0x40u;

    if (!condition_2) {
        address_word_0 = 0u - address_word_0;
    }

    if (!uniform_condition_0) {
        uniform_address_word_4 = 0u - uniform_address_word_4 + 0x100000u;

        uniform_address_word_5 = uniform_address_word_4 << (0xbu & 31u);

        uniform_address_word_4 = uniform_address_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_1);

    if (!condition_0) {
        address_word_0 = ~packed_half_4;
    }

    if (!condition_1) {
        packed_half_4 = packed_half_2 + 0x6000u;
    }

    address_word_3 = address_word_0 * 0x80u + address_word_3;

    if (!condition_1) {
        address_word_10 = (packed_half_9 << 0x18u) + packed_half_4;
    }

    packed_half_4 = address_word_3 * 0x8u;

    uniform_address_word_6 =
        broadcast_warp_word(address_word_10, !condition_1, uniform_address_word_6);

    packed_half_11 = matrix_accumulator_13 * 0x100000u + packed_half_4;

    split_address_words(packed_half_6, packed_half_7,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_6, packed_half_7));

    split_address_words(address_word_10, packed_half_11,
                        int64_t(int32_t(matrix_accumulator_12)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_6, packed_half_7));

    packed_half_6 = address_word_5 & 0xffffffe0u;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    address_word_5 = matrix_accumulator_15 * address_word_0 + address_word_0;

    packed_half_6 = packed_half_6 + 0x20u;

    address_word_5 = 0u - address_word_5 + matrix_accumulator_14;

    index_word_133 = uint32_t(int32_t(packed_half_6) >> 0x4u);

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    packed_half_6 = address_word_5 * 0x8u + packed_half_124;

    condition_2 = int32_t(packed_half_6) >= int32_t(index_word_133);

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x10u);
#endif
    }

    __syncthreads();

    packed_half_7 = packed_half_6 * 0x1000u;

    condition_1 = int32_t(packed_half_6) >= int32_t(index_word_133);

    matrix_accumulator_14 = matrix_accumulator_13 * 0x1000u + packed_half_7;

    if (!condition_2) {
        matrix_accumulator_15 = packed_half_6 + matrix_accumulator_13;
    }

    address_word_21 = !condition_1 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_17 = 0x1u;

    packed_half_129 = packed_half_124 * 0x100u;

    if (!condition_2) {
        matrix_accumulator_15 = (matrix_accumulator_15 << 0xcu) + 0x80u;
    }

    packed_half_8 = permute_signed(0u, packed_half_8, 0x7610u);

    if (!condition_2) {
        address_word_16 = packed_half_129 + 0x80u;
    }

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11),
                                        matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
                                        matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x200ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x400ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x600ull,
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x20000ull,
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x20200ull,
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x20400ull,
                                        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
                                        matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_10, packed_half_11) + 0x20600ull,
                                        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                        matrix_fragment_79, memory_ranges, error_counts);

    if (!condition_2) {
        packed_half_8 = permute_signed(matrix_accumulator_17, packed_half_8, 0x7610u);
    }

    packed_half_134 = (packed_half_9 << 0x18u) + packed_half_2;

    address_word_19 = 0u;

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    address_word_10 = 0x80u;

    if (!condition_2) {
        address_word_10 = address_word_21 + 0x80u;
    }

    packed_half_11 = 0x80u;

    if (!condition_2) {
        packed_half_11 = packed_half_129 + 0x80u;

        condition_3 = uint32_t(matrix_accumulator_15) != uint32_t(address_word_10);
    }

    matrix_accumulator_15 = 0u;

    if (!condition_2) {
        condition_0 = uint32_t(address_word_16) != uint32_t(packed_half_11);

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_10 = !condition_3 ? packed_half_8 : 0u;

        address_word_10 = !condition_0 ? address_word_10 : 0u;

        packed_half_8 = permute_signed(address_word_10, packed_half_8, 0x7610u);
    }

    packed_half_8 = packed_half_8 & 0xffffu;

    address_word_10 = packed_half_134;

    packed_half_11 = address_word_19;

    condition_0 = uint32_t(packed_half_8) != 0u;

    split_address_words(address_word_10, packed_half_11,
                        uint64_t(uint32_t(packed_half_129)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_10, packed_half_11));

    if (!condition_2) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (!condition_2) {
            address_word_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];
        }

        split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

        address_word_19 = packed_half_124 * 0x400u + packed_half_134;

        if (!condition_2) {
            split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_16, matrix_accumulator_17));
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            packed_half_8 = matrix_accumulator_12 * 0x10u + address_word_19;

            matrix_accumulator_20 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            matrix_accumulator_20 =
                permute_signed(matrix_accumulator_20, matrix_accumulator_17, 0x5410u);

            address_word_21 = matrix_accumulator_20;

            matrix_accumulator_22 = matrix_accumulator_20;

            matrix_accumulator_23 = matrix_accumulator_20;

            store_four_shared_words<CheckBounds>(shared_tile, packed_half_8, matrix_accumulator_20,
                                                 address_word_21, matrix_accumulator_22,
                                                 matrix_accumulator_23, shared_bytes, memory_ranges,
                                                 error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_5 = 0u;

                packed_half_8 = packed_half_2 + 0x6000u;

                uniform_address_word_10 =
                    broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

                uniform_address_word_11 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

                uniform_address_word_4 =
                    broadcast_warp_word(packed_half_8, true, uniform_address_word_4);

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_6);

                packed_half_8 = 0x200u;

                uniform_address_word_7 = (uniform_address_word_5 << 0x18u) + uniform_address_word_4;

                uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_4 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_4 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            packed_half_8 = matrix_accumulator_12 * 0x10u + address_word_19;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, packed_half_8 + 0x200u, address_word_16, matrix_accumulator_17,
                address_word_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            packed_half_8 = packed_half_6 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

            packed_half_8 = (packed_half_8 << 0xcu) + 0x80u;

            address_word_21 = !condition_1 ? packed_half_8 : 0u;

            if (!condition_2) {
                split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                                    int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_6 = 0u;

                packed_half_8 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_4);

                uniform_address_word_10 =
                    broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

                uniform_address_word_5 =
                    broadcast_warp_word(packed_half_8, true, uniform_address_word_5);

                uniform_address_word_11 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

                packed_half_8 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

                uniform_address_word_5 = (uniform_address_word_6 << 0x18u) + uniform_address_word_5;

                uniform_address_word_6 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_6 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_4 =
                broadcast_warp_word(address_word_10, true, uniform_address_word_4);

            uniform_address_word_6 = 0u;

            uniform_address_word_5 =
                broadcast_warp_word(packed_half_11, true, uniform_address_word_5);

            uniform_address_word_12 = 0x40u;

            packed_half_8 = packed_half_2 + 0x6000u;

            uniform_address_word_10 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

            uniform_address_word_11 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

            uniform_address_word_7 =
                broadcast_warp_word(packed_half_8, true, uniform_address_word_7);

            packed_half_8 = 0x400u;

            if (!condition_2) {
                split_address_words(address_word_16, matrix_accumulator_17,
                                    combine_words(uniform_address_word_4, uniform_address_word_5));
            }

            split_address_words(uniform_address_word_4, uniform_address_word_5, 0ull);

            uniform_address_word_4 =
                broadcast_warp_word(address_word_16, !condition_2, uniform_address_word_4);

            uniform_address_word_5 =
                broadcast_warp_word(matrix_accumulator_17, !condition_2, uniform_address_word_5);

            uniform_address_word_7 = (uniform_address_word_6 << 0x18u) + uniform_address_word_7;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    packed_half_8 = packed_half_6 + 0x4u;

    address_word_16 = matrix_accumulator_14 + 0x4000u;

    matrix_accumulator_17 = 0x80u;

    matrix_accumulator_22 = 0x1u;

    address_word_18 = permute_signed(0u, address_word_18, 0x7610u);

    condition_4 = int32_t(packed_half_8) >= int32_t(index_word_133);

    condition_3 = int32_t(packed_half_8) >= int32_t(index_word_133);

    packed_half_127 = packed_half_129 + 0x400u;

    address_word_21 = !condition_3 ? address_word_16 : 0u;

    address_word_16 = 0x80u;

    if (!condition_4) {
        address_word_19 = packed_half_8 + matrix_accumulator_13;

        matrix_accumulator_17 = address_word_21 + 0x80u;

        address_word_16 = packed_half_129 + 0x480u;

        address_word_18 = permute_signed(matrix_accumulator_22, address_word_18, 0x7610u);
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_4) {
        matrix_accumulator_20 = (address_word_19 << 0xcu) + 0x80u;

        address_word_19 = packed_half_129 + 0x480u;

        condition_5 = uint32_t(matrix_accumulator_20) != uint32_t(matrix_accumulator_17);

        condition_0 = uint32_t(address_word_19) != uint32_t(address_word_16);

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_19 = !condition_5 ? address_word_18 : 0u;

        address_word_19 = !condition_0 ? address_word_19 : 0u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);
    }

    address_word_18 = address_word_18 & 0xffffu;

    condition_0 = uint32_t(address_word_18) != 0u;

    split_address_words(address_word_18, address_word_19, 0ull);

    if (!condition_4) {
        split_address_words(address_word_18, address_word_19,
                            combine_words(address_word_10, packed_half_11) + 0x1000u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_17 = packed_half_127 * 0x4u + packed_half_134;

            address_word_16 = matrix_accumulator_12 * 0x10u + matrix_accumulator_17;

            matrix_accumulator_20 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            matrix_accumulator_20 = permute_signed(matrix_accumulator_20, address_word_19, 0x5410u);

            address_word_21 = matrix_accumulator_20;

            matrix_accumulator_22 = matrix_accumulator_20;

            matrix_accumulator_23 = matrix_accumulator_20;

            store_four_shared_words<CheckBounds>(shared_tile, address_word_16,
                                                 matrix_accumulator_20, address_word_21,
                                                 matrix_accumulator_22, matrix_accumulator_23,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_4) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_18, address_word_19, 0ull);

            if (!condition_4) {
                split_address_words(address_word_18, address_word_19,
                                    int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_5 = 0u;

                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_10 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_10);

                uniform_address_word_11 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_11);

                uniform_address_word_4 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_4);

                address_word_16 = packed_half_127 * 0x4u + packed_half_134;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_6);

                address_word_16 = 0x200u;

                uniform_address_word_7 = (uniform_address_word_5 << 0x18u) + uniform_address_word_4;

                uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_4 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_4 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_17 = packed_half_127 * 0x4u + packed_half_134;

            matrix_accumulator_20 = matrix_accumulator_12 * 0x10u + matrix_accumulator_17;

            address_word_16 = uniform_address_word_4;

            address_word_19 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, address_word_19, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_20 + 0x200u, address_word_16, matrix_accumulator_17,
                address_word_18, address_word_19, shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_4) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            address_word_18 = packed_half_8 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_20, address_word_21, 0ull);

            address_word_18 = (address_word_18 << 0xcu) + 0x80u;

            address_word_19 = !condition_3 ? address_word_18 : 0u;

            if (!condition_4) {
                split_address_words(matrix_accumulator_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_6 = 0u;

                matrix_accumulator_17 = packed_half_127 * 0x4u + packed_half_134;

                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_10 =
                    broadcast_warp_word(matrix_accumulator_20, true, uniform_address_word_10);

                uniform_address_word_11 =
                    broadcast_warp_word(address_word_21, true, uniform_address_word_11);

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_17, true, uniform_address_word_4);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                address_word_16 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

                uniform_address_word_5 = (uniform_address_word_6 << 0x18u) + uniform_address_word_5;

                uniform_address_word_6 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_6 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_10, uniform_address_word_11),
                    uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_6 = 0u;

            address_word_16 = packed_half_2 + 0x6000u;

            uniform_address_word_4 =
                broadcast_warp_word(address_word_18, true, uniform_address_word_4);

            uniform_address_word_12 = 0x40u;

            uniform_address_word_5 =
                broadcast_warp_word(address_word_19, true, uniform_address_word_5);

            uniform_address_word_7 =
                broadcast_warp_word(address_word_16, true, uniform_address_word_7);

            uniform_address_word_10 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

            uniform_address_word_11 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

            address_word_16 = 0x400u;

            uniform_address_word_7 = (uniform_address_word_6 << 0x18u) + uniform_address_word_7;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_16 = matrix_accumulator_14 + 0x100u;

    if (!condition_2) {
        matrix_accumulator_20 = packed_half_6 + matrix_accumulator_13;
    }

    address_word_19 = 0x1u;

    matrix_accumulator_17 = 0x80u;

    address_word_18 = permute_signed(0u, address_word_18, 0x7610u);

    packed_half_128 = matrix_accumulator_13 * 0x400u;

    address_word_21 = !condition_1 ? address_word_16 : 0u;

    if (!condition_2) {
        matrix_accumulator_20 = (matrix_accumulator_20 << 0xcu) + 0x180u;
    }

    address_word_16 = 0x80u;

    if (!condition_2) {
        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_2) {
        matrix_accumulator_17 = address_word_21 + 0x80u;

        address_word_19 = packed_half_129 + 0x80u;

        address_word_16 = packed_half_129 + 0x80u;
    }

    packed_half_132 = packed_half_128 + 0x40u;

    if (!condition_2) {
        condition_5 = uint32_t(matrix_accumulator_20) != uint32_t(matrix_accumulator_17);

        condition_0 = uint32_t(address_word_19) != uint32_t(address_word_16);

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_19 = !condition_5 ? address_word_18 : 0u;

        address_word_19 = !condition_0 ? address_word_19 : 0u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);
    }

    address_word_18 = address_word_18 & 0xffffu;

    condition_0 = uint32_t(address_word_18) != 0u;

    split_address_words(address_word_18, address_word_19, 0ull);

    if (!condition_2) {
        split_address_words(address_word_18, address_word_19,
                            combine_words(address_word_10, packed_half_11) + 0x2000u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (!condition_2) {
            address_word_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];
        }

        split_address_words(address_word_18, address_word_19, 0ull);

        matrix_accumulator_23 = packed_half_124 * 0x400u + packed_half_134;

        if (!condition_2) {
            split_address_words(address_word_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_16, matrix_accumulator_17));
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_20 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_20 + 0x2000u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x2000u;

                uniform_address_word_5 = uniform_address_word_5 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_23 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_23 + 0x2200u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            address_word_18 = packed_half_6 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_20, address_word_21, 0ull);

            address_word_18 = (address_word_18 << 0xcu) + 0x180u;

            address_word_19 = !condition_1 ? address_word_18 : 0u;

            if (!condition_2) {
                split_address_words(matrix_accumulator_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(matrix_accumulator_20, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_21, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x2200u;

                uniform_address_word_5 = uniform_address_word_5 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            address_word_16 = packed_half_2 + 0x6000u;

            uniform_address_word_4 =
                broadcast_warp_word(address_word_18, true, uniform_address_word_4);

            uniform_address_word_12 = 0x40u;

            uniform_address_word_5 =
                broadcast_warp_word(address_word_19, true, uniform_address_word_5);

            address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

            uniform_address_word_10 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

            uniform_address_word_7 =
                broadcast_warp_word(address_word_16, true, uniform_address_word_7);

            uniform_address_word_11 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

            matrix_accumulator_17 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_7 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, address_word_16 + 0x8u, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_16 = matrix_accumulator_14 + 0x4100u;

    if (!condition_4) {
        matrix_accumulator_20 = packed_half_8 + matrix_accumulator_13;
    }

    address_word_19 = 0x1u;

    matrix_accumulator_17 = 0x80u;

    address_word_18 = permute_signed(0u, address_word_18, 0x7610u);

    address_word_21 = !condition_3 ? address_word_16 : 0u;

    if (!condition_4) {
        matrix_accumulator_20 = (matrix_accumulator_20 << 0xcu) + 0x180u;
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    address_word_16 = 0x80u;

    if (!condition_4) {
        matrix_accumulator_17 = address_word_21 + 0x80u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);

        address_word_19 = packed_half_129 + 0x480u;

        address_word_16 = packed_half_129 + 0x480u;

        condition_5 = uint32_t(matrix_accumulator_20) != uint32_t(matrix_accumulator_17);

        condition_0 = uint32_t(address_word_19) != uint32_t(address_word_16);

        address_word_19 = !condition_5 ? address_word_18 : 0u;

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_19 = !condition_0 ? address_word_19 : 0u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);
    }

    address_word_18 = address_word_18 & 0xffffu;

    condition_0 = uint32_t(address_word_18) != 0u;

    split_address_words(address_word_18, address_word_19, 0ull);

    if (!condition_4) {
        split_address_words(address_word_18, address_word_19,
                            combine_words(address_word_10, packed_half_11) + 0x3000u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (!condition_4) {
            address_word_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];
        }

        split_address_words(address_word_18, address_word_19, 0ull);

        matrix_accumulator_23 = packed_half_127 * 0x4u + packed_half_134;

        if (!condition_4) {
            split_address_words(address_word_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_16, matrix_accumulator_17));
        }
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_20 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_20 + 0x2000u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x2000u;

                uniform_address_word_5 = uniform_address_word_5 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_23 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_23 + 0x2200u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            if (!condition_4) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            address_word_18 = packed_half_8 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_20, address_word_21, 0ull);

            address_word_18 = (address_word_18 << 0xcu) + 0x180u;

            address_word_19 = !condition_3 ? address_word_18 : 0u;

            if (!condition_4) {
                split_address_words(matrix_accumulator_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(matrix_accumulator_20, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_21, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x2200u;

                uniform_address_word_5 = uniform_address_word_5 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x8u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            address_word_16 = packed_half_2 + 0x6000u;

            uniform_address_word_4 =
                broadcast_warp_word(address_word_18, true, uniform_address_word_4);

            uniform_address_word_12 = 0x40u;

            uniform_address_word_5 =
                broadcast_warp_word(address_word_19, true, uniform_address_word_5);

            address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

            uniform_address_word_10 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

            uniform_address_word_7 =
                broadcast_warp_word(address_word_16, true, uniform_address_word_7);

            uniform_address_word_11 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

            matrix_accumulator_17 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_7 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, address_word_16 + 0x8u, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_16 = matrix_accumulator_14 + 0x200u;

    if (!condition_2) {
        matrix_accumulator_20 = packed_half_6 + matrix_accumulator_13;
    }

    address_word_19 = 0x1u;

    matrix_accumulator_17 = 0x80u;

    address_word_18 = permute_signed(0u, address_word_18, 0x7610u);

    address_word_21 = !condition_1 ? address_word_16 : 0u;

    if (!condition_2) {
        matrix_accumulator_20 = (matrix_accumulator_20 << 0xcu) + 0x280u;
    }

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    address_word_16 = 0x80u;

    if (!condition_2) {
        matrix_accumulator_17 = address_word_21 + 0x80u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);

        address_word_19 = packed_half_129 + 0x80u;

        address_word_16 = packed_half_129 + 0x80u;

        condition_5 = uint32_t(matrix_accumulator_20) != uint32_t(matrix_accumulator_17);

        condition_0 = uint32_t(address_word_19) != uint32_t(address_word_16);

        address_word_19 = !condition_5 ? address_word_18 : 0u;

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_19 = !condition_0 ? address_word_19 : 0u;

        address_word_18 = permute_signed(address_word_19, address_word_18, 0x7610u);
    }

    address_word_18 = address_word_18 & 0xffffu;

    condition_0 = uint32_t(address_word_18) != 0u;

    split_address_words(address_word_18, address_word_19, 0ull);

    if (!condition_2) {
        split_address_words(address_word_18, address_word_19,
                            combine_words(address_word_10, packed_half_11) + 0x4000u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (!condition_2) {
            address_word_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];
        }

        split_address_words(address_word_18, address_word_19, 0ull);

        matrix_accumulator_23 = packed_half_124 * 0x400u + packed_half_134;

        if (!condition_2) {
            split_address_words(address_word_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_16, matrix_accumulator_17));
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_20 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_20 + 0x4000u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4000u;

                uniform_address_word_5 = uniform_address_word_5 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_23 = matrix_accumulator_12 * 0x10u + matrix_accumulator_23;

            address_word_16 = uniform_address_word_4;

            matrix_accumulator_17 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, matrix_accumulator_17, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_23 + 0x4200u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                address_word_16 = parameters[parameter_input_low_0];
                matrix_accumulator_17 = parameters[parameter_input_high_1];
            }

            address_word_18 = packed_half_6 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(matrix_accumulator_20, address_word_21, 0ull);

            address_word_18 = (address_word_18 << 0xcu) + 0x280u;

            address_word_19 = !condition_1 ? address_word_18 : 0u;

            if (!condition_2) {
                split_address_words(matrix_accumulator_20, address_word_21,
                                    int64_t(int32_t(address_word_19)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_16, matrix_accumulator_17));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_16 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(matrix_accumulator_20, true, uniform_address_word_6);

                address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_21, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_16, true, uniform_address_word_5);

                matrix_accumulator_17 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4200u;

                uniform_address_word_5 = uniform_address_word_5 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_16 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            address_word_16 = packed_half_2 + 0x6000u;

            uniform_address_word_4 =
                broadcast_warp_word(address_word_18, true, uniform_address_word_4);

            uniform_address_word_12 = 0x40u;

            uniform_address_word_5 =
                broadcast_warp_word(address_word_19, true, uniform_address_word_5);

            address_word_16 = (packed_half_9 << 0x18u) + address_word_16;

            uniform_address_word_10 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_10);

            uniform_address_word_7 =
                broadcast_warp_word(address_word_16, true, uniform_address_word_7);

            uniform_address_word_11 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_11);

            matrix_accumulator_17 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_7 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, address_word_16 + 0x10u, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_14 = matrix_accumulator_14 + 0x4200u;

    if (!condition_4) {
        address_word_19 = packed_half_8 + matrix_accumulator_13;
    }

    address_word_18 = 0x1u;

    address_word_16 = 0x80u;

    matrix_accumulator_17 = 0x80u;

    address_word_21 = !condition_3 ? matrix_accumulator_14 : 0u;

    matrix_accumulator_14 = permute_signed(0u, matrix_accumulator_14, 0x7610u);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23, 0ull);

    if (!condition_4) {
        matrix_accumulator_17 = packed_half_129 + 0x480u;

        address_word_19 = (address_word_19 << 0xcu) + 0x280u;

        address_word_16 = address_word_21 + 0x80u;

        matrix_accumulator_14 = permute_signed(address_word_18, matrix_accumulator_14, 0x7610u);

        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            combine_words(address_word_10, packed_half_11) + 0x5000u);

        address_word_18 = packed_half_129 + 0x480u;

        condition_1 = uint32_t(address_word_19) != uint32_t(address_word_16);

        condition_0 = uint32_t(address_word_18) != uint32_t(matrix_accumulator_17);

        address_word_18 = !condition_1 ? matrix_accumulator_14 : 0u;

        address_word_16 = parameters[parameter_input_low_0];
        matrix_accumulator_17 = parameters[parameter_input_high_1];

        address_word_18 = !condition_0 ? address_word_18 : 0u;

        matrix_accumulator_14 = permute_signed(address_word_18, matrix_accumulator_14, 0x7610u);
    }

    split_address_words(address_word_18, address_word_19, 0ull);

    matrix_accumulator_14 = matrix_accumulator_14 & 0xffffu;

    condition_0 = uint32_t(matrix_accumulator_14) != 0u;

    if (!condition_4) {
        split_address_words(address_word_18, address_word_19,
                            int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_16, matrix_accumulator_17));
    }
    if (!condition_0) {
        if (!condition_4) {
            address_word_10 = parameters[parameter_input_low_0];
            packed_half_11 = parameters[parameter_input_high_1];
        }

        split_address_words(address_word_18, address_word_19, 0ull);

        matrix_accumulator_17 = packed_half_127 * 0x4u + packed_half_134;

        if (!condition_4) {
            split_address_words(address_word_18, address_word_19,
                                int64_t(int32_t(address_word_21)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_10, packed_half_11));
        }
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            address_word_10 = matrix_accumulator_12 * 0x10u + matrix_accumulator_17;

            matrix_accumulator_20 = uniform_address_word_4;

            packed_half_11 = uniform_address_word_4;

            matrix_accumulator_20 = permute_signed(matrix_accumulator_20, packed_half_11, 0x5410u);

            address_word_21 = matrix_accumulator_20;

            matrix_accumulator_22 = matrix_accumulator_20;

            matrix_accumulator_23 = matrix_accumulator_20;

            store_four_shared_words<CheckBounds>(shared_tile, address_word_10 + 0x4000u,
                                                 matrix_accumulator_20, address_word_21,
                                                 matrix_accumulator_22, matrix_accumulator_23,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_10 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_17, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_6);

                address_word_10 = (packed_half_9 << 0x18u) + address_word_10;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_10, true, uniform_address_word_5);

                packed_half_11 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4000u;

                uniform_address_word_5 = uniform_address_word_5 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_10 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_4) {
            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 = matrix_accumulator_12 * 0x10u + matrix_accumulator_17;

            address_word_16 = uniform_address_word_4;

            packed_half_11 = uniform_address_word_4;

            address_word_16 = permute_signed(address_word_16, packed_half_11, 0x5410u);

            matrix_accumulator_17 = address_word_16;

            address_word_18 = address_word_16;

            address_word_19 = address_word_16;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_12 + 0x4200u,
                                                 address_word_16, matrix_accumulator_17,
                                                 address_word_18, address_word_19, shared_bytes,
                                                 memory_ranges, error_counts);
        } else {
            if (!condition_4) {
                address_word_10 = parameters[parameter_input_low_0];
                packed_half_11 = parameters[parameter_input_high_1];
            }

            matrix_accumulator_13 = packed_half_8 + matrix_accumulator_13;

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_18, address_word_19, 0ull);

            matrix_accumulator_13 = (matrix_accumulator_13 << 0xcu) + 0x280u;

            matrix_accumulator_13 = !condition_3 ? matrix_accumulator_13 : 0u;

            if (!condition_4) {
                split_address_words(address_word_18, address_word_19,
                                    int64_t(int32_t(matrix_accumulator_13)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_10, packed_half_11));
            }

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                address_word_10 = packed_half_2 + 0x6000u;

                uniform_address_word_4 =
                    broadcast_warp_word(matrix_accumulator_17, true, uniform_address_word_4);

                uniform_address_word_10 = 0x20u;

                uniform_address_word_6 =
                    broadcast_warp_word(address_word_18, true, uniform_address_word_6);

                address_word_10 = (packed_half_9 << 0x18u) + address_word_10;

                uniform_address_word_7 =
                    broadcast_warp_word(address_word_19, true, uniform_address_word_7);

                uniform_address_word_5 =
                    broadcast_warp_word(address_word_10, true, uniform_address_word_5);

                packed_half_11 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x4200u;

                uniform_address_word_5 = uniform_address_word_5 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, address_word_10 + 0x10u, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            address_word_10 = packed_half_2 + 0x6000u;

            uniform_address_word_4 =
                broadcast_warp_word(matrix_accumulator_22, true, uniform_address_word_4);

            uniform_address_word_12 = 0x40u;

            uniform_address_word_5 =
                broadcast_warp_word(matrix_accumulator_23, true, uniform_address_word_5);

            address_word_10 = (packed_half_9 << 0x18u) + address_word_10;

            uniform_address_word_10 =
                broadcast_warp_word(address_word_18, true, uniform_address_word_10);

            uniform_address_word_7 =
                broadcast_warp_word(address_word_10, true, uniform_address_word_7);

            uniform_address_word_11 =
                broadcast_warp_word(address_word_19, true, uniform_address_word_11);

            packed_half_11 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

            uniform_address_word_7 = uniform_address_word_7 + 0x10u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_10, uniform_address_word_11),
                uniform_address_word_12 * 16u, address_word_10 + 0x10u, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_10, uniform_address_word_11),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_10 = packed_half_2 + 0x6000u;

    packed_half_128 = packed_half_128 + 0xc0u;

    packed_half_130 = packed_half_124 * 0x2u;

    packed_half_9 = (packed_half_9 << 0x18u) + address_word_10;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = packed_half_9;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(address_word_10, packed_half_11, async_tokens[operand_a >> 3 & 1]);
    }
#else
    address_word_10 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_0 =
        wait_copy_barrier(shared_tile, packed_half_9, async_tokens[(((packed_half_9) >> 3) & 1)]);
#else
    __syncthreads();
    condition_0 = true;
#endif
    if (!condition_0) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_0 = wait_copy_barrier(shared_tile, packed_half_9,
                                            async_tokens[(((packed_half_9) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
        } while (!condition_0);
    }
    packed_half_126 = packed_half_124 * 0x800u;

    packed_half_9 = packed_half_129 + 0x80u;

    packed_half_125 = packed_half_124 * 0x200u;

    packed_half_11 = 0u;

    matrix_accumulator_14 = 0u;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    address_word_16 = 0;
    matrix_accumulator_17 = 0;

    address_word_18 = 0;
    address_word_19 = 0;

    matrix_accumulator_20 = 0;
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

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    packed_half_130 = packed_half_130 & 0x7fcu;

    packed_half_126 = packed_half_126 & 0x1ff000u;

    packed_half_125 = packed_half_125 & 0x7fc00u;
    do {
        packed_half_137 = uint32_t(uint32_t(packed_half_11) >> 0x6u);

        address_word_10 = permute_signed(packed_half_137, 0u, 0x9910u);

        __syncwarp(__activemask());

        address_word_10 = address_word_10 * 0xabu;

        condition_0 = uint32_t(packed_half_11) > uint32_t(0x3bfu);

        condition_1 = uint32_t(packed_half_11) > uint32_t(0x33fu);

        address_word_10 = address_word_10 & 0xffffu;

        address_word_10 = uint32_t(uint32_t(address_word_10) >> 0x9u);

        address_word_10 = permute_signed(address_word_10, 0u, 0x9910u);

        address_word_10 = address_word_10 * 0x3u;

        matrix_fragment_80 = 0u - address_word_10;

        address_word_10 = threadIdx.x & 31u;

        matrix_fragment_80 = permute_signed(matrix_fragment_80, 0u, 0x7710u);

        packed_half_137 = packed_half_137 + matrix_fragment_80;

        packed_half_136 = packed_half_137 * 0x2000u;

        packed_half_136 = 0x1fe000u & packed_half_136;

        packed_half_135 = packed_half_134 + packed_half_136;

        matrix_fragment_81 = packed_half_126 + packed_half_135;

        matrix_fragment_85 = packed_half_125 * 0x4u + packed_half_135;

        matrix_fragment_81 = address_word_10 * 0x10u + matrix_fragment_81;

        tile_word_138 = address_word_10 * 0x10u + matrix_fragment_85;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_81, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_138 + 0x400u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_138 + 0x600u, matrix_fragment_84, matrix_fragment_85,
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
            shared_tile, tile_word_138 + 0x200u, matrix_fragment_88, matrix_fragment_89,
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
            shared_tile, tile_word_138 + 0x800u, matrix_fragment_80, matrix_fragment_81,
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
            shared_tile, tile_word_138 + 0xc00u, matrix_fragment_92, matrix_fragment_93,
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
            shared_tile, tile_word_138 + 0xe00u, matrix_fragment_84, matrix_fragment_85,
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

        multiply_accumulate_half(matrix_accumulator_20, address_word_21, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_20,
                                 address_word_21);

        multiply_accumulate_half(address_word_18, address_word_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_58, matrix_fragment_59, address_word_18,
                                 address_word_19);

        multiply_accumulate_half(address_word_16, matrix_accumulator_17, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_60, matrix_fragment_61, address_word_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_138 + 0xa00u, matrix_fragment_80, matrix_fragment_81,
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

        multiply_accumulate_half(matrix_accumulator_20, address_word_21, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_20,
                                 address_word_21);

        multiply_accumulate_half(address_word_18, address_word_19, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_74, matrix_fragment_75, address_word_18,
                                 address_word_19);

        multiply_accumulate_half(address_word_16, matrix_accumulator_17, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_76, matrix_fragment_77, address_word_16,
                                 matrix_accumulator_17);

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
        if (!condition_0) {
            matrix_fragment_48 = parameters[parameter_trained_weights_low_6];
            matrix_fragment_49 = parameters[parameter_trained_weights_high_7];

            matrix_fragment_82 = packed_half_11 + 0x40u;

            matrix_fragment_51 = packed_half_132 + packed_half_11;

            matrix_fragment_82 = uint32_t(uint32_t(matrix_fragment_82) >> 0x6u);

            matrix_fragment_51 = matrix_fragment_51 * 0x400u + packed_half_4;

            matrix_fragment_83 = permute_signed(matrix_fragment_82, 0u, 0x9910u);

            matrix_fragment_83 = matrix_fragment_83 * 0xabu;

            matrix_fragment_85 = 0u;

            matrix_fragment_83 = matrix_fragment_83 & 0xffffu;

            split_address_words(matrix_fragment_48, matrix_fragment_49,
                                int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_48, matrix_fragment_49));

            matrix_fragment_83 = uint32_t(uint32_t(matrix_fragment_83) >> 0x9u);

            matrix_fragment_83 = permute_signed(matrix_fragment_83, 0u, 0x9910u);

            split_address_words(matrix_fragment_80, matrix_fragment_81,
                                int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_48, matrix_fragment_49));

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

            matrix_fragment_80 = matrix_fragment_83 * 0x3u;

            matrix_fragment_80 = 0u - matrix_fragment_80;

            matrix_fragment_81 = permute_signed(matrix_fragment_80, 0u, 0x7710u);

            matrix_fragment_80 = packed_half_2 + 0x6000u;

            matrix_fragment_82 = matrix_fragment_82 + matrix_fragment_81;

            matrix_fragment_83 = (matrix_fragment_85 << 0x18u) + matrix_fragment_80;

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
            condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_83,
                                            async_tokens[(((matrix_fragment_83) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
            if (!condition_0) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_0 =
                        wait_copy_barrier(shared_tile, matrix_fragment_83,
                                          async_tokens[(((matrix_fragment_83) >> 3) & 1)]);
#else
                    __syncthreads();
                    condition_0 = true;
#endif
                } while (!condition_0);
            }
        }
        matrix_fragment_87 = packed_half_137 & 0xffu;

        packed_half_137 = 0u;
        if (!condition_1) {
            matrix_fragment_86 = packed_half_128 + packed_half_11;

            condition_1 = int32_t(packed_half_6) >= int32_t(index_word_133);

            condition_2 = int32_t(packed_half_6) >= int32_t(index_word_133);

            matrix_fragment_95 = 0u;

            packed_half_2 = 0x80u;

            matrix_fragment_86 = matrix_fragment_86 * 0x4u + packed_half_7;

            matrix_fragment_83 = 0x1u;

            matrix_fragment_82 = permute_signed(0u, matrix_fragment_82, 0x7610u);

            matrix_fragment_92 = 0u;

            matrix_fragment_80 = 0x80u;

            matrix_fragment_93 = !condition_2 ? matrix_fragment_86 : 0u;

            uniform_address_word_4 = 0x400u;

            if (!condition_1) {
                matrix_fragment_81 = matrix_fragment_86 + 0x80u;

                packed_half_2 = matrix_fragment_93 + 0x80u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_83, matrix_fragment_82, 0x7610u);

                matrix_fragment_80 = packed_half_9;

                condition_3 = uint32_t(matrix_fragment_81) != uint32_t(packed_half_2);

                condition_0 = uint32_t(packed_half_9) != uint32_t(matrix_fragment_80);

                packed_half_2 = !condition_3 ? matrix_fragment_82 : 0u;

                matrix_fragment_80 = parameters[parameter_input_low_0];
                matrix_fragment_81 = parameters[parameter_input_high_1];

                matrix_fragment_83 = !condition_0 ? packed_half_2 : 0u;
            }

            packed_half_2 = uniform_address_word_4;

            if (!condition_1) {
                matrix_fragment_82 =
                    permute_signed(matrix_fragment_83, matrix_fragment_82, 0x7610u);
            }

            matrix_fragment_84 = packed_half_134;

            matrix_fragment_85 = matrix_fragment_95;

            matrix_fragment_83 = packed_half_2 + 0x6000u;

            matrix_fragment_82 = matrix_fragment_82 & 0xffffu;

            split_address_words(matrix_fragment_94, matrix_fragment_95, 0ull);

            split_address_words(matrix_fragment_84, matrix_fragment_85,
                                combine_words(matrix_fragment_84, matrix_fragment_85) +
                                    combine_words(packed_half_136, packed_half_137));

            condition_0 = uint32_t(matrix_fragment_82) != 0u;

            matrix_fragment_92 = (matrix_fragment_92 << 0x18u) + matrix_fragment_83;

            split_address_words(matrix_fragment_82, matrix_fragment_83, 0ull);

            split_address_words(matrix_fragment_84, matrix_fragment_85,
                                uint64_t(uint32_t(packed_half_129)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_84, matrix_fragment_85));

            matrix_fragment_87 = matrix_fragment_87 * 0x8u + matrix_fragment_92;

            if (!condition_1) {
                split_address_words(matrix_fragment_94, matrix_fragment_95,
                                    combine_words(matrix_fragment_84, matrix_fragment_85));

                split_address_words(matrix_fragment_82, matrix_fragment_83,
                                    int64_t(int32_t(matrix_fragment_93)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_80, matrix_fragment_81));
            }
            if (!condition_0) {
                if (!condition_1) {
                    matrix_fragment_80 = parameters[parameter_input_low_0];
                    matrix_fragment_81 = parameters[parameter_input_high_1];
                }

                split_address_words(matrix_fragment_82, matrix_fragment_83, 0ull);

                matrix_fragment_95 = packed_half_124 * 0x400u + packed_half_135;

                if (!condition_1) {
                    split_address_words(matrix_fragment_82, matrix_fragment_83,
                                        int64_t(int32_t(matrix_fragment_93)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_80, matrix_fragment_81));
                }
                if (condition_1) {
                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_92 = address_word_10 * 0x10u + matrix_fragment_95;

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_81 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_81, 0x5410u);

                    matrix_fragment_81 = matrix_fragment_80;

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_92,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    uniform_address_word_4 = 0xffffffffu;

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        uniform_address_word_6 =
                            broadcast_warp_word(matrix_fragment_82, true, uniform_address_word_6);

                        uniform_address_word_10 = 0x20u;

                        uniform_address_word_7 =
                            broadcast_warp_word(matrix_fragment_83, true, uniform_address_word_7);

                        uniform_address_word_4 =
                            broadcast_warp_word(matrix_fragment_95, true, uniform_address_word_4);

                        uniform_address_word_5 =
                            broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_5);

                        matrix_fragment_80 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
                if (condition_1) {
                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_95 = address_word_10 * 0x10u + matrix_fragment_95;

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_81 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_81, 0x5410u);

                    matrix_fragment_81 = matrix_fragment_80;

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_95 + 0x200u,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    if (!condition_1) {
                        matrix_fragment_80 = parameters[parameter_input_low_0];
                        matrix_fragment_81 = parameters[parameter_input_high_1];
                    }

                    matrix_fragment_83 = matrix_fragment_86 + 0x80u;

                    uniform_address_word_4 = 0xffffffffu;

                    split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                    matrix_fragment_83 = !condition_2 ? matrix_fragment_83 : 0u;

                    if (!condition_1) {
                        split_address_words(
                            matrix_fragment_92, matrix_fragment_93,
                            int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));
                    }

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        uniform_address_word_4 =
                            broadcast_warp_word(matrix_fragment_95, true, uniform_address_word_4);

                        uniform_address_word_10 = 0x20u;

                        uniform_address_word_6 =
                            broadcast_warp_word(matrix_fragment_92, true, uniform_address_word_6);

                        uniform_address_word_7 =
                            broadcast_warp_word(matrix_fragment_93, true, uniform_address_word_7);

                        uniform_address_word_5 =
                            broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_5);

                        matrix_fragment_80 = 0x200u;

                        uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    uniform_address_word_4 =
                        broadcast_warp_word(matrix_fragment_94, true, uniform_address_word_4);

                    uniform_address_word_12 = 0x40u;

                    uniform_address_word_5 =
                        broadcast_warp_word(matrix_fragment_95, true, uniform_address_word_5);

                    uniform_address_word_10 =
                        broadcast_warp_word(matrix_fragment_82, true, uniform_address_word_10);

                    uniform_address_word_11 =
                        broadcast_warp_word(matrix_fragment_83, true, uniform_address_word_11);

                    uniform_address_word_7 =
                        broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_7);

                    matrix_fragment_80 = 0x400u;

                    uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_12 * 16u, matrix_fragment_87, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_12 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            condition_1 = int32_t(packed_half_8) >= int32_t(index_word_133);

            condition_2 = int32_t(packed_half_8) >= int32_t(index_word_133);

            matrix_fragment_80 = matrix_fragment_86 + 0x4000u;

            matrix_fragment_81 = 0x80u;

            matrix_fragment_92 = 0x1u;

            matrix_fragment_82 = permute_signed(0u, matrix_fragment_82, 0x7610u);

            split_address_words(matrix_fragment_94, matrix_fragment_95, 0ull);

            matrix_fragment_83 = !condition_2 ? matrix_fragment_80 : 0u;

            if (!condition_1) {
                matrix_fragment_80 = matrix_fragment_86 + 0x4080u;

                matrix_fragment_81 = matrix_fragment_83 + 0x80u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_92, matrix_fragment_82, 0x7610u);

                split_address_words(matrix_fragment_94, matrix_fragment_95,
                                    combine_words(matrix_fragment_84, matrix_fragment_85) +
                                        0x1000u);

                condition_0 = uint32_t(matrix_fragment_80) != uint32_t(matrix_fragment_81);

                matrix_fragment_80 = parameters[parameter_input_low_0];
                matrix_fragment_81 = parameters[parameter_input_high_1];

                matrix_fragment_92 = !condition_0 ? matrix_fragment_82 : 0u;

                matrix_fragment_82 =
                    permute_signed(matrix_fragment_92, matrix_fragment_82, 0x7610u);
            }

            split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

            matrix_fragment_82 = matrix_fragment_82 & 0xffffu;

            condition_0 = uint32_t(matrix_fragment_82) != 0u;

            if (!condition_1) {
                split_address_words(matrix_fragment_92, matrix_fragment_93,
                                    int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_80, matrix_fragment_81));
            }
            if (!condition_0) {
                if (!condition_1) {
                    matrix_fragment_80 = parameters[parameter_input_low_0];
                    matrix_fragment_81 = parameters[parameter_input_high_1];
                }

                split_address_words(matrix_fragment_84, matrix_fragment_85, 0ull);

                packed_half_135 = packed_half_127 * 0x4u + packed_half_135;

                if (!condition_1) {
                    split_address_words(matrix_fragment_84, matrix_fragment_85,
                                        int64_t(int32_t(matrix_fragment_83)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_80, matrix_fragment_81));
                }
                if (condition_1) {
                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    matrix_fragment_84 = address_word_10 * 0x10u + packed_half_135;

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_81 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_81, 0x5410u);

                    matrix_fragment_81 = matrix_fragment_80;

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_84,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    uniform_address_word_4 = 0xffffffffu;

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        uniform_address_word_6 =
                            broadcast_warp_word(matrix_fragment_84, true, uniform_address_word_6);

                        uniform_address_word_10 = 0x20u;

                        uniform_address_word_7 =
                            broadcast_warp_word(matrix_fragment_85, true, uniform_address_word_7);

                        uniform_address_word_4 =
                            broadcast_warp_word(packed_half_135, true, uniform_address_word_4);

                        uniform_address_word_5 =
                            broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_5);

                        matrix_fragment_80 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
                if (condition_1) {
                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    packed_half_135 = address_word_10 * 0x10u + packed_half_135;

                    matrix_fragment_80 = uniform_address_word_4;

                    matrix_fragment_81 = uniform_address_word_4;

                    matrix_fragment_80 =
                        permute_signed(matrix_fragment_80, matrix_fragment_81, 0x5410u);

                    matrix_fragment_81 = matrix_fragment_80;

                    matrix_fragment_82 = matrix_fragment_80;

                    matrix_fragment_83 = matrix_fragment_80;

                    store_four_shared_words<CheckBounds>(shared_tile, packed_half_135 + 0x200u,
                                                         matrix_fragment_80, matrix_fragment_81,
                                                         matrix_fragment_82, matrix_fragment_83,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    if (!condition_1) {
                        matrix_fragment_80 = parameters[parameter_input_low_0];
                        matrix_fragment_81 = parameters[parameter_input_high_1];
                    }

                    matrix_fragment_83 = matrix_fragment_86 + 0x4080u;

                    uniform_address_word_4 = 0xffffffffu;

                    split_address_words(matrix_fragment_84, matrix_fragment_85, 0ull);

                    matrix_fragment_83 = !condition_2 ? matrix_fragment_83 : 0u;

                    if (!condition_1) {
                        split_address_words(
                            matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(matrix_fragment_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));
                    }

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        uniform_address_word_4 =
                            broadcast_warp_word(packed_half_135, true, uniform_address_word_4);

                        uniform_address_word_10 = 0x20u;

                        uniform_address_word_6 =
                            broadcast_warp_word(matrix_fragment_84, true, uniform_address_word_6);

                        uniform_address_word_7 =
                            broadcast_warp_word(matrix_fragment_85, true, uniform_address_word_7);

                        uniform_address_word_5 =
                            broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_5);

                        matrix_fragment_80 = 0x200u;

                        uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, matrix_fragment_87, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_6, uniform_address_word_7),
                            uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    uniform_address_word_4 =
                        broadcast_warp_word(matrix_fragment_94, true, uniform_address_word_4);

                    uniform_address_word_12 = 0x40u;

                    uniform_address_word_5 =
                        broadcast_warp_word(matrix_fragment_95, true, uniform_address_word_5);

                    uniform_address_word_10 =
                        broadcast_warp_word(matrix_fragment_92, true, uniform_address_word_10);

                    uniform_address_word_11 =
                        broadcast_warp_word(matrix_fragment_93, true, uniform_address_word_11);

                    uniform_address_word_7 =
                        broadcast_warp_word(matrix_fragment_87, true, uniform_address_word_7);

                    matrix_fragment_80 = 0x400u;

                    uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_12 * 16u, matrix_fragment_87, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_12 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_0 = uint32_t(packed_half_11) >= uint32_t(0x3c0u);

        packed_half_11 = packed_half_11 + 0x40u;
    } while (!condition_0);
    packed_half_2 = threadIdx.z;

    packed_half_130 = address_word_5 * 0x8u + packed_half_130;

    packed_half_7 = 0x2b28u;

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                       0x44004400u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                                       0x44004400u);

    address_word_3 = packed_half_130 * 0x1000u + address_word_3;

    packed_half_4 = packed_half_130 + 0x1u;

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    packed_half_6 = packed_half_130 + 0x2u;

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    address_word_3 = address_word_3 * 0x4u;

    condition_2 = int32_t(packed_half_4) >= int32_t(index_word_133);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       0x44004400u);

    condition_3 = int32_t(packed_half_6) >= int32_t(index_word_133);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       0x44004400u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    condition_1 = int32_t(packed_half_130) >= int32_t(index_word_133);

    matrix_fragment_48 = packed_half_130 + 0x3u;

    matrix_fragment_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x44004400u);

    matrix_fragment_51 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x44004400u);

    condition_4 = int32_t(matrix_fragment_48) >= int32_t(index_word_133);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    condition_0 = (packed_half_131 | packed_half_2) != 0;

    packed_half_2 = permute_signed(packed_half_7, packed_half_2, 0x7610u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                       0x44004400u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x44004400u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                                       0x44004400u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       0x44004400u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_74), 0x3b283b28u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                                       0x44004400u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                                       0x44004400u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_77), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                                       0x44004400u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       0x44004400u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_80), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_76), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       0x44004400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       0x44004400u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_81), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_15), 0x44004400u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                       0x44004400u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_78), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       0x44004400u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_82), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       0x44004400u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                       0x44004400u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_86), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       0x44004400u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       0x44004400u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       0x44004400u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       0x44004400u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_87), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                       0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_12), 0x44004400u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       0x44004400u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_accumulator_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_94), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       0x44004400u);

    matrix_accumulator_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_accumulator_120), 0x3b283b28u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(packed_half_124), 0x3b283b28u);

    packed_half_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3b283b28u);

    packed_half_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_127 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       0x44004400u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       0x44004400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    packed_half_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       0x44004400u);

    packed_half_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(packed_half_125), 0x3b283b28u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    packed_half_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_126), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       0x44004400u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       0x44004400u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(packed_half_129), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(packed_half_128), 0x3b283b28u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       0x44004400u);

    packed_half_132 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                                       0x44004400u);

    packed_half_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    packed_half_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    packed_half_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(packed_half_130), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       0x44004400u);

    packed_half_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_131), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                                       0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                                       0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       0x44004400u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_4 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       0x44004400u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       0x44004400u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

    packed_half_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_21), 0x44004400u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_9),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_4), 0xc400c400u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_61),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_18), 0x44004400u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_9),
        select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_4),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_accumulator_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    packed_half_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_19), 0x44004400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_9), 0xc400c400u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_4),
        select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

    index_word_133 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                                        0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_8),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x44004400u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

    packed_half_134 =
        maximum_half_pairs(select_half_operand<2, false, false>(index_word_133), 0xc400c400u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

    packed_half_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(address_word_16), 0x44004400u);

    packed_half_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_11),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_136 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_135), 0xc400c400u);

    packed_half_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_8),
        select_half_operand<2, false, false>(packed_half_4), 0x3b283b28u);

    packed_half_8 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                       0x44004400u);

    packed_half_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_11),
        select_half_operand<2, false, false>(packed_half_6), 0x3b283b28u);

    packed_half_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_7), 0xc400c400u);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_65));

    packed_half_6 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       0x44004400u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_21),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    packed_half_11 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

    matrix_accumulator_20 = parameters[parameter_output_low_4];
    address_word_21 = parameters[parameter_output_high_5];

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_8), 0xc400c400u);

    packed_half_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_7),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0xc400c400u);

    packed_half_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_134),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_11), 0xc400c400u);

    packed_half_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_7),
        select_half_operand<2, false, false>(packed_half_8), 0x3b283b28u);

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_6),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<2, false, false>(packed_half_11), 0x3b283b28u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(packed_half_7), 0x3b283b28u);

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_53),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107), 0x44004400u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<2, false, false>(packed_half_11), 0x3b283b28u);

    packed_half_11 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x44004400u);

    packed_half_137 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_136),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_accumulator_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_105), 0x44004400u);

    matrix_accumulator_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_11), 0xc400c400u);

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_accumulator_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    split_address_words(matrix_accumulator_20, address_word_21,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_20, address_word_21));

    packed_half_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(packed_half_11), 0x3b283b28u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_fragment_50 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_14), 0x44004400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_60),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_13), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_42), 0x44004400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_43), 0x44004400u);

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_61),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    index_word_133 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(index_word_133), 0x3b283b28u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_134),
        select_half_operand<2, false, false>(packed_half_135), 0x3b283b28u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_136),
        select_half_operand<2, false, false>(packed_half_137), 0x3b283b28u);

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    packed_half_124 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                          select_half_operand<2, false, false>(packed_half_124));

    packed_half_125 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                          select_half_operand<2, false, false>(packed_half_125));

    packed_half_126 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                          select_half_operand<2, false, false>(packed_half_126));

    packed_half_127 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                          select_half_operand<2, false, false>(packed_half_127));

    packed_half_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(packed_half_128));

    packed_half_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(packed_half_129));

    packed_half_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(packed_half_130));

    packed_half_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(packed_half_131));

    packed_half_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(packed_half_132));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_64));

    packed_half_9 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_18),
                                        select_half_operand<2, false, false>(packed_half_9));

    packed_half_4 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_19),
                                        select_half_operand<2, false, false>(packed_half_4));

    packed_half_8 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_16),
                                        select_half_operand<2, false, false>(packed_half_8));

    packed_half_6 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
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

    packed_half_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(packed_half_11));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_14),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_fragment_62));
    if (!condition_1) {
        matrix_accumulator_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                          select_half_operand<2, false, false>(0u));

        address_word_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                          select_half_operand<2, false, false>(0u));

        packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                                      select_half_operand<2, false, false>(0u));

        address_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                          select_half_operand<2, false, false>(matrix_accumulator_12));

        address_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                        select_half_operand<2, false, false>(address_word_16));

        matrix_accumulator_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                          select_half_operand<2, false, false>(matrix_accumulator_14));

        matrix_accumulator_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                          select_half_operand<2, false, false>(packed_half_2));

        matrix_accumulator_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                          select_half_operand<2, false, false>(address_word_3));

        split_address_words(packed_half_2, address_word_3,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, address_word_21));

        matrix_accumulator_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                          select_half_operand<2, false, false>(matrix_accumulator_23));

        address_word_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                          select_half_operand<2, false, false>(matrix_accumulator_25));

        address_word_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                          select_half_operand<2, false, false>(matrix_accumulator_27));

        store_four_words_streaming<CheckBounds>(combine_words(packed_half_2, address_word_3),
                                                matrix_accumulator_12, matrix_accumulator_13,
                                                matrix_accumulator_14, matrix_accumulator_15,
                                                memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, address_word_3) + 0x200ull, address_word_16,
            matrix_accumulator_17, address_word_18, address_word_19, memory_ranges, error_counts);
    }
    split_address_words(packed_half_2, address_word_3,
                        combine_words(matrix_accumulator_20, address_word_21) + 0x10000u);

    matrix_accumulator_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(0u));

    address_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    address_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                    select_half_operand<2, false, false>(0u));

    address_word_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_125),
                                          select_half_operand<2, false, false>(0u));

    packed_half_126 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_126),
                                    select_half_operand<2, false, false>(0u));

    packed_half_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_127),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(0u));

    packed_half_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                    select_half_operand<2, false, false>(0u));

    packed_half_131 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                    select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        matrix_accumulator_13 = permute_signed(matrix_accumulator_13, address_word_16, 0x5410u);

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_17, 0x5410u);

        split_address_words(address_word_16, matrix_accumulator_17,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, address_word_3));

        matrix_accumulator_15 = permute_signed(matrix_accumulator_15, address_word_18, 0x5410u);

        matrix_accumulator_14 = permute_signed(matrix_accumulator_14, address_word_19, 0x5410u);

        split_address_words(address_word_18, address_word_19,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, address_word_21));

        matrix_accumulator_22 = permute_signed(matrix_accumulator_22, packed_half_131, 0x5410u);

        matrix_accumulator_23 = permute_signed(matrix_accumulator_23, packed_half_130, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_16, matrix_accumulator_17), matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);

        matrix_accumulator_24 = permute_signed(matrix_accumulator_24, packed_half_127, 0x5410u);

        matrix_accumulator_25 = permute_signed(matrix_accumulator_25, packed_half_126, 0x5410u);

        matrix_accumulator_12 = matrix_accumulator_25;

        matrix_accumulator_13 = matrix_accumulator_24;

        matrix_accumulator_14 = matrix_accumulator_23;

        matrix_accumulator_15 = matrix_accumulator_22;

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_18, address_word_19) + 0x10200ull, matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }
    split_address_words(matrix_accumulator_20, address_word_21,
                        combine_words(matrix_accumulator_20, address_word_21) + 0x20000u);

    address_word_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_9),
                                    select_half_operand<2, false, false>(0u));

    address_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_4),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_132),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                          select_half_operand<2, false, false>(0u));

    address_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(0u));

    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_8),
                                  select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_6),
                                  select_half_operand<2, false, false>(0u));

    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                  select_half_operand<2, false, false>(0u));

    packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_7),
                                  select_half_operand<2, false, false>(0u));
    if (!condition_3) {
        packed_half_4 = permute_signed(packed_half_4, packed_half_7, 0x5410u);

        split_address_words(packed_half_6, packed_half_7,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, address_word_21));

        matrix_accumulator_15 = permute_signed(matrix_accumulator_15, matrix_fragment_64, 0x5410u);

        matrix_accumulator_14 = permute_signed(matrix_accumulator_14, matrix_fragment_63, 0x5410u);

        split_address_words(packed_half_2, address_word_3,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, address_word_3));

        matrix_accumulator_13 = permute_signed(matrix_accumulator_13, address_word_16, 0x5410u);

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_17, 0x5410u);

        packed_half_9 = permute_signed(packed_half_8, packed_half_9, 0x5410u);

        packed_half_8 = permute_signed(matrix_fragment_66, address_word_19, 0x5410u);

        store_four_words_streaming<CheckBounds>(combine_words(packed_half_6, packed_half_7),
                                                matrix_accumulator_12, matrix_accumulator_13,
                                                matrix_accumulator_14, matrix_accumulator_15,
                                                memory_ranges, error_counts);

        matrix_accumulator_17 = permute_signed(matrix_fragment_65, address_word_18, 0x5410u);

        matrix_accumulator_12 = matrix_accumulator_17;

        matrix_accumulator_13 = packed_half_8;

        matrix_accumulator_14 = packed_half_9;

        matrix_accumulator_15 = packed_half_4;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, address_word_3) + 0x10200ull, matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }
    packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                                  select_half_operand<2, false, false>(0u));

    packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                  select_half_operand<2, false, false>(0u));

    address_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                          select_half_operand<2, false, false>(0u));

    packed_half_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                  select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        split_address_words(matrix_accumulator_20, address_word_21,
                            int64_t(int32_t(address_word_10)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, address_word_21));

        packed_half_11 = permute_signed(packed_half_11, packed_half_2, 0x5410u);

        address_word_10 = permute_signed(matrix_fragment_51, matrix_fragment_50, 0x5410u);

        packed_half_9 = permute_signed(packed_half_9, matrix_fragment_48, 0x5410u);

        packed_half_8 = permute_signed(packed_half_8, address_word_3, 0x5410u);

        matrix_accumulator_15 = permute_signed(matrix_accumulator_15, matrix_fragment_62, 0x5410u);

        matrix_accumulator_14 = permute_signed(matrix_accumulator_14, matrix_fragment_61, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_20, address_word_21) + 0x10000ull, packed_half_8,
            packed_half_9, address_word_10, packed_half_11, memory_ranges, error_counts);

        matrix_accumulator_13 = permute_signed(matrix_accumulator_13, packed_half_4, 0x5410u);

        matrix_accumulator_12 = permute_signed(matrix_accumulator_12, packed_half_7, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_20, address_word_21) + 0x10200ull,
            matrix_accumulator_12, matrix_accumulator_13, matrix_accumulator_14,
            matrix_accumulator_15, memory_ranges, error_counts);
    }
    __syncwarp(__activemask());

    __syncthreads();

    if (condition_0) {
        return;
    }

    packed_half_2 = parameters[parameter_publication_optional_low_14];
    address_word_3 = parameters[parameter_publication_optional_high_15];

    address_word_5 = address_word_5 * 0x20u + address_word_0;

    split_address_words(packed_half_2, address_word_3,
                        int64_t(int32_t(address_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, address_word_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, address_word_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__
__maxnreg__(256) void phoenix_global_attention_transformer_feed_forward_expand_publish(
    KernelArguments parameters) {
    run_global_attention_transformer_feed_forward_expand_publish<false>(parameters.words, nullptr,
                                                                        nullptr);
}
