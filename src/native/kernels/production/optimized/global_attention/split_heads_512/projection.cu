#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/projection.cu
// 512-channel split-head stage. Separate projection/feed-forward modules use the same persistent feature layout.
// The filename identifies which stage is executed; wait/view variants preserve producer-consumer synchronization.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 13336
#include "../../../common/async_memory.cuh"
#include "../../../common/cached_vector_access.cuh"

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
    parameter_geometry_or_control_16 = 16,
    parameter_geometry_or_control_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[18];
};
static_assert(sizeof(KernelArguments) == 72, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_split_heads_512_projection(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t block_index_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t index_word_4 = 0;
    uint32_t address_word_5 = 0;
    uint32_t address_word_6 = 0;
    uint32_t address_word_7 = 0;
    uint32_t address_word_8 = 0;
    uint32_t address_word_9 = 0;
    uint32_t packed_half_10 = 0;
    uint32_t index_word_11 = 0;
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
    uint32_t reused_word_88 = 0;
    uint32_t index_word_89 = 0;
    uint32_t index_word_90 = 0;
    uint32_t index_word_91 = 0;
    uint32_t index_word_92 = 0;
    uint32_t index_word_93 = 0;
    uint32_t index_word_94 = 0;
    uint32_t address_word_95 = 0;
    uint32_t index_word_96 = 0;
    uint32_t index_word_97 = 0;
    uint32_t index_word_98 = 0;
    uint32_t index_word_99 = 0;
    uint32_t warp_lane_100 = 0;
    uint32_t index_word_101 = 0;
    uint32_t index_word_102 = 0;
    uint32_t index_word_103 = 0;
    uint32_t tile_word_104 = 0;
    uint32_t index_word_105 = 0;
    uint32_t index_word_107 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_address_word_11 = 0;
    uint32_t uniform_reused_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
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
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    matrix_accumulator_15 = parameters[parameter_tensor_width_9];

    matrix_accumulator_30 = threadIdx.y;

    uniform_index_word_10 = parameters[parameter_geometry_or_control_17];

    index_word_11 = threadIdx.x;

    uniform_address_word_9 = parameters[parameter_tensor_height_8];

    matrix_accumulator_13 = threadIdx.x & 31u;

    uniform_reused_word_12 = 0u;
    uniform_reused_word_13 = 0u;

    matrix_accumulator_31 = blockIdx.y;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_index_word_10));

    block_index_0 = matrix_accumulator_15 + 0xffffffffu;

    uniform_index_word_4 = 0u - uniform_index_word_4 + uniform_address_word_9;

    address_word_3 = uint32_t(int32_t(block_index_0) >> 0x1fu);

    uniform_condition_0 = uint32_t(uniform_index_word_4) != 0u;

    address_word_3 = uint32_t(combine_words(address_word_3, 0u) << 0x3u >> 32) + block_index_0;

    block_index_0 = blockIdx.x;

    index_word_97 = uint32_t(int64_t(int32_t(address_word_3)) * (1ll << 0x1du) >> 32) + 0x1u;

    matrix_accumulator_17 = signed_word_magnitude(index_word_97);

    if (uniform_condition_0) {
        uniform_index_word_4 = uniform_index_word_4 + 0xffffffffu;
    }

    address_word_7 = signed_word_magnitude(index_word_97);

    index_word_4 = __float_as_uint(__int2float_ru(matrix_accumulator_17));

    matrix_fragment_65 = ~index_word_97;

    if (uniform_condition_0) {
        uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);
    }

    address_word_7 = 0u - address_word_7;

    if (uniform_condition_0) {
        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x3u >> 32) + uniform_index_word_4;
    }

    uniform_index_word_4 = 0u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_5)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    index_word_4 = reciprocal_bits(index_word_4);

    address_word_2 = index_word_4 + 0xffffffeu;

    index_word_4 = signed_word_magnitude(block_index_0);

    address_word_3 = __float2uint_rz(__uint_as_float(address_word_2));

    address_word_2 = 0u;

    address_word_6 = 0u - address_word_3;

    address_word_5 = address_word_6 * matrix_accumulator_17;

    address_word_3 = uint32_t(uint64_t(address_word_3) * address_word_5 +
                                  combine_words(address_word_2, address_word_3) >>
                              32);

    address_word_5 = address_word_7;

    address_word_2 = block_index_0 ^ index_word_97;

    address_word_6 = parameters[parameter_trained_weights_low_6];
    address_word_7 = parameters[parameter_trained_weights_high_7];

    address_word_3 = uint32_t(uint64_t(address_word_3) * index_word_4 >> 32);

    index_word_4 = address_word_3 * address_word_5 + index_word_4;

    condition_1 = uint32_t(matrix_accumulator_17) > uint32_t(index_word_4);

    if (!condition_1) {
        index_word_4 = index_word_4 + (0u - matrix_accumulator_17);

        address_word_3 = address_word_3 + 0x1u;
    }

    condition_1 = int32_t(address_word_2) >= int32_t(0u);

    address_word_2 = blockIdx.z;

    condition_0 = uint32_t(index_word_4) >= uint32_t(matrix_accumulator_17);

    if (condition_0) {
        address_word_3 = address_word_3 + 0x1u;
    }

    condition_0 = int32_t(index_word_97) != int32_t(0u);

    address_word_8 = address_word_3;

    address_word_3 = matrix_accumulator_30 * 0x40u;

    if (!condition_1) {
        address_word_8 = 0u - address_word_8;
    }

    address_word_3 = address_word_3 & 0xc0u;

    index_word_92 = index_word_11 | matrix_accumulator_30;
    condition_1 = index_word_92 != 0;

    address_word_8 = !condition_0 ? matrix_fragment_65 : address_word_8;

    address_word_3 = address_word_8 * 0x100u + address_word_3;

    address_word_5 = address_word_3 * 0x8u;

    address_word_9 = address_word_2 * 0x10000u + address_word_5;

    split_address_words(address_word_6, address_word_7,
                        int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_6, address_word_7));

    split_address_words(address_word_6, address_word_7,
                        int64_t(int32_t(matrix_accumulator_13)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_6, address_word_7));
    if (!condition_1) {
        uniform_index_word_7 = 0u;

        uniform_address_word_6 = blockDim.x;

        uniform_address_word_8 = blockDim.y;

        uniform_index_word_5 = 0x400u;

        uniform_index_word_5 = uniform_index_word_5 + 0x3000u;

        uniform_address_word_6 = uniform_address_word_6 * uniform_address_word_8;

        uniform_index_word_5 = (uniform_index_word_7 << 0x18u) + uniform_index_word_5;

        uniform_address_word_6 = 0u - uniform_address_word_6 + 0x100000u;

        uniform_index_word_7 = uniform_address_word_6 << (0xbu & 31u);

        uniform_address_word_6 = uniform_address_word_6 << (0x1u & 31u);

        __threadfence_block();

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_5);
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_5 + 0x8u);
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_5 + 0x10u);
#endif
    }
    packed_half_10 = parameters[parameter_geometry_or_control_16];

    phoenix_architecture::finish_compatibility_copies();

    condition_2 = int32_t(block_index_0) >= int32_t(0u);

    condition_1 = uint32_t(matrix_accumulator_17) > uint32_t(index_word_4);

    address_word_9 = index_word_4 + (0u - matrix_accumulator_17);

    matrix_fragment_64 = matrix_accumulator_30 & 0x3u;

    address_word_8 = min(int32_t(0u), int32_t(packed_half_10));

    uniform_index_word_5 = uint32_t(int32_t(uniform_address_word_9) >> 0x1fu);

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7),
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    address_word_8 = 0u - address_word_8 + matrix_accumulator_15;

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x200ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    condition_3 = uint32_t(address_word_8) != 0u;

    index_word_4 = !condition_1 ? address_word_9 : index_word_4;

    matrix_fragment_64 = index_word_11 | matrix_fragment_64;
    condition_1 = matrix_fragment_64 != 0;

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x400ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    if (!condition_2) {
        index_word_4 = 0u - index_word_4;
    }

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x600ull,
                                        matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
                                        matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x4000ull,
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);

    if (condition_3) {
        address_word_8 = address_word_8 + 0xffffffffu;
    }

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x4200ull,
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_address_word_9;

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x4400ull,
                                        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38,
                                        matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_6, address_word_7) + 0x4600ull,
                                        matrix_fragment_32, matrix_fragment_33, matrix_fragment_34,
                                        matrix_fragment_35, memory_ranges, error_counts);

    matrix_fragment_65 = !condition_0 ? matrix_fragment_65 : index_word_4;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    index_word_93 = 0u;

    address_word_95 = matrix_accumulator_31 * 0x2u;

    matrix_fragment_66 = address_word_3 + 0x20u;

    if (condition_3) {
        address_word_7 = uint32_t(int32_t(address_word_8) >> 0x1fu);
    }

    address_word_6 = uint32_t(int32_t(matrix_accumulator_15) >> 0x1fu);

    if (condition_3) {
        address_word_7 = uint32_t(combine_words(address_word_7, 0u) << 0x3u >> 32) + address_word_8;
    }

    address_word_6 =
        uint32_t(combine_words(address_word_6, 0u) << 0x2u >> 32) + matrix_accumulator_15;

    if (condition_3) {
        index_word_93 = uint32_t(int64_t(int32_t(address_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    address_word_7 = matrix_fragment_65 * 0x2u;

    index_word_4 = uint32_t(int32_t(address_word_6) >> 0x2u);
    if (!condition_1) {
        address_word_6 = matrix_accumulator_31 * 0x8u + matrix_accumulator_30;

        address_word_6 = address_word_6 + (0u - uniform_index_word_10);

        address_word_9 = address_word_6 + 0x4u;

        address_word_8 = address_word_6 + 0x3u;

        address_word_6 = max(int32_t(0u), int32_t(address_word_6));

        condition_0 = int32_t(address_word_9) > int32_t(0u);

        address_word_8 = uint32_t(combine_words(address_word_8, 0u) << 0x1du >> 32) + 0x1u;

        address_word_9 = matrix_fragment_65 * 0x8u + (0u - packed_half_10);

        address_word_6 = uint32_t(uint32_t(address_word_6) >> 0x3u);

        address_word_8 = condition_0 ? address_word_8 : 0u;

        packed_half_10 = address_word_9 + 0x8u;

        matrix_accumulator_13 = min(int32_t(address_word_8), int32_t(uniform_index_word_4));

        address_word_8 = address_word_9 + 0x7u;

        condition_0 = int32_t(packed_half_10) > int32_t(0u);

        address_word_9 = max(int32_t(0u), int32_t(address_word_9));

        condition_1 = int32_t(address_word_6) >= int32_t(matrix_accumulator_13);

        address_word_8 = uint32_t(combine_words(address_word_8, 0u) << 0x1du >> 32) + 0x1u;

        address_word_8 = condition_0 ? address_word_8 : 0u;
        if (!condition_1) {
            matrix_accumulator_12 = uint32_t(uint32_t(address_word_9) >> 0x3u);

            matrix_accumulator_17 = index_word_93 * 0x4u;

            matrix_accumulator_15 = min(int32_t(address_word_8), int32_t(index_word_93));
            do {
                condition_0 = int32_t(matrix_accumulator_12) >= int32_t(matrix_accumulator_15);
                if (!condition_0) {
                    address_word_8 = address_word_2 + address_word_6;

                    packed_half_10 = matrix_accumulator_12;

                    matrix_accumulator_19 = matrix_accumulator_17 * address_word_8;
                    do {
                        address_word_8 = parameters[parameter_wait_low_14];
                        address_word_9 = parameters[parameter_wait_high_15];

                        index_word_11 = matrix_accumulator_19 + packed_half_10;

                        split_address_words(address_word_8, address_word_9,
                                            int64_t(int32_t(index_word_11)) *
                                                    int64_t(int32_t(0x4u)) +
                                                combine_words(address_word_8, address_word_9));

                        index_word_11 = pipeline_read<CheckBounds>(
                            combine_words(address_word_8, address_word_9), memory_ranges,
                            error_counts);

                        packed_half_10 = packed_half_10 + 0x1u;

                        condition_1 = uint32_t(packed_half_10) != uint32_t(matrix_accumulator_15);

                        condition_0 = int32_t(index_word_11) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                index_word_11 = pipeline_read<CheckBounds>(
                                    combine_words(address_word_8, address_word_9), memory_ranges,
                                    error_counts);

                                condition_0 = int32_t(index_word_11) > -1;
                            } while (!condition_0);
                        }
                    } while (condition_1);
                }
                address_word_6 = address_word_6 + 0x1u;

                condition_0 = uint32_t(address_word_6) != uint32_t(matrix_accumulator_13);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    index_word_11 = parameters[parameter_tensor_height_8];

    index_word_90 = uint32_t(uint32_t(matrix_accumulator_30) >> 0x2u);

    address_word_6 = uint32_t(uint32_t(matrix_accumulator_30) >> 0x1u);

    address_word_8 = matrix_accumulator_30 * 0x20u;

    phoenix_architecture::finish_compatibility_copies();

    index_word_91 = address_word_95 + index_word_90;

    address_word_6 = address_word_6 & 0x1u;

    condition_0 = false;

    reused_word_88 = index_word_91 * index_word_4;

    address_word_9 = address_word_7 + address_word_6;

    uniform_index_word_10 = 0u;

    matrix_accumulator_13 = address_word_9;

    index_word_11 = index_word_11 & 0xfffffffcu;

    condition_2 = uint32_t(index_word_11) != uint32_t(0x4u);

    condition_1 = int32_t(index_word_91) < int32_t(uniform_index_word_5) || !condition_2;

    address_word_6 = condition_2 ? reused_word_88 : 0u;
    if (condition_1) {
        packed_half_10 = parameters[parameter_tensor_width_9];

        matrix_accumulator_13 = 0u;

        packed_half_10 = packed_half_10 & 0xfffffffcu;

        condition_2 = uint32_t(packed_half_10) != uint32_t(0x4u);

        condition_0 = int32_t(address_word_9) < int32_t(index_word_4) || !condition_2;

        if (condition_2) {
            matrix_accumulator_13 = address_word_9;
        }
    }
    if (condition_0) {
        matrix_accumulator_18 = parameters[parameter_input_low_0];
        matrix_accumulator_19 = parameters[parameter_input_high_1];
    }

    index_word_89 = address_word_8 & 0x20u;

    address_word_8 = matrix_accumulator_30 * 0x80u;

    if (condition_0) {
        packed_half_10 = address_word_6 + matrix_accumulator_13;
    }

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    index_word_89 = address_word_2 * 0x200u + index_word_89;

    matrix_accumulator_15 = address_word_8 & 0x100u;

    uniform_address_word_9 = 0x400u;

    address_word_8 = address_word_8 & 0x80u;

    if (condition_0) {
        packed_half_10 = packed_half_10 * 0x200u + index_word_89;
    }

    uniform_address_word_6 = (uniform_index_word_10 << 0x18u) + uniform_address_word_9;

    matrix_accumulator_15 = index_word_90 * 0x200u + matrix_accumulator_15;

    if (condition_0) {
        packed_half_10 = packed_half_10 * 0x4u;
    }

    address_word_8 = matrix_accumulator_15 + address_word_8;

    split_address_words(matrix_accumulator_14, matrix_accumulator_15, 0ull);

    if (condition_0) {
        matrix_accumulator_12 = packed_half_10;

        matrix_accumulator_17 = uint32_t(int32_t(packed_half_10) >> 0x1fu);
    }

    address_word_8 = (address_word_8 << 0x2u) + uniform_address_word_6;

    if (condition_0) {
        const uint64_t byte_address_0 =
            scaled_word_address(combine_words(matrix_accumulator_18, matrix_accumulator_19),
                                combine_words(matrix_accumulator_12, matrix_accumulator_17));
        packed_half_10 = uint32_t(byte_address_0);

        matrix_accumulator_14 = packed_half_10;

        matrix_accumulator_13 = uint32_t(byte_address_0 >> 32);

        matrix_accumulator_15 = matrix_accumulator_13;
    }
    if (!condition_0) {
        matrix_accumulator_13 = threadIdx.x & 31u;

        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_6;

        matrix_accumulator_15 = uniform_address_word_6;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        packed_half_10 = matrix_accumulator_13 * 0x10u + address_word_8;

        matrix_accumulator_13 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, packed_half_10, matrix_accumulator_12,
                                             matrix_accumulator_13, matrix_accumulator_14,
                                             matrix_accumulator_15, shared_bytes, memory_ranges,
                                             error_counts);
    } else {
        uniform_address_word_6 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_16 = broadcast_active_lane(matrix_accumulator_14);

            uniform_address_word_6 = uniform_address_word_9 + 0x3000u;

            uniform_address_word_17 = broadcast_active_lane(matrix_accumulator_15);

            uniform_address_word_14 = broadcast_active_lane(address_word_8);

            uniform_address_word_15 = (uniform_index_word_10 << 0x18u) + uniform_address_word_6;

            packed_half_10 = 0x200u;

            uniform_address_word_6 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_16, uniform_address_word_17),
                uniform_address_word_6 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_16, uniform_address_word_17),
                                   uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = false;

    matrix_accumulator_13 = address_word_9;
    if (condition_1) {
        packed_half_10 = parameters[parameter_tensor_width_9];

        matrix_accumulator_13 = 0u;

        packed_half_10 = packed_half_10 & 0xfffffffcu;

        condition_2 = uint32_t(packed_half_10) != uint32_t(0x4u);

        condition_0 = int32_t(address_word_9) < int32_t(index_word_4) || !condition_2;

        if (condition_2) {
            matrix_accumulator_13 = address_word_9;
        }
    }
    packed_half_10 = (address_word_2 << 0x9u) + 0x40u;
    if (!condition_0) {
        matrix_accumulator_13 = threadIdx.x & 31u;

        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_6;

        matrix_accumulator_15 = uniform_address_word_6;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_16 = matrix_accumulator_13 * 0x10u + address_word_8;

        matrix_accumulator_13 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_16 + 0x1000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];

            matrix_accumulator_12 = address_word_6 + matrix_accumulator_13;
        }

        uniform_address_word_6 = 0xffffffffu;

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            matrix_accumulator_12 = matrix_accumulator_12 * 0x200u + index_word_89;

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
            uniform_index_word_7 = broadcast_active_lane(address_word_8);

            uniform_address_word_6 = uniform_address_word_9 + 0x3000u;

            uniform_address_word_14 = broadcast_active_lane(matrix_accumulator_18);

            uniform_address_word_11 = 0x20u;

            uniform_address_word_15 = broadcast_active_lane(matrix_accumulator_19);

            uniform_address_word_8 = (uniform_index_word_10 << 0x18u) + uniform_address_word_6;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_6 = uniform_index_word_7 + 0x1000u;

            uniform_index_word_7 = uniform_address_word_8 + 0x8u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_14, uniform_address_word_15),
                uniform_address_word_11 * 16u, uniform_address_word_8 + 0x8u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_14, uniform_address_word_15),
                                   uniform_address_word_11 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    condition_0 = uint32_t(matrix_fragment_64) != 0u;
    if (!condition_0) {
        matrix_accumulator_14 = parameters[parameter_geometry_or_control_16];
        matrix_accumulator_15 = parameters[parameter_geometry_or_control_17];

        matrix_accumulator_12 = matrix_accumulator_31 * 0x8u + matrix_accumulator_30;

        matrix_accumulator_12 = matrix_accumulator_12 + (0u - matrix_accumulator_15);

        matrix_accumulator_15 = matrix_accumulator_12 + 0x4u;

        matrix_accumulator_13 = matrix_accumulator_12 + 0x3u;

        matrix_accumulator_12 = max(int32_t(0u), int32_t(matrix_accumulator_12));

        condition_0 = int32_t(matrix_accumulator_15) > int32_t(0u);

        matrix_accumulator_13 =
            uint32_t(combine_words(matrix_accumulator_13, 0u) << 0x1du >> 32) + 0x1u;

        matrix_accumulator_15 = matrix_fragment_65 * 0x8u + (0u - matrix_accumulator_14);

        matrix_accumulator_14 = uint32_t(uint32_t(matrix_accumulator_12) >> 0x3u);

        matrix_accumulator_13 = condition_0 ? matrix_accumulator_13 : 0u;

        matrix_accumulator_16 = matrix_accumulator_15 + 0x8u;

        matrix_accumulator_12 = matrix_accumulator_15 + 0x7u;

        matrix_accumulator_15 = max(int32_t(0u), int32_t(matrix_accumulator_15));

        matrix_accumulator_21 = min(int32_t(matrix_accumulator_13), int32_t(uniform_index_word_4));

        condition_2 = int32_t(matrix_accumulator_16) > int32_t(0u);

        matrix_accumulator_12 =
            uint32_t(combine_words(matrix_accumulator_12, 0u) << 0x1du >> 32) + 0x1u;

        condition_0 = int32_t(matrix_accumulator_14) >= int32_t(matrix_accumulator_21);

        matrix_accumulator_16 = condition_2 ? matrix_accumulator_12 : 0u;
        if (!condition_0) {
            matrix_accumulator_12 = packed_half_10 + 0x40u;

            matrix_accumulator_18 = uint32_t(uint32_t(matrix_accumulator_15) >> 0x3u);

            matrix_accumulator_23 = min(int32_t(matrix_accumulator_16), int32_t(index_word_93));

            matrix_accumulator_15 = uint32_t(uint32_t(matrix_accumulator_12) >> 0x7u);
            do {
                condition_0 = int32_t(matrix_accumulator_18) >= int32_t(matrix_accumulator_23);
                if (!condition_0) {
                    matrix_accumulator_12 = matrix_accumulator_14 * 0x4u + matrix_accumulator_15;

                    matrix_accumulator_16 = matrix_accumulator_18;

                    matrix_accumulator_19 = matrix_accumulator_12 * index_word_93;
                    do {
                        matrix_accumulator_12 = parameters[parameter_wait_low_14];
                        matrix_accumulator_13 = parameters[parameter_wait_high_15];

                        matrix_accumulator_17 = matrix_accumulator_19 + matrix_accumulator_16;

                        split_address_words(
                            matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));

                        matrix_accumulator_17 = pipeline_read<CheckBounds>(
                            combine_words(matrix_accumulator_12, matrix_accumulator_13),
                            memory_ranges, error_counts);

                        matrix_accumulator_16 = matrix_accumulator_16 + 0x1u;

                        condition_2 =
                            uint32_t(matrix_accumulator_16) != uint32_t(matrix_accumulator_23);

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
                    } while (condition_2);
                }
                matrix_accumulator_14 = matrix_accumulator_14 + 0x1u;

                condition_0 = uint32_t(matrix_accumulator_14) != uint32_t(matrix_accumulator_21);
            } while (condition_0);
        }
    }
    __syncwarp(__activemask());

    phoenix_architecture::finish_compatibility_copies();

    condition_0 = false;

    matrix_accumulator_13 = address_word_9;
    if (condition_1) {
        matrix_accumulator_12 = parameters[parameter_tensor_width_9];

        matrix_accumulator_13 = 0u;

        matrix_accumulator_12 = matrix_accumulator_12 & 0xfffffffcu;

        condition_1 = uint32_t(matrix_accumulator_12) != uint32_t(0x4u);

        condition_0 = int32_t(address_word_9) < int32_t(index_word_4) || !condition_1;

        if (condition_1) {
            matrix_accumulator_13 = address_word_9;
        }
    }
    if (!condition_0) {
        matrix_accumulator_13 = threadIdx.x & 31u;

        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_accumulator_12 = uniform_address_word_6;

        matrix_accumulator_15 = uniform_address_word_6;

        matrix_accumulator_12 =
            permute_signed(matrix_accumulator_12, matrix_accumulator_15, 0x5410u);

        matrix_accumulator_14 = matrix_accumulator_12;

        matrix_accumulator_15 = matrix_accumulator_12;

        matrix_accumulator_16 = matrix_accumulator_13 * 0x10u + address_word_8;

        matrix_accumulator_13 = matrix_accumulator_12;

        store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_16 + 0x2000u,
                                             matrix_accumulator_12, matrix_accumulator_13,
                                             matrix_accumulator_14, matrix_accumulator_15,
                                             shared_bytes, memory_ranges, error_counts);
    } else {
        if (condition_0) {
            matrix_accumulator_16 = parameters[parameter_input_low_0];
            matrix_accumulator_17 = parameters[parameter_input_high_1];

            matrix_accumulator_12 = address_word_6 + matrix_accumulator_13;
        }

        __syncwarp(__activemask());

        if (condition_0) {
            matrix_accumulator_12 = matrix_accumulator_12 * 0x200u + index_word_89;
        }

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;

        split_address_words(matrix_accumulator_18, matrix_accumulator_19, 0ull);

        if (condition_0) {
            matrix_accumulator_14 = (matrix_accumulator_12 << 0x2u) + 0x200u;
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        if (condition_0) {
            matrix_accumulator_15 = uint32_t(int32_t(matrix_accumulator_14) >> 0x1fu);

            matrix_accumulator_12 = matrix_accumulator_14;

            matrix_accumulator_13 = matrix_accumulator_15;

            const uint64_t byte_address_2 =
                scaled_word_address(combine_words(matrix_accumulator_16, matrix_accumulator_17),
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));
            matrix_accumulator_14 = uint32_t(byte_address_2);

            matrix_accumulator_13 = uint32_t(byte_address_2 >> 32);

            matrix_accumulator_18 = matrix_accumulator_14;

            matrix_accumulator_19 = matrix_accumulator_13;
        }
        if (condition_1) {
            uniform_index_word_7 = 0u;

            check_single_lane<CheckBounds>(error_counts);
            uniform_address_word_9 = address_word_8;

            uniform_address_word_6 = 0x400u;

            check_single_lane<CheckBounds>(error_counts);
            uniform_index_word_10 = matrix_accumulator_18;

            uniform_address_word_6 = uniform_address_word_6 + 0x3000u;

            check_single_lane<CheckBounds>(error_counts);
            uniform_address_word_11 = matrix_accumulator_19;

            matrix_accumulator_12 = 0x200u;

            uniform_address_word_8 = (uniform_index_word_7 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = uniform_address_word_9 + 0x2000u;

            uniform_index_word_7 = uniform_address_word_8 + 0x10u;

            uniform_address_word_9 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_index_word_10, uniform_address_word_11),
                uniform_address_word_9 * 16u, uniform_address_word_8 + 0x10u, shared_bytes,
                memory_ranges, error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_index_word_10, uniform_address_word_11),
                                   uniform_address_word_9 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_13 = 0u;

    matrix_accumulator_14 = parameters[parameter_tensor_width_9];

    matrix_accumulator_12 = 0x400u;

    condition_1 = int32_t(index_word_91) >= int32_t(uniform_index_word_5);

    condition_0 = int32_t(index_word_91) >= int32_t(uniform_index_word_5);

    matrix_accumulator_16 = threadIdx.x & 31u;

    matrix_accumulator_12 = matrix_accumulator_12 + 0x3000u;

    condition_1 = uint32_t(index_word_11) != uint32_t(0x4u) && condition_1;

    condition_0 = uint32_t(index_word_11) == uint32_t(0x4u) || !condition_0;

    index_word_11 = address_word_7 + 0x1u;

    matrix_accumulator_28 = parameters[parameter_trained_weights_low_6];
    matrix_accumulator_29 = parameters[parameter_trained_weights_high_7];

    matrix_accumulator_14 = matrix_accumulator_14 & 0xfffffffcu;

    condition_3 = uint32_t(matrix_accumulator_14) == uint32_t(0x4u) || condition_1;

    condition_1 = uint32_t(matrix_accumulator_14) == uint32_t(0x4u) && !condition_1;

    matrix_accumulator_25 = (matrix_accumulator_13 << 0x18u) + matrix_accumulator_12;

    condition_2 = int32_t(address_word_7) < int32_t(index_word_4) || condition_3;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_accumulator_25;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_12 = 0u;
#endif

    condition_4 = (int32_t(index_word_11) < int32_t(index_word_4)) || (condition_3);

    matrix_fragment_73 = !condition_1 ? address_word_7 : 0u;

    matrix_accumulator_17 = matrix_accumulator_16;

    condition_2 = condition_2 & condition_0;

    condition_4 = condition_4 & condition_0;

    matrix_fragment_67 = matrix_accumulator_17 & 0x7ffffffcu;

    matrix_accumulator_77 = !condition_1 ? index_word_11 : 0u;

    matrix_fragment_67 = 0u - matrix_fragment_67 + matrix_accumulator_16;

    if (condition_2) {
        matrix_accumulator_14 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_15 = parameters[parameter_input_skip_high_3];

        matrix_accumulator_18 = address_word_6 + matrix_fragment_73;
    }

    if (!condition_2) {
        matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    matrix_accumulator_24 = matrix_fragment_67 * 0x2u + address_word_3;

    if (condition_2) {
        matrix_accumulator_17 = matrix_accumulator_18 * 0x200u + address_word_3;
    }

    if (condition_4) {
        matrix_accumulator_18 = address_word_6 + matrix_accumulator_77;
    }

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_5 = wait_copy_barrier(shared_tile, matrix_accumulator_25,
                                    async_tokens[(((matrix_accumulator_25) >> 3) & 1)]);
#else
    phoenix_architecture::finish_compatibility_copies();
    condition_5 = true;
#endif

    if (!condition_2) {
        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_23, matrix_accumulator_23, 0x5410u);
    }

    if (condition_2) {
        matrix_accumulator_17 = matrix_accumulator_17 * 0x4u;
    }

    if (condition_4) {
        matrix_accumulator_18 = matrix_accumulator_18 * 0x200u + address_word_3;
    }

    if (!condition_2) {
        matrix_accumulator_20 = matrix_accumulator_23;

        matrix_accumulator_21 = matrix_accumulator_23;

        matrix_accumulator_22 = matrix_accumulator_23;
    }

    if (condition_4) {
        matrix_fragment_68 = parameters[parameter_input_skip_low_2];
        matrix_fragment_69 = parameters[parameter_input_skip_high_3];
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_14, matrix_accumulator_15,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));
    }

    if (condition_4) {
        matrix_accumulator_17 = matrix_accumulator_18 * 0x4u;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_18, matrix_accumulator_19,
                            int64_t(int32_t(matrix_accumulator_16)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));
    }
    if (!condition_5) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_5 = wait_copy_barrier(shared_tile, matrix_accumulator_25,
                                            async_tokens[(((matrix_accumulator_25) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_5 = true;
#endif
        } while (!condition_5);
    }
    if (condition_2) {
        matrix_accumulator_14 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_15 = parameters[parameter_input_skip_high_3];
    }

    matrix_fragment_74 = address_word_3 + 0x18u;

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_18, matrix_accumulator_19), matrix_accumulator_20,
            matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
            error_counts);
    }

    matrix_fragment_71 = uint32_t(int32_t(matrix_accumulator_24) >> 0x1u);

    if (condition_4) {
        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_68, matrix_fragment_69));

        matrix_accumulator_12 = parameters[parameter_input_skip_low_2];
        matrix_accumulator_13 = parameters[parameter_input_skip_high_3];
    }

    matrix_fragment_74 = matrix_fragment_67 * 0x2u + matrix_fragment_74;

    if (condition_2) {
        matrix_accumulator_19 = address_word_6 + matrix_fragment_73;
    }

    matrix_accumulator_18 = address_word_3 + 0x10u;

    if (!condition_4) {
        matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    matrix_accumulator_17 =
        uint32_t(combine_words(matrix_fragment_74, 0u) << 0x1u >> 32) + matrix_fragment_74;

    if (condition_2) {
        matrix_accumulator_19 = matrix_accumulator_19 * 0x200u + matrix_fragment_66;
    }

    matrix_fragment_72 = address_word_3 + 0x8u;

    if (!condition_4) {
        matrix_accumulator_27 =
            permute_signed(matrix_accumulator_27, matrix_accumulator_27, 0x5410u);
    }

    matrix_accumulator_18 = matrix_fragment_67 * 0x2u + matrix_accumulator_18;

    if (condition_4) {
        matrix_fragment_73 = address_word_6 + matrix_accumulator_77;
    }

    split_address_words(matrix_fragment_70, matrix_fragment_71,
                        int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_17) >> 0x1u);

    if (condition_2) {
        matrix_accumulator_19 = matrix_accumulator_19 * 0x4u;
    }

    if (!condition_4) {
        matrix_accumulator_24 = matrix_accumulator_27;

        matrix_accumulator_25 = matrix_accumulator_27;

        matrix_accumulator_26 = matrix_accumulator_27;
    }

    matrix_fragment_72 = matrix_fragment_67 * 0x2u + matrix_fragment_72;

    matrix_accumulator_77 = uint32_t(int32_t(matrix_accumulator_18) >> 0x1u);

    if (condition_4) {
        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            int64_t(int32_t(matrix_accumulator_16)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_68, matrix_fragment_69));
    }

    matrix_accumulator_85 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_70, matrix_fragment_71) + 0x40000ull, memory_ranges,
        error_counts);

    if (condition_2) {
        split_address_words(matrix_accumulator_18, matrix_accumulator_19,
                            int64_t(int32_t(matrix_accumulator_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_14, matrix_accumulator_15));
    }

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_72) >> 0x1u);

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_68, matrix_fragment_69), matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);

        matrix_fragment_73 = matrix_fragment_73 * 0x200u + matrix_fragment_66;
    }

    split_address_words(matrix_fragment_70, matrix_fragment_71,
                        int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (!condition_2) {
        matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_4) {
        matrix_fragment_73 = matrix_fragment_73 * 0x4u;
    }

    if (!condition_2) {
        matrix_accumulator_15 =
            permute_signed(matrix_accumulator_17, matrix_accumulator_17, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            int64_t(int32_t(matrix_accumulator_16)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_18, matrix_accumulator_19));
    }

    if (!condition_4) {
        matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_71 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_70, matrix_fragment_71) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (!condition_2) {
        matrix_accumulator_14 = matrix_accumulator_15;
    }

    if (!condition_4) {
        matrix_accumulator_19 =
            permute_signed(matrix_accumulator_19, matrix_accumulator_19, 0x5410u);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(matrix_fragment_73)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_12, matrix_accumulator_13));
    }

    if (!condition_2) {
        matrix_accumulator_12 = matrix_accumulator_15;

        matrix_accumulator_13 = matrix_accumulator_15;
    }

    matrix_accumulator_83 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_fragment_66 = matrix_fragment_67 * 0x2u + matrix_fragment_66;

    if (!condition_4) {
        matrix_accumulator_17 = matrix_accumulator_19;

        matrix_accumulator_18 = matrix_accumulator_19;
    }

    matrix_fragment_70 = address_word_3 + 0x28u;

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_68, matrix_fragment_69), matrix_accumulator_12,
            matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
            error_counts);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            int64_t(int32_t(matrix_accumulator_16)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_72, matrix_fragment_73));
    }

    if (!condition_4) {
        matrix_accumulator_16 = matrix_accumulator_19;
    }

    matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_66) >> 0x1u);

    matrix_fragment_68 = address_word_3 + 0x38u;

    matrix_fragment_66 = address_word_3 + 0x30u;

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        int64_t(int32_t(matrix_fragment_73)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_74, matrix_fragment_75), matrix_accumulator_16,
            matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
            error_counts);
    }

    matrix_fragment_70 = matrix_fragment_67 * 0x2u + matrix_fragment_70;

    matrix_fragment_68 = matrix_fragment_67 * 0x2u + matrix_fragment_68;

    matrix_fragment_72 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_76, matrix_accumulator_77,
                        int64_t(int32_t(matrix_accumulator_77)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    matrix_fragment_70 =
        uint32_t(combine_words(matrix_fragment_70, 0u) << 0x1u >> 32) + matrix_fragment_70;

    matrix_fragment_66 = matrix_fragment_67 * 0x2u + matrix_fragment_66;

    matrix_fragment_68 =
        uint32_t(combine_words(matrix_fragment_68, 0u) << 0x1u >> 32) + matrix_fragment_68;

    matrix_accumulator_81 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_76, matrix_accumulator_77) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_fragment_75 = uint32_t(int32_t(matrix_fragment_66) >> 0x1u);

    matrix_fragment_67 = uint32_t(int32_t(matrix_fragment_68) >> 0x1u);

    matrix_accumulator_77 = uint32_t(int32_t(matrix_fragment_70) >> 0x1u);

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        int64_t(int32_t(matrix_fragment_75)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    split_address_words(matrix_accumulator_76, matrix_accumulator_77,
                        int64_t(int32_t(matrix_accumulator_77)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    matrix_fragment_68 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0x40000ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                        int64_t(int32_t(matrix_fragment_67)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_28, matrix_accumulator_29));

    matrix_fragment_70 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_76, matrix_accumulator_77) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_fragment_66 = parameters[parameter_geometry_or_control_16];
    matrix_fragment_67 = parameters[parameter_geometry_or_control_17];

    matrix_fragment_69 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_28, matrix_accumulator_29) + 0x40000ull, memory_ranges,
        error_counts);

    matrix_accumulator_86 = matrix_accumulator_31 * 0x8u + matrix_accumulator_30;

    matrix_fragment_73 = matrix_accumulator_86 + (0u - matrix_fragment_67);

    matrix_fragment_66 = matrix_fragment_65 * 0x8u + (0u - matrix_fragment_66);

    matrix_fragment_67 = matrix_fragment_73 + 0x3u;

    matrix_fragment_65 = matrix_fragment_66 + 0x8u;

    matrix_fragment_74 = uint32_t(combine_words(matrix_fragment_67, 0u) << 0x1du >> 32) + 0x1u;

    index_word_101 = max(int32_t(0u), int32_t(matrix_fragment_73));

    condition_4 = int32_t(matrix_fragment_65) > int32_t(0u);

    index_word_101 = uint32_t(uint32_t(index_word_101) >> 0x3u);

    index_word_99 = max(int32_t(0u), int32_t(matrix_fragment_66));

    index_word_103 = 0u;

    index_word_99 = uint32_t(uint32_t(index_word_99) >> 0x3u);

    matrix_accumulator_79 = unpack_fp8_bits(matrix_accumulator_21);

    matrix_accumulator_77 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

    matrix_accumulator_30 = unpack_fp8_bits(matrix_accumulator_22);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

    matrix_accumulator_31 = unpack_fp8_bits(matrix_accumulator_23);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

    matrix_accumulator_78 = unpack_fp8_bits(matrix_accumulator_20);

    matrix_accumulator_20 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_25);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_25 >> 16);

    matrix_accumulator_84 = unpack_fp8_bits(matrix_accumulator_24);

    matrix_accumulator_82 = unpack_fp8_bits(matrix_accumulator_24 >> 16);

    matrix_accumulator_25 = unpack_fp8_bits(matrix_accumulator_27);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_71 = matrix_fragment_73 + 0x4u;

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_86 = unpack_fp8_bits(matrix_accumulator_14 >> 16);

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_14);

    condition_2 = int32_t(matrix_fragment_71) > int32_t(0u);

    matrix_accumulator_80 = unpack_fp8_bits(matrix_accumulator_26);

    matrix_accumulator_14 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_fragment_67 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_fragment_71 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_14),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_74 = condition_2 ? matrix_fragment_74 : 0u;

    matrix_accumulator_14 = matrix_fragment_66 + 0x7u;

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_12);

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    index_word_96 = min(int32_t(matrix_fragment_74), int32_t(uniform_index_word_4));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_65 = uint32_t(combine_words(matrix_accumulator_14, 0u) << 0x1du >> 32) + 0x1u;

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_12 >> 16);

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_13);

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_25 = unpack_fp8_bits(matrix_accumulator_13 >> 16);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_15);

    matrix_accumulator_87 = unpack_fp8_bits(matrix_accumulator_15 >> 16);

    matrix_accumulator_12 = unpack_fp8_bits(matrix_accumulator_16);

    matrix_accumulator_13 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    matrix_accumulator_15 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    condition_2 = int32_t(index_word_101) >= int32_t(index_word_96);

    index_word_94 = condition_4 ? matrix_fragment_65 : 0u;

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_12),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_13),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    condition_2 = uint32_t(matrix_fragment_64) != 0u || condition_2;

    matrix_accumulator_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    index_word_94 = min(int32_t(index_word_94), int32_t(index_word_93));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_69));
    do {
        index_word_102 = uint32_t(uint32_t(index_word_103) >> 0x6u);

        index_word_98 = 0u;

        index_word_105 = 0x400u;

        matrix_fragment_64 = permute_signed(index_word_102, 0u, 0x9910u);

        warp_lane_100 = threadIdx.x & 31u;

        __syncwarp(__activemask());

        matrix_fragment_64 = matrix_fragment_64 * 0xabu;

        condition_5 = uint32_t(index_word_103) > uint32_t(0x1bfu);

        condition_4 = uint32_t(index_word_103) > uint32_t(0x13fu);

        matrix_fragment_64 = matrix_fragment_64 & 0xffffu;

        matrix_fragment_64 = uint32_t(uint32_t(matrix_fragment_64) >> 0x9u);

        matrix_fragment_64 = permute_signed(matrix_fragment_64, 0u, 0x9910u);

        matrix_fragment_64 = matrix_fragment_64 * 0x3u;

        matrix_fragment_64 = 0u - matrix_fragment_64;

        matrix_fragment_65 = permute_signed(matrix_fragment_64, 0u, 0x7710u);

        index_word_102 = index_word_102 + matrix_fragment_65;

        matrix_fragment_65 = (index_word_98 << 0x18u) + index_word_105;

        index_word_107 = index_word_102 * 0x1000u;

        index_word_107 = 0xff000u & index_word_107;

        matrix_fragment_64 = index_word_90 * 0x800u + index_word_107;

        matrix_fragment_65 = matrix_fragment_65 + matrix_fragment_64;

        tile_word_104 = warp_lane_100 * 0x10u + matrix_fragment_65;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_104, matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
            matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_104 + 0x400u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_104 + 0x600u, matrix_fragment_72, matrix_fragment_73,
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
            shared_tile, tile_word_104 + 0x200u, matrix_fragment_68, matrix_fragment_69,
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
        if (!condition_5) {
            matrix_fragment_66 = index_word_103 + 0x40u;

            matrix_fragment_32 = parameters[parameter_trained_weights_low_6];
            matrix_fragment_33 = parameters[parameter_trained_weights_high_7];

            matrix_fragment_34 = packed_half_10 + index_word_103;

            matrix_fragment_66 = uint32_t(uint32_t(matrix_fragment_66) >> 0x6u);

            matrix_fragment_35 = matrix_fragment_34 * 0x80u + address_word_5;

            matrix_fragment_67 = permute_signed(matrix_fragment_66, 0u, 0x9910u);

            matrix_fragment_67 = matrix_fragment_67 * 0xabu;

            matrix_fragment_67 = matrix_fragment_67 & 0xffffu;

            matrix_fragment_67 = uint32_t(uint32_t(matrix_fragment_67) >> 0x9u);

            split_address_words(matrix_fragment_32, matrix_fragment_33,
                                int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_32, matrix_fragment_33));

            matrix_fragment_67 = permute_signed(matrix_fragment_67, 0u, 0x9910u);

            split_address_words(matrix_fragment_64, matrix_fragment_65,
                                int64_t(int32_t(warp_lane_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_32, matrix_fragment_33));

            matrix_fragment_67 = matrix_fragment_67 * 0x3u;

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65), matrix_fragment_60,
                matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
                error_counts);

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
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x4000ull,
                matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x4200ull,
                matrix_fragment_40, matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x4400ull,
                matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_64, matrix_fragment_65) + 0x4600ull,
                matrix_fragment_32, matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                memory_ranges, error_counts);

            index_word_105 = index_word_105 + 0x3000u;

            matrix_fragment_65 = 0u - matrix_fragment_67;

            matrix_fragment_67 = (index_word_98 << 0x18u) + index_word_105;

            matrix_fragment_65 = permute_signed(matrix_fragment_65, 0u, 0x7710u);

            matrix_fragment_66 = matrix_fragment_66 + matrix_fragment_65;

            matrix_fragment_66 = matrix_fragment_66 & 0xffu;

            matrix_fragment_67 = matrix_fragment_66 * 0x8u + matrix_fragment_67;

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
            condition_5 = wait_copy_barrier(shared_tile, matrix_fragment_67,
                                            async_tokens[(((matrix_fragment_67) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_5 = true;
#endif
            if (!condition_5) {
                do {
#if DLSS5_ENABLE_ASYNC_COPIES
                    condition_5 =
                        wait_copy_barrier(shared_tile, matrix_fragment_67,
                                          async_tokens[(((matrix_fragment_67) >> 3) & 1)]);
#else
                    phoenix_architecture::finish_compatibility_copies();
                    condition_5 = true;
#endif
                } while (!condition_5);
            }
        }
        matrix_fragment_66 = index_word_103 + 0xc0u;

        index_word_102 = index_word_102 & 0xffu;
        if (!condition_4) {
            condition_4 = (matrix_fragment_66 & 0x40u) != 0;
            if (!condition_4) {
                if (!condition_2) {
                    matrix_fragment_64 = address_word_2 * 0x200u + matrix_fragment_66;

                    matrix_fragment_67 = index_word_101;

                    matrix_fragment_68 = uint32_t(uint32_t(matrix_fragment_64) >> 0x7u);
                    do {
                        condition_4 = int32_t(index_word_99) >= int32_t(index_word_94);
                        if (!condition_4) {
                            matrix_fragment_64 = matrix_fragment_67 * 0x4u + matrix_fragment_68;

                            matrix_fragment_69 = index_word_99;

                            matrix_fragment_70 = matrix_fragment_64 * index_word_93;
                            do {
                                matrix_fragment_64 = parameters[parameter_wait_low_14];
                                matrix_fragment_65 = parameters[parameter_wait_high_15];

                                matrix_fragment_71 = matrix_fragment_70 + matrix_fragment_69;

                                split_address_words(
                                    matrix_fragment_64, matrix_fragment_65,
                                    int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_64, matrix_fragment_65));

                                matrix_fragment_71 = pipeline_read<CheckBounds>(
                                    combine_words(matrix_fragment_64, matrix_fragment_65),
                                    memory_ranges, error_counts);

                                matrix_fragment_69 = matrix_fragment_69 + 0x1u;

                                condition_5 =
                                    uint32_t(matrix_fragment_69) != uint32_t(index_word_94);

                                condition_4 = int32_t(matrix_fragment_71) > -1;
                                if (!condition_4) {
                                    do {
                                        __nanosleep(0x40u);

                                        matrix_fragment_71 = pipeline_read<CheckBounds>(
                                            combine_words(matrix_fragment_64, matrix_fragment_65),
                                            memory_ranges, error_counts);

                                        condition_4 = int32_t(matrix_fragment_71) > -1;
                                    } while (!condition_4);
                                }
                            } while (condition_5);
                        }
                        matrix_fragment_67 = matrix_fragment_67 + 0x1u;

                        condition_4 = uint32_t(matrix_fragment_67) != uint32_t(index_word_96);
                    } while (condition_4);
                }
                __syncwarp(__activemask());

                phoenix_architecture::finish_compatibility_copies();
            }
            condition_4 = int32_t(address_word_9) < int32_t(index_word_4) || condition_3;

            condition_4 = condition_4 & condition_0;
            if (!condition_4) {
                matrix_fragment_68 = threadIdx.x & 31u;

                uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                index_word_107 = address_word_8 + index_word_107;

                matrix_fragment_64 = uniform_address_word_6;

                matrix_fragment_65 = uniform_address_word_6;

                matrix_fragment_64 =
                    permute_signed(matrix_fragment_64, matrix_fragment_65, 0x5410u);

                matrix_fragment_65 = matrix_fragment_64;

                matrix_fragment_66 = matrix_fragment_64;

                matrix_fragment_67 = matrix_fragment_64;

                matrix_fragment_68 = matrix_fragment_68 * 0x10u + index_word_107;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_68,
                                                     matrix_fragment_64, matrix_fragment_65,
                                                     matrix_fragment_66, matrix_fragment_67,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (condition_4) {
                    matrix_fragment_67 = !condition_1 ? address_word_9 : 0u;

                    matrix_fragment_64 = parameters[parameter_input_low_0];
                    matrix_fragment_65 = parameters[parameter_input_high_1];

                    matrix_fragment_66 = index_word_89 + matrix_fragment_66;
                }

                __syncwarp(__activemask());

                if (condition_4) {
                    matrix_fragment_67 = address_word_6 + matrix_fragment_67;
                }

                condition_5 = int(threadIdx.x) == __ffs(__activemask()) - 1;

                if (condition_4) {
                    matrix_fragment_66 = matrix_fragment_67 * 0x200u + matrix_fragment_66;

                    matrix_fragment_69 = matrix_fragment_66 * 0x4u;
                }

                split_address_words(matrix_fragment_66, matrix_fragment_67, 0ull);

                if (condition_4) {
                    matrix_fragment_66 = matrix_fragment_69;

                    matrix_fragment_67 = uint32_t(int32_t(matrix_fragment_69) >> 0x1fu);
                }

                split_address_words(matrix_fragment_68, matrix_fragment_69, 0ull);

                if (condition_4) {
                    const uint64_t byte_address_3 =
                        scaled_word_address(combine_words(matrix_fragment_64, matrix_fragment_65),
                                            combine_words(matrix_fragment_66, matrix_fragment_67));
                    matrix_fragment_71 = uint32_t(byte_address_3);

                    matrix_fragment_69 = uint32_t(byte_address_3 >> 32);

                    matrix_fragment_68 = matrix_fragment_71;
                }
                if (condition_5) {
                    uniform_index_word_7 = 0u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_9 = index_word_107;

                    uniform_address_word_6 = 0x400u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_8 = address_word_8;

                    uniform_address_word_6 = uniform_address_word_6 + 0x3000u;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_index_word_10 = index_word_102;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_14 = matrix_fragment_68;

                    check_single_lane<CheckBounds>(error_counts);
                    uniform_address_word_15 = matrix_fragment_69;

                    matrix_fragment_64 = 0x200u;

                    uniform_index_word_7 = (uniform_index_word_7 << 0x18u) + uniform_address_word_6;

                    uniform_address_word_6 = uniform_address_word_8 + uniform_address_word_9;

                    uniform_index_word_7 = (uniform_index_word_10 << 0x3u) + uniform_index_word_7;

                    uniform_address_word_8 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_8 * 16u, uniform_index_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_14, uniform_address_word_15),
                        uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        }
        condition_4 = uint32_t(index_word_103) >= uint32_t(0x1c0u);

        index_word_103 = index_word_103 + 0x40u;
    } while (!condition_4);
    address_word_5 = threadIdx.z;

    address_word_8 = parameters[parameter_output_low_4];
    address_word_9 = parameters[parameter_output_high_5];

    condition_0 = int32_t(index_word_91) >= int32_t(uniform_index_word_5);

    reused_word_88 = address_word_7 + reused_word_88;

    condition_1 = int32_t(address_word_7) >= int32_t(index_word_4) || condition_0;

    address_word_3 = reused_word_88 * 0x200u + address_word_3;

    condition_0 = int32_t(index_word_11) >= int32_t(index_word_4) || condition_0;

    address_word_3 = address_word_3 * 0x4u;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_8, address_word_9));

    condition_2 = (address_word_5 | index_word_92) != 0;
    if (!condition_1) {
        matrix_fragment_33 = threadIdx.x & 31u;

        address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       select_half_operand<2, false, false>(0u));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                          select_half_operand<2, false, false>(address_word_8));

        address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(0u));

        index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                     select_half_operand<2, false, false>(0u));

        matrix_accumulator_23 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(packed_half_10));

        address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                          select_half_operand<2, false, false>(0u));

        address_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                          select_half_operand<2, false, false>(0u));

        address_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(index_word_4));

        address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(address_word_5));

        matrix_accumulator_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                          select_half_operand<2, false, false>(matrix_accumulator_27));

        index_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(address_word_6));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_26 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(matrix_accumulator_29));

        split_address_words(index_word_4, address_word_5,
                            int64_t(int32_t(matrix_fragment_33)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_27 = matrix_accumulator_23;

        store_four_words_streaming<CheckBounds>(combine_words(index_word_4, address_word_5),
                                                address_word_8, address_word_9, packed_half_10,
                                                index_word_11, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(index_word_4, address_word_5) + 0x200ull, matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);
    }
    index_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                      select_half_operand<2, false, false>(0u));

    index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                 select_half_operand<2, false, false>(0u));

    address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                   select_half_operand<2, false, false>(0u));

    address_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                   select_half_operand<2, false, false>(0u));

    address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                   select_half_operand<2, false, false>(0u));

    address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                   select_half_operand<2, false, false>(0u));

    address_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                   select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_14),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        matrix_accumulator_21 = threadIdx.x & 31u;

        address_word_7 = permute_signed(address_word_7, packed_half_10, 0x5410u);

        address_word_6 = permute_signed(address_word_6, index_word_11, 0x5410u);

        address_word_5 = permute_signed(address_word_5, address_word_8, 0x5410u);

        index_word_4 = permute_signed(index_word_4, address_word_9, 0x5410u);

        index_word_11 = permute_signed(matrix_accumulator_12, matrix_accumulator_17, 0x5410u);

        packed_half_10 = permute_signed(matrix_accumulator_14, matrix_accumulator_15, 0x5410u);

        address_word_9 = permute_signed(matrix_accumulator_20, matrix_accumulator_13, 0x5410u);

        address_word_8 = permute_signed(matrix_accumulator_18, matrix_accumulator_19, 0x5410u);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(matrix_accumulator_21)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, address_word_3));

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, address_word_3) + 0x2000ull, index_word_4, address_word_5,
            address_word_6, address_word_7, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, address_word_3) + 0x2200ull, address_word_8,
            address_word_9, packed_half_10, index_word_11, memory_ranges, error_counts);
    }
    __syncwarp(__activemask());

    phoenix_architecture::finish_compatibility_copies();

    if (condition_2) {
        return;
    }

    address_word_2 = parameters[parameter_publication_optional_low_12];
    address_word_3 = parameters[parameter_publication_optional_high_13];

    address_word_95 = index_word_97 * address_word_95 + block_index_0;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_95)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(address_word_2, address_word_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__ __maxnreg__(142) void phoenix_global_attention_split_heads_512_projection(
    KernelArguments parameters) {
    run_global_attention_split_heads_512_projection<false>(parameters.words, nullptr, nullptr);
}
// The large-input entry doubles theoretical block residency on RTX 5080
// without spilling. Small inputs retain the original instruction schedule.
extern "C" __global__ __maxnreg__(128) void
phoenix_global_attention_split_heads_512_projection_large_input(KernelArguments parameters) {
    run_global_attention_split_heads_512_projection<false>(parameters.words, nullptr, nullptr);
}
