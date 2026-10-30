#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: global_attention/split_heads_512/feed_forward_input_view_tile_sync.cu
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
    parameter_publication_low_12 = 12,
    parameter_publication_high_13 = 13,
};

struct alignas(8) KernelArguments {
    uint32_t words[14];
};
static_assert(sizeof(KernelArguments) == 56, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void
run_global_attention_split_heads_512_feed_forward_input_view_tile_sync(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t index_word_3 = 0;
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
    uint32_t matrix_accumulator_52 = 0;
    uint32_t matrix_accumulator_53 = 0;
    uint32_t matrix_accumulator_54 = 0;
    uint32_t matrix_accumulator_55 = 0;
    uint32_t matrix_accumulator_56 = 0;
    uint32_t matrix_accumulator_57 = 0;
    uint32_t matrix_accumulator_58 = 0;
    uint32_t matrix_accumulator_59 = 0;
    uint32_t matrix_accumulator_60 = 0;
    uint32_t matrix_accumulator_61 = 0;
    uint32_t matrix_accumulator_62 = 0;
    uint32_t matrix_accumulator_63 = 0;
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
    uint32_t packed_half_144 = 0;
    uint32_t packed_half_145 = 0;
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
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t matrix_accumulator_158 = 0;
    uint32_t matrix_accumulator_159 = 0;
    uint32_t matrix_accumulator_160 = 0;
    uint32_t matrix_accumulator_161 = 0;
    uint32_t matrix_accumulator_162 = 0;
    uint32_t matrix_accumulator_163 = 0;
    uint32_t matrix_accumulator_164 = 0;
    uint32_t matrix_accumulator_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_block_index_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    packed_half_0 = threadIdx.x;

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_4];
    index_word_3 = parameters[parameter_trained_weights_high_5];

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    reused_word_1 = reused_word_1 + (0u - 0x60u);

    matrix_accumulator_138 = threadIdx.y;

    matrix_fragment_6 = blockIdx.z;

    matrix_accumulator_143 = parameters[parameter_tensor_width_7];

    matrix_fragment_29 = threadIdx.x & 31u;

    condition_0 = (matrix_accumulator_138 | packed_half_0) != 0;

    matrix_accumulator_135 = matrix_fragment_6 * 0x4u + matrix_accumulator_138;

    matrix_accumulator_135 = matrix_accumulator_135 * 0x40u;

    matrix_accumulator_135 = matrix_accumulator_135 * 0x8u;

    if (!condition_0) {
        matrix_fragment_4 = blockDim.x;

        matrix_fragment_7 = 0u;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    uniform_condition_1 = uniform_condition_0;

    if (!condition_0) {
        matrix_fragment_5 = blockDim.y;

        matrix_fragment_5 = matrix_fragment_4 * matrix_fragment_5;

        matrix_fragment_4 = 0x400u;
    }

    uniform_index_word_4 =
        broadcast_warp_word(matrix_fragment_5, !condition_0, uniform_index_word_4);

    if (!condition_0) {
        matrix_fragment_4 = matrix_fragment_4 + 0x2000u;
    }

    matrix_fragment_5 = matrix_accumulator_135 + 0x80u;

    if (!condition_0) {
        matrix_fragment_7 = (matrix_fragment_7 << 0x18u) + matrix_fragment_4;
    }

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, index_word_3));

    uniform_block_index_6 =
        broadcast_warp_word(matrix_fragment_7, !condition_0, uniform_block_index_6);

    split_address_words(matrix_accumulator_2, index_word_3,
                        uint64_t(uint32_t(matrix_accumulator_135)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, index_word_3));

    if (!uniform_condition_0) {
        uniform_index_word_4 = 0u - uniform_index_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_index_word_4 << (0xbu & 31u);

        uniform_index_word_4 = uniform_index_word_4 << (0x1u & 31u);
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x200u);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x400u);

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, index_word_3));

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_block_index_6);
#endif
    }

    split_address_words(matrix_fragment_22, matrix_fragment_23,
                        int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    split_address_words(matrix_fragment_28, matrix_fragment_29,
                        int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_block_index_6 + 0x8u);
#endif
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_20, matrix_fragment_21),
                                        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
                                        matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x4000ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_22, matrix_fragment_23),
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x4000ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_28, matrix_fragment_29),
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_28, matrix_fragment_29) + 0x4000ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    condition_0 = uint32_t(matrix_accumulator_143) != 0u;

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                        matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_16, matrix_fragment_17) + 0x4000ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    if (condition_0) {
        index_word_3 = matrix_accumulator_143 + 0xffffffffu;

        matrix_fragment_16 = uint32_t(int32_t(index_word_3) >> 0x1fu);
    }

    uniform_block_index_6 = blockIdx.y;

    if (condition_0) {
        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_16, 0u) << 0x3u >> 32) + index_word_3;
    }

    matrix_fragment_18 = 0u;

    if (condition_0) {
        matrix_fragment_18 =
            uint32_t(int64_t(int32_t(matrix_fragment_16)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_2 = blockIdx.x;

    matrix_fragment_19 = matrix_accumulator_138 * 0x20u + packed_half_0;

    thread_scratch[22] = matrix_fragment_18;

    condition_1 = uint32_t(matrix_fragment_19) > uint32_t(0x3ffu);

    uniform_address_word_8 = uniform_block_index_6 << (0x3u & 31u);
    if (!condition_1) {
        matrix_fragment_28 = parameters[parameter_tensor_height_6];

        matrix_fragment_16 = matrix_fragment_19 & 0x70u;

        matrix_fragment_23 = packed_half_0 * 0x8u;

        matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_19) >> 0x6u);

        matrix_fragment_21 = 0u;

        matrix_fragment_16 = uint32_t(uint32_t(matrix_fragment_16) >> 0x4u);

        matrix_fragment_17 = matrix_fragment_17 & 0x4u;

        matrix_fragment_31 = 0u;

        index_word_3 = uint32_t(uint32_t(matrix_fragment_19) >> 0x7u);

        matrix_fragment_17 = matrix_fragment_17 | 0x3u & matrix_fragment_16;

        index_word_3 = index_word_3 & 0x4u;

        matrix_fragment_16 = matrix_fragment_16 | 0x8u & matrix_fragment_23;

        matrix_fragment_18 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_29 = matrix_fragment_19 & 0x3ffu;

        index_word_3 =
            uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1eu >> 32) + index_word_3;

        condition_0 = int32_t(matrix_fragment_18) >= int32_t(matrix_accumulator_143);

        matrix_fragment_16 = 0x400u;

        condition_1 = uint32_t(matrix_fragment_28) != uint32_t(0x1u);

        index_word_3 = index_word_3 + uniform_address_word_8;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        condition_2 = int32_t(index_word_3) >= int32_t(matrix_fragment_28) && condition_1;

        matrix_fragment_22 = condition_1 ? index_word_3 : 0u;

        condition_0 = condition_2 | condition_0;

        matrix_fragment_16 = (matrix_fragment_21 << 0x18u) + matrix_fragment_16;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        matrix_fragment_17 = matrix_fragment_31;

        if (!condition_2) {
            matrix_fragment_18 = condition_3 ? matrix_fragment_18 : 0u;
        }
        if (!condition_0) {
            matrix_fragment_23 = matrix_fragment_23 & 0x10u;

            index_word_3 = uint32_t(uint32_t(matrix_fragment_19) >> 0x2u);

            matrix_fragment_20 = matrix_fragment_23 | 0xcu & packed_half_0;

            index_word_3 = index_word_3 & 0x20u;

            index_word_3 = index_word_3 + matrix_fragment_20;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

            matrix_fragment_22 = index_word_3 * matrix_fragment_28 + matrix_fragment_22;

            index_word_3 = packed_half_0 & 0xcu;

            matrix_fragment_22 = matrix_fragment_22 * matrix_accumulator_143;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(matrix_fragment_29)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x37fu);
    if (!condition_0) {
        matrix_fragment_29 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 + 0x80u;

        matrix_fragment_17 = matrix_fragment_19 & 0x7fu;

        matrix_fragment_28 = packed_half_0 * 0x8u;

        matrix_fragment_23 = 0u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_18 = uint32_t(uint32_t(index_word_3) >> 0x6u);

        matrix_fragment_16 = uint32_t(uint32_t(index_word_3) >> 0x7u);

        matrix_fragment_30 = 0u;

        matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_17) >> 0x4u);

        matrix_fragment_18 = matrix_fragment_18 & 0x4u;

        matrix_fragment_21 = matrix_fragment_16 & 0xcu;

        matrix_fragment_16 = matrix_fragment_17 | 0x8u & matrix_fragment_28;

        matrix_fragment_17 = matrix_fragment_18 | 0x3u & matrix_fragment_17;

        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1eu >> 32) + matrix_fragment_21;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        matrix_fragment_22 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        condition_2 = uint32_t(matrix_fragment_29) != uint32_t(0x1u);

        matrix_fragment_18 = matrix_fragment_16 + uniform_address_word_8;

        matrix_fragment_16 = 0x400u;

        condition_0 = int32_t(matrix_fragment_22) >= int32_t(matrix_accumulator_143);

        condition_1 = int32_t(matrix_fragment_18) >= int32_t(matrix_fragment_29) && condition_2;

        matrix_fragment_18 = condition_2 ? matrix_fragment_18 : 0u;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        matrix_fragment_17 = (matrix_fragment_23 << 0x18u) + matrix_fragment_16;

        condition_0 = condition_1 | condition_0;

        matrix_fragment_16 = index_word_3 & 0x780u;

        if (!condition_1) {
            matrix_fragment_22 = condition_3 ? matrix_fragment_22 : 0u;
        }

        matrix_fragment_23 = matrix_fragment_16 | 0x7fu & matrix_fragment_19;

        matrix_fragment_16 = matrix_fragment_17;

        matrix_fragment_17 = matrix_fragment_30;
        if (!condition_0) {
            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            matrix_fragment_21 = matrix_fragment_28 & 0x10u;

            index_word_3 = index_word_3 & 0x20u;

            matrix_fragment_20 = matrix_fragment_21 | 0xcu & packed_half_0;

            index_word_3 = matrix_fragment_20 + index_word_3;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

            matrix_fragment_18 = index_word_3 * matrix_fragment_29 + matrix_fragment_18;

            index_word_3 = packed_half_0 & 0xcu;

            matrix_fragment_18 = matrix_fragment_18 * matrix_accumulator_143;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(matrix_fragment_23)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x2ffu);
    if (!condition_0) {
        matrix_fragment_29 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 + 0x100u;

        matrix_fragment_18 = matrix_fragment_19 & 0x7fu;

        matrix_fragment_23 = packed_half_0 * 0x8u;

        matrix_fragment_20 = 0u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_17 = uint32_t(uint32_t(index_word_3) >> 0x6u);

        matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_18) >> 0x4u);

        matrix_fragment_30 = 0u;

        matrix_fragment_17 = matrix_fragment_17 & 0x4u;

        matrix_fragment_16 = uint32_t(uint32_t(index_word_3) >> 0x7u);

        matrix_fragment_17 = matrix_fragment_17 | 0x3u & matrix_fragment_18;

        matrix_fragment_16 = matrix_fragment_16 & 0xcu;

        matrix_fragment_21 = matrix_fragment_18 | 0x8u & matrix_fragment_23;

        matrix_fragment_28 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        matrix_fragment_17 = 0x400u;

        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_21, 0u) << 0x1eu >> 32) + matrix_fragment_16;

        condition_2 = uint32_t(matrix_fragment_29) != uint32_t(0x1u);

        condition_0 = int32_t(matrix_fragment_28) >= int32_t(matrix_accumulator_143);

        matrix_fragment_22 = matrix_fragment_16 + uniform_address_word_8;

        matrix_fragment_16 = index_word_3 & 0x700u;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        condition_1 = int32_t(matrix_fragment_22) >= int32_t(matrix_fragment_29) && condition_2;

        matrix_fragment_18 = matrix_fragment_16 | 0x80u & matrix_fragment_19;

        matrix_fragment_22 = condition_2 ? matrix_fragment_22 : 0u;

        matrix_fragment_17 = (matrix_fragment_20 << 0x18u) + matrix_fragment_17;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        condition_0 = condition_1 | condition_0;

        matrix_fragment_16 = matrix_fragment_17;

        matrix_fragment_17 = matrix_fragment_30;

        index_word_3 = matrix_fragment_18 | 0x7fu & matrix_fragment_19;

        if (!condition_1) {
            matrix_fragment_28 = condition_3 ? matrix_fragment_28 : 0u;
        }
        if (!condition_0) {
            matrix_fragment_23 = matrix_fragment_23 & 0x10u;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_19) >> 0x2u);

            matrix_fragment_23 = matrix_fragment_23 | 0xcu & packed_half_0;

            matrix_fragment_18 = matrix_fragment_18 & 0x20u;

            matrix_fragment_18 = matrix_fragment_18 + matrix_fragment_23;

            matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_18) >> 0x4u);

            matrix_fragment_22 = matrix_fragment_18 * matrix_fragment_29 + matrix_fragment_22;

            matrix_fragment_18 = packed_half_0 & 0xcu;

            matrix_fragment_22 = matrix_fragment_22 * matrix_accumulator_143;

            matrix_fragment_23 = uint32_t(uint32_t(matrix_fragment_18) >> 0x2u);

            matrix_fragment_23 = matrix_fragment_22 * 0x4u + matrix_fragment_23;

            matrix_fragment_23 = matrix_fragment_28 * 0x4u + matrix_fragment_23;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(index_word_3)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x27fu);
    if (!condition_0) {
        matrix_fragment_31 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 + 0x180u;

        matrix_fragment_18 = matrix_fragment_19 & 0x7fu;

        matrix_fragment_29 = packed_half_0 * 0x8u;

        matrix_fragment_23 = 0u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_17 = uint32_t(uint32_t(index_word_3) >> 0x6u);

        matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_18) >> 0x4u);

        matrix_fragment_28 = 0u;

        matrix_fragment_17 = matrix_fragment_17 & 0x4u;

        matrix_fragment_16 = uint32_t(uint32_t(index_word_3) >> 0x7u);

        matrix_fragment_17 = matrix_fragment_17 | 0x3u & matrix_fragment_18;

        matrix_fragment_16 = matrix_fragment_16 & 0xcu;

        matrix_fragment_21 = matrix_fragment_18 | 0x8u & matrix_fragment_29;

        matrix_fragment_22 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_21, 0u) << 0x1eu >> 32) + matrix_fragment_16;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        condition_2 = uint32_t(matrix_fragment_31) != uint32_t(0x1u);

        condition_0 = int32_t(matrix_fragment_22) >= int32_t(matrix_accumulator_143);

        matrix_fragment_18 = matrix_fragment_16 + uniform_address_word_8;

        matrix_fragment_16 = 0x400u;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        condition_1 = int32_t(matrix_fragment_18) >= int32_t(matrix_fragment_31) && condition_2;

        matrix_fragment_18 = condition_2 ? matrix_fragment_18 : 0u;

        matrix_fragment_17 = (matrix_fragment_23 << 0x18u) + matrix_fragment_16;

        condition_0 = condition_1 | condition_0;

        matrix_fragment_16 = index_word_3 & 0x780u;

        matrix_fragment_23 = matrix_fragment_16 | 0x7fu & matrix_fragment_19;

        matrix_fragment_16 = matrix_fragment_17;

        matrix_fragment_17 = matrix_fragment_28;

        if (!condition_1) {
            matrix_fragment_22 = condition_3 ? matrix_fragment_22 : 0u;
        }
        if (!condition_0) {
            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            matrix_fragment_29 = matrix_fragment_29 & 0x10u;

            index_word_3 = index_word_3 & 0x20u;

            matrix_fragment_20 = matrix_fragment_29 | 0xcu & packed_half_0;

            index_word_3 = matrix_fragment_20 + index_word_3;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

            matrix_fragment_18 = index_word_3 * matrix_fragment_31 + matrix_fragment_18;

            index_word_3 = packed_half_0 & 0xcu;

            matrix_fragment_18 = matrix_fragment_18 * matrix_accumulator_143;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(matrix_fragment_23)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x1ffu);
    if (!condition_0) {
        matrix_fragment_31 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 & 0x70u;

        matrix_fragment_23 = packed_half_0 * 0x8u;

        matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_19) >> 0x6u);

        matrix_fragment_20 = 0u;

        matrix_fragment_16 = uint32_t(uint32_t(index_word_3) >> 0x4u);

        index_word_3 = matrix_fragment_17 & 0x3fcu;

        matrix_fragment_29 = 0u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        index_word_3 = index_word_3 | 0x3u & matrix_fragment_16;

        matrix_fragment_16 = matrix_fragment_16 | 0x8u & matrix_fragment_23;

        matrix_fragment_22 = matrix_accumulator_2 * 0x8u + index_word_3;

        index_word_3 =
            uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1eu >> 32) + uniform_address_word_8;

        condition_0 = int32_t(matrix_fragment_22) >= int32_t(matrix_accumulator_143);

        matrix_fragment_18 = index_word_3 + 0x4u;

        index_word_3 = 0x400u;

        condition_1 = uint32_t(matrix_fragment_31) != uint32_t(0x1u);

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        condition_2 = int32_t(matrix_fragment_18) >= int32_t(matrix_fragment_31) && condition_1;

        matrix_fragment_18 = condition_1 ? matrix_fragment_18 : 0u;

        condition_0 = condition_2 | condition_0;

        matrix_fragment_16 = (matrix_fragment_20 << 0x18u) + index_word_3;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        index_word_3 = matrix_fragment_19 + 0x200u;

        matrix_fragment_17 = matrix_fragment_29;

        if (!condition_2) {
            matrix_fragment_22 = condition_3 ? matrix_fragment_22 : 0u;
        }
        if (!condition_0) {
            matrix_fragment_23 = matrix_fragment_23 & 0x10u;

            matrix_fragment_20 = uint32_t(uint32_t(matrix_fragment_19) >> 0x2u);

            matrix_fragment_23 = matrix_fragment_23 | 0xcu & packed_half_0;

            matrix_fragment_20 = matrix_fragment_20 & 0x20u;

            matrix_fragment_23 = matrix_fragment_20 + matrix_fragment_23;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            matrix_fragment_23 = uint32_t(uint32_t(matrix_fragment_23) >> 0x4u);

            matrix_fragment_18 = matrix_fragment_23 * matrix_fragment_31 + matrix_fragment_18;

            matrix_fragment_23 = packed_half_0 & 0xcu;

            matrix_fragment_18 = matrix_fragment_18 * matrix_accumulator_143;

            matrix_fragment_23 = uint32_t(uint32_t(matrix_fragment_23) >> 0x2u);

            matrix_fragment_23 = matrix_fragment_18 * 0x4u + matrix_fragment_23;

            matrix_fragment_23 = matrix_fragment_22 * 0x4u + matrix_fragment_23;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(index_word_3)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x17fu);
    if (!condition_0) {
        matrix_fragment_23 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 + 0x280u;

        matrix_fragment_20 = 0u;

        matrix_fragment_17 = matrix_fragment_19 & 0x7fu;

        matrix_fragment_28 = packed_half_0 * 0x8u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_16 = uint32_t(uint32_t(index_word_3) >> 0x6u);

        matrix_fragment_30 = 0u;

        matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_17) >> 0x4u);

        matrix_fragment_16 = matrix_fragment_16 & 0x4u;

        matrix_fragment_18 = matrix_fragment_17 | 0x8u & matrix_fragment_28;

        matrix_fragment_17 = matrix_fragment_16 | 0x3u & matrix_fragment_17;

        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_18, 0u) << 0x1eu >> 32) + uniform_address_word_8;

        matrix_fragment_22 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        matrix_fragment_17 = 0x400u;

        matrix_fragment_18 = matrix_fragment_16 + 0x4u;

        matrix_fragment_16 = index_word_3 & 0x180u;

        condition_2 = uint32_t(matrix_fragment_23) != uint32_t(0x1u);

        condition_0 = int32_t(matrix_fragment_22) >= int32_t(matrix_accumulator_143);

        matrix_fragment_21 = matrix_fragment_16 | 0x7fu & matrix_fragment_19;

        condition_1 = int32_t(matrix_fragment_18) >= int32_t(matrix_fragment_23) && condition_2;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        matrix_fragment_18 = condition_2 ? matrix_fragment_18 : 0u;

        matrix_fragment_17 = (matrix_fragment_20 << 0x18u) + matrix_fragment_17;

        condition_0 = condition_1 | condition_0;

        matrix_fragment_16 = matrix_fragment_17;

        matrix_fragment_17 = matrix_fragment_30;

        if (!condition_1) {
            matrix_fragment_22 = condition_3 ? matrix_fragment_22 : 0u;
        }

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(matrix_fragment_21)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);
        if (!condition_0) {
            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            matrix_fragment_21 = matrix_fragment_28 & 0x10u;

            index_word_3 = index_word_3 & 0x20u;

            matrix_fragment_20 = matrix_fragment_21 | 0xcu & packed_half_0;

            index_word_3 = matrix_fragment_20 + index_word_3;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

            matrix_fragment_18 = index_word_3 * matrix_fragment_23 + matrix_fragment_18;

            index_word_3 = packed_half_0 & 0xcu;

            matrix_fragment_18 = matrix_fragment_18 * matrix_accumulator_143;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x800u);
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0xffu);
    if (!condition_0) {
        matrix_fragment_28 = parameters[parameter_tensor_height_6];

        index_word_3 = matrix_fragment_19 & 0x70u;

        matrix_fragment_22 = packed_half_0 * 0x8u;

        uniform_index_word_4 = 0x400u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        uniform_index_word_5 = 0u;

        matrix_fragment_17 = index_word_3 & 0x3u;

        matrix_fragment_16 = index_word_3 | 0x8u & matrix_fragment_22;

        index_word_3 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        matrix_fragment_16 =
            uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1eu >> 32) + uniform_address_word_8;

        uniform_index_word_7 = 0u;

        matrix_fragment_17 = 0u;

        matrix_fragment_18 = index_word_3 + 0x4u;

        index_word_3 = matrix_fragment_16 + 0x4u;

        matrix_fragment_16 = matrix_fragment_19 * 0x4u;

        condition_1 = uint32_t(matrix_fragment_28) != uint32_t(0x1u);

        condition_0 = int32_t(matrix_fragment_18) >= int32_t(matrix_accumulator_143);

        matrix_fragment_16 = matrix_fragment_16 & 0x3fcu;

        condition_2 = int32_t(index_word_3) >= int32_t(matrix_fragment_28) && condition_1;

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        matrix_fragment_23 = condition_1 ? index_word_3 : 0u;

        uniform_index_word_4 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

        condition_0 = condition_2 | condition_0;

        uniform_index_word_5 = uniform_index_word_7;

        if (!condition_2) {
            matrix_fragment_18 = condition_3 ? matrix_fragment_18 : 0u;
        }

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) +
                                combine_words(uniform_index_word_4, uniform_index_word_5));
        if (!condition_0) {
            index_word_3 = uint32_t(uint32_t(matrix_fragment_19) >> 0x2u);

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            index_word_3 = index_word_3 & 0x20u;

            index_word_3 = index_word_3 | 0x10u & matrix_fragment_22;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

            matrix_fragment_22 = index_word_3 * matrix_fragment_28 + matrix_fragment_23;

            index_word_3 = packed_half_0 & 0xcu;

            matrix_fragment_22 = matrix_fragment_22 * matrix_accumulator_143;

            index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0xc00u);
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x7fu);
    if (!condition_0) {
        matrix_fragment_28 = parameters[parameter_tensor_height_6];

        matrix_fragment_16 = packed_half_0 * 0x8u;

        index_word_3 = uint32_t(uint32_t(matrix_fragment_19) >> 0x4u);

        matrix_fragment_21 = 0u;

        condition_3 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u);

        matrix_fragment_16 = matrix_fragment_16 & 0x8u;

        matrix_fragment_23 = 0u;

        matrix_fragment_17 = index_word_3 & 0x3u;

        index_word_3 = matrix_fragment_16 + index_word_3;

        matrix_fragment_16 = 0x400u;

        matrix_fragment_17 = matrix_accumulator_2 * 0x8u + matrix_fragment_17;

        index_word_3 =
            uint32_t(combine_words(index_word_3, 0u) << 0x1eu >> 32) + uniform_address_word_8;

        matrix_fragment_18 = matrix_fragment_17 + 0x4u;

        index_word_3 = index_word_3 + 0x4u;

        condition_0 = int32_t(matrix_fragment_18) >= int32_t(matrix_accumulator_143);

        condition_1 = uint32_t(matrix_fragment_28) != uint32_t(0x1u);

        condition_0 = uint32_t(matrix_accumulator_143) != uint32_t(0x1u) && condition_0;

        condition_2 = int32_t(index_word_3) >= int32_t(matrix_fragment_28) && condition_1;

        index_word_3 = condition_1 ? index_word_3 : 0u;

        condition_0 = condition_2 | condition_0;

        matrix_fragment_16 = (matrix_fragment_21 << 0x18u) + matrix_fragment_16;

        split_address_words(matrix_fragment_20, matrix_fragment_21, 0ull);

        matrix_fragment_17 = matrix_fragment_23;

        if (!condition_2) {
            matrix_fragment_18 = condition_3 ? matrix_fragment_18 : 0u;
        }

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
        if (!condition_0) {
            matrix_fragment_20 = uint32_t(uint32_t(packed_half_0) >> 0x1u);

            matrix_fragment_22 = matrix_fragment_20 & 0x1u;

            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];

            matrix_fragment_22 = 0x2u | matrix_fragment_22;

            matrix_fragment_22 = matrix_fragment_22 * matrix_fragment_28 + index_word_3;

            index_word_3 = uint32_t(uint32_t(packed_half_0) >> 0x2u);

            matrix_fragment_22 = matrix_fragment_22 * matrix_accumulator_143;

            index_word_3 = index_word_3 & 0x3u;

            index_word_3 = matrix_fragment_22 * 0x4u + index_word_3;

            index_word_3 = matrix_fragment_18 * 0x4u + index_word_3;

            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0xe00u);
        if (condition_0) {
            shared_write<CheckBounds>(shared_tile, matrix_fragment_16, 0u, shared_bytes,
                                      error_counts);
        } else {
            matrix_fragment_17 = matrix_fragment_16;

            {
                const uint32_t operand_a = matrix_fragment_17;
                const uint64_t global_address =
                    combine_words(matrix_fragment_20, matrix_fragment_21);
                bool valid = true;
                if constexpr (CheckBounds) {
                    valid = false;
                    for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                        valid |= global_address >= memory_ranges[byte_offset].base &&
                                 global_address - memory_ranges[byte_offset].base <=
                                     memory_ranges[byte_offset].size &&
                                 4 <= memory_ranges[byte_offset].size -
                                          (global_address - memory_ranges[byte_offset].base);
                    if (!valid)
                        atomicAdd(error_counts, 1u);
                }
                if constexpr (CheckBounds) {
                    if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                        atomicAdd(error_counts, 1u);
                        valid = false;
                    }
                }
                if (valid) {
                    uint32_t shared_words =
                        uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                 "l"(global_address)
                                 : "memory");
#else
                    phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                }
            }
        }
    }
    uniform_index_word_7 = 0u;
#if PHOENIX_STYLE_ARCH >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif

    uniform_index_word_4 = 0x400u;

    index_word_3 = matrix_fragment_19 & 0x7fu;

    uniform_index_word_5 = uniform_index_word_4 + 0x2000u;

    matrix_fragment_18 = packed_half_0 * 0x8u;

    matrix_fragment_23 = uint32_t(uint32_t(matrix_fragment_19) >> 0x4u);

    index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x4u);

    matrix_fragment_29 = matrix_fragment_23 & 0x3u;

    matrix_fragment_20 = matrix_fragment_18 & 0x8u;

    matrix_fragment_21 = index_word_3 & 0x3u;

    matrix_fragment_84 = matrix_accumulator_2 * 0x8u + matrix_fragment_29;

    matrix_fragment_22 = matrix_fragment_19 + 0x80u;

    packed_half_145 = matrix_fragment_19 + 0x180u;

    packed_half_144 = matrix_fragment_19 + 0x280u;

    matrix_fragment_82 = matrix_fragment_20 + matrix_fragment_23;

    matrix_accumulator_2 = matrix_accumulator_2 * 0x8u + matrix_fragment_21;

    matrix_accumulator_137 = uint32_t(uint32_t(matrix_fragment_19) >> 0x2u);

    matrix_fragment_21 = uint32_t(uint32_t(matrix_fragment_22) >> 0x2u);

    uniform_index_word_5 = (uniform_index_word_7 << 0x18u) + uniform_index_word_5;
#if PHOENIX_STYLE_ARCH >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif

    matrix_fragment_23 = uint32_t(uint32_t(packed_half_145) >> 0x2u);

    matrix_fragment_28 = uint32_t(uint32_t(packed_half_144) >> 0x2u);

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = uniform_index_word_5;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_fragment_16 = 0u;
#endif

    {
        matrix_fragment_20 = matrix_accumulator_137 & 0x20u;
    }

    matrix_accumulator_137 = matrix_accumulator_137 & 0x3fe0u;

    matrix_fragment_21 = matrix_fragment_21 & 0x20u;

    matrix_fragment_23 = matrix_fragment_23 & 0x20u;

    matrix_accumulator_139 = matrix_fragment_28 & 0x20u;

    index_word_3 = index_word_3 | 0x8u & matrix_fragment_18;

    matrix_fragment_20 = matrix_fragment_20 | 0x10u & matrix_fragment_18;

    matrix_accumulator_137 = matrix_accumulator_137 | 0x10u & matrix_fragment_18;

    matrix_accumulator_141 = matrix_fragment_21 | 0x10u & matrix_fragment_18;

    matrix_accumulator_140 = matrix_fragment_23 | 0x10u & matrix_fragment_18;

    matrix_accumulator_139 = matrix_accumulator_139 | 0x10u & matrix_fragment_18;

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_0 = wait_copy_barrier(shared_tile, uniform_index_word_5,
                                    async_tokens[(((uniform_index_word_5) >> 3) & 1)]);
#else
    __syncthreads();
    condition_0 = true;
#endif
    if (!condition_0) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_0 = wait_copy_barrier(shared_tile, uniform_index_word_5,
                                            async_tokens[(((uniform_index_word_5) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
        } while (!condition_0);
    }
    matrix_accumulator_146 = matrix_fragment_19 + 0x100u;

    matrix_fragment_16 = uint32_t(uint32_t(matrix_fragment_19) >> 0x6u);

    matrix_accumulator_143 = matrix_accumulator_143 * 0x4u;

    matrix_accumulator_131 = uint32_t(uint32_t(matrix_fragment_22) >> 0x7u);

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_22) >> 0x6u);

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_136 = matrix_fragment_22 & 0x780u;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_fragment_18 = uint32_t(uint32_t(matrix_accumulator_146) >> 0x6u);

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    index_word_3 = uint32_t(uint32_t(index_word_3) >> 0x2u);

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_133 = uint32_t(uint32_t(matrix_fragment_19) >> 0x7u);

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_fragment_23 = matrix_fragment_16 & 0x4u;

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_125 = matrix_fragment_16 & 0x3fcu;

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_129 = uint32_t(uint32_t(matrix_accumulator_146) >> 0x7u);

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_accumulator_127 = uint32_t(uint32_t(packed_half_145) >> 0x7u);

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_22 = uint32_t(uint32_t(packed_half_145) >> 0x6u);

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_fragment_80 = uint32_t(uint32_t(packed_half_144) >> 0x6u);

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_16 = uint32_t(uint32_t(packed_half_144) >> 0x7u);

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

    matrix_fragment_83 = matrix_fragment_18 & 0x4u;

    matrix_accumulator_38 = 0;
    matrix_accumulator_39 = 0;

    matrix_accumulator_133 = matrix_accumulator_133 & 0x1fcu;

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    matrix_accumulator_131 = matrix_accumulator_131 & 0x3fcu;

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    matrix_fragment_81 = matrix_fragment_17 & 0x4u;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_accumulator_129 = matrix_accumulator_129 & 0x3fcu;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_127 = matrix_accumulator_127 & 0x3fcu;

    matrix_accumulator_48 = 0;
    matrix_accumulator_49 = 0;

    matrix_fragment_85 = matrix_fragment_22 & 0x4u;

    matrix_accumulator_50 = 0;
    matrix_accumulator_51 = 0;

    matrix_fragment_87 = matrix_fragment_80 & 0x4u;

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    matrix_fragment_18 = index_word_3 + uniform_address_word_8;

    matrix_fragment_82 =
        uint32_t(combine_words(matrix_fragment_82, 0u) << 0x1eu >> 32) + uniform_address_word_8;

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_accumulator_124 = index_word_3 | 0x3fcu & matrix_fragment_16;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_142 = packed_half_0 & 0xcu;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_132 = matrix_fragment_23 + matrix_accumulator_2;

    matrix_accumulator_125 = matrix_accumulator_125 + matrix_accumulator_2;

    matrix_accumulator_130 = matrix_fragment_81 + matrix_accumulator_2;

    matrix_accumulator_128 = matrix_fragment_83 + matrix_accumulator_2;

    matrix_accumulator_126 = matrix_fragment_85 + matrix_accumulator_2;

    index_word_3 = matrix_fragment_87 + matrix_accumulator_2;

    matrix_accumulator_133 = matrix_accumulator_133 + matrix_fragment_18;

    matrix_accumulator_131 = matrix_accumulator_131 + matrix_fragment_18;

    matrix_accumulator_129 = matrix_accumulator_129 + matrix_fragment_18;

    matrix_accumulator_127 = matrix_accumulator_127 + matrix_fragment_18;

    matrix_fragment_21 = 0u;

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_142 = uint32_t(uint32_t(matrix_accumulator_142) >> 0x2u);

    uniform_index_word_7 = (uniform_index_word_7 << 0x18u) + uniform_index_word_4;

    matrix_accumulator_136 = matrix_accumulator_136 | 0x7fu & matrix_fragment_19;

    matrix_fragment_17 = matrix_fragment_84 + 0x4u;

    matrix_fragment_16 = matrix_fragment_82 + 0x4u;

    matrix_accumulator_2 = matrix_accumulator_2 + 0x4u;

    matrix_accumulator_124 = matrix_accumulator_124 + uniform_address_word_8;

    matrix_fragment_18 = matrix_fragment_18 + 0x4u;
    do {
        uniform_address_word_8 = 0u;

        matrix_fragment_80 = matrix_fragment_21 * 0x40u;

        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x3ffu);

        matrix_fragment_83 = 0u;

        matrix_fragment_22 = matrix_fragment_21 + 0x40u;

        matrix_fragment_80 = matrix_fragment_80 & 0x1000u;

        condition_2 = uint32_t(matrix_fragment_21) >= uint32_t(0x180u);

        matrix_accumulator_134 = uint32_t(uint32_t(matrix_fragment_21) >> 0x6u);

        matrix_fragment_82 = 0x1000u ^ matrix_fragment_80;

        matrix_fragment_23 = uint32_t(uint32_t(matrix_fragment_22) >> 0x4u);

        matrix_fragment_21 = matrix_fragment_22;

        uniform_index_word_4 = uniform_index_word_7;

        uniform_index_word_5 = uniform_address_word_8;

        split_address_words(matrix_fragment_82, matrix_fragment_83,
                            combine_words(matrix_fragment_82, matrix_fragment_83) +
                                combine_words(uniform_index_word_4, uniform_index_word_5));
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 =
                uint32_t(combine_words(matrix_fragment_20, 0u) << 0x1cu >> 32) + matrix_fragment_23;

            matrix_fragment_87 = matrix_fragment_19 & 0x3ffu;

            condition_3 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 =
                int32_t(matrix_accumulator_133) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_132) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_0;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;

            matrix_fragment_93 = !condition_3 ? matrix_accumulator_133 : 0u;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_81 * matrix_fragment_92 + matrix_fragment_93;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_132 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x37fu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 =
                uint32_t(combine_words(matrix_accumulator_141, 0u) << 0x1cu >> 32) +
                matrix_fragment_23;

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_accumulator_136)) *
                                        uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 =
                int32_t(matrix_accumulator_131) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_130) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_0;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;

            matrix_fragment_93 = !condition_3 ? matrix_accumulator_131 : 0u;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_81 * matrix_fragment_92 + matrix_fragment_93;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_130 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x2ffu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_94 =
                uint32_t(combine_words(matrix_fragment_20, 0u) << 0x1cu >> 32) + matrix_fragment_23;

            matrix_fragment_84 = matrix_accumulator_146 & 0x700u;

            condition_3 = uint32_t(matrix_fragment_94) >= uint32_t(0x20u);

            matrix_fragment_84 = matrix_fragment_84 | 0x80u & matrix_fragment_19;

            matrix_fragment_87 = matrix_fragment_84 | 0x7fu & matrix_fragment_19;

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 =
                int32_t(matrix_accumulator_129) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_128) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_94) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_94) < uint32_t(0x20u) && condition_0;

            matrix_fragment_81 = !condition_3 ? matrix_accumulator_129 : 0u;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_94 * matrix_fragment_92 + matrix_fragment_81;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_128 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x27fu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 =
                uint32_t(combine_words(matrix_accumulator_140, 0u) << 0x1cu >> 32) +
                matrix_fragment_23;

            matrix_fragment_84 = packed_half_145 & 0x780u;

            condition_3 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            matrix_fragment_87 = matrix_fragment_84 | 0x7fu & matrix_fragment_19;

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 =
                int32_t(matrix_accumulator_127) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_126) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_0;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;

            matrix_fragment_93 = !condition_3 ? matrix_accumulator_127 : 0u;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_81 * matrix_fragment_92 + matrix_fragment_93;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_126 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x1ffu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_94 =
                uint32_t(combine_words(matrix_fragment_20, 0u) << 0x1cu >> 32) + matrix_fragment_23;

            matrix_fragment_87 = matrix_fragment_19 + 0x200u;

            condition_3 = uint32_t(matrix_fragment_94) >= uint32_t(0x20u);

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 = int32_t(matrix_fragment_18) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_125) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_94) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_94) < uint32_t(0x20u) && condition_0;

            matrix_fragment_81 = !condition_3 ? matrix_fragment_18 : 0u;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_94 * matrix_fragment_92 + matrix_fragment_81;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_125 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x17fu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 =
                uint32_t(combine_words(matrix_accumulator_139, 0u) << 0x1cu >> 32) +
                matrix_fragment_23;

            matrix_fragment_84 = packed_half_144 & 0x780u;

            condition_3 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            matrix_fragment_87 = matrix_fragment_84 | 0x7fu & matrix_fragment_19;

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 =
                int32_t(matrix_accumulator_124) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 = int32_t(index_word_3) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_0;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;

            matrix_fragment_93 = !condition_3 ? matrix_accumulator_124 : 0u;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_81 * matrix_fragment_92 + matrix_fragment_93;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = index_word_3 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0xffu);
        if (!condition_0) {
            matrix_fragment_92 = parameters[parameter_tensor_height_6];
            matrix_fragment_93 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 =
                uint32_t(combine_words(matrix_accumulator_137, 0u) << 0x1cu >> 32) +
                matrix_fragment_23;

            matrix_fragment_84 = matrix_fragment_19 * 0x4u;

            matrix_fragment_85 = 0u;

            condition_3 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            matrix_fragment_84 = matrix_fragment_84 & 0x3fcu;

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                combine_words(matrix_fragment_82, matrix_fragment_83) +
                                    combine_words(matrix_fragment_84, matrix_fragment_85));

            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                combine_words(matrix_fragment_86, matrix_fragment_87) + 0xc00u);

            condition_3 = uint32_t(matrix_fragment_92) != uint32_t(0x1u) && !condition_3;

            condition_1 = uint32_t(matrix_fragment_93) != uint32_t(0x1u);

            condition_0 = int32_t(matrix_fragment_18) < int32_t(matrix_fragment_92) || !condition_3;

            condition_1 =
                int32_t(matrix_accumulator_2) < int32_t(matrix_fragment_93) || !condition_1;

            condition_3 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_3);

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_0;

            condition_1 = condition_0 & condition_1;

            condition_4 = uint32_t(matrix_fragment_93) == uint32_t(0x1u) && condition_0;

            matrix_fragment_93 = !condition_3 ? matrix_fragment_18 : 0u;
            if (!condition_1) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_86, 0u, shared_bytes,
                                          error_counts);
            } else {
                if (condition_1) {
                    matrix_fragment_84 = parameters[parameter_input_low_0];
                    matrix_fragment_85 = parameters[parameter_input_high_1];

                    matrix_fragment_92 =
                        matrix_fragment_81 * matrix_fragment_92 + matrix_fragment_93;
                }

                matrix_fragment_87 = matrix_fragment_86;

                if (condition_1) {
                    matrix_fragment_81 = matrix_accumulator_2 * 0x4u;

                    matrix_fragment_94 =
                        matrix_accumulator_143 * matrix_fragment_92 + matrix_accumulator_142;
                }

                split_address_words(matrix_fragment_92, matrix_fragment_93, 0ull);

                if (condition_1) {
                    matrix_fragment_81 = !condition_4 ? matrix_fragment_81 : 0u;

                    matrix_fragment_81 = matrix_fragment_81 + matrix_fragment_94;

                    split_address_words(matrix_fragment_92, matrix_fragment_93,
                                        int64_t(int32_t(matrix_fragment_81)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_84, matrix_fragment_85));
                }

                {
                    const uint32_t operand_a = matrix_fragment_87;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_92, matrix_fragment_93);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        condition_0 = uint32_t(matrix_fragment_19) > uint32_t(0x7fu);
        if (!condition_0) {
            matrix_fragment_84 = parameters[parameter_tensor_height_6];
            matrix_fragment_85 = parameters[parameter_tensor_width_7];

            matrix_fragment_81 = uint32_t(uint32_t(packed_half_0) >> 0x1u);

            split_address_words(matrix_fragment_82, matrix_fragment_83,
                                uint64_t(uint32_t(matrix_fragment_19)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            matrix_fragment_86 = matrix_fragment_81 & 0x1u;

            matrix_fragment_81 = matrix_fragment_86 + 0x2u + matrix_fragment_23;

            split_address_words(matrix_fragment_86, matrix_fragment_87, 0ull);

            condition_0 = uint32_t(matrix_fragment_81) >= uint32_t(0x20u);

            condition_0 = uint32_t(matrix_fragment_84) != uint32_t(0x1u) && !condition_0;

            condition_3 = uint32_t(matrix_fragment_85) != uint32_t(0x1u);

            condition_1 = int32_t(matrix_fragment_16) < int32_t(matrix_fragment_84) || !condition_0;

            condition_3 = int32_t(matrix_fragment_17) < int32_t(matrix_fragment_85) || !condition_3;

            condition_0 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) != bool(condition_0);

            condition_1 = uint32_t(matrix_fragment_81) < uint32_t(0x20u) && condition_1;

            condition_3 = condition_1 & condition_3;

            condition_1 = uint32_t(matrix_fragment_85) == uint32_t(0x1u) && condition_1;

            matrix_fragment_85 = !condition_0 ? matrix_fragment_16 : 0u;
            if (condition_3) {
                matrix_fragment_86 = parameters[parameter_input_low_0];
                matrix_fragment_87 = parameters[parameter_input_high_1];

                matrix_fragment_92 = uint32_t(uint32_t(packed_half_0) >> 0x2u);

                matrix_fragment_23 = matrix_fragment_17 * 0x4u;

                matrix_fragment_84 = matrix_fragment_81 * matrix_fragment_84 + matrix_fragment_85;

                matrix_fragment_92 = matrix_fragment_92 & 0x3u;

                matrix_fragment_23 = !condition_1 ? matrix_fragment_23 : 0u;

                matrix_fragment_84 =
                    matrix_accumulator_143 * matrix_fragment_84 + matrix_fragment_92;

                matrix_fragment_23 = matrix_fragment_23 + matrix_fragment_84;

                split_address_words(matrix_fragment_86, matrix_fragment_87,
                                    int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_86, matrix_fragment_87));
            }
            split_address_words(matrix_fragment_82, matrix_fragment_83,
                                combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe00u);
            if (!condition_3) {
                shared_write<CheckBounds>(shared_tile, matrix_fragment_82, 0u, shared_bytes,
                                          error_counts);
            } else {
                matrix_fragment_83 = matrix_fragment_82;

                {
                    const uint32_t operand_a = matrix_fragment_83;
                    const uint64_t global_address =
                        combine_words(matrix_fragment_86, matrix_fragment_87);
                    bool valid = true;
                    if constexpr (CheckBounds) {
                        valid = false;
                        for (int byte_offset = 0; byte_offset < 16; ++byte_offset)
                            valid |= global_address >= memory_ranges[byte_offset].base &&
                                     global_address - memory_ranges[byte_offset].base <=
                                         memory_ranges[byte_offset].size &&
                                     4 <= memory_ranges[byte_offset].size -
                                              (global_address - memory_ranges[byte_offset].base);
                        if (!valid)
                            atomicAdd(error_counts, 1u);
                    }
                    if constexpr (CheckBounds) {
                        if (operand_a > shared_bytes - 4 || operand_a & 3 || global_address & 3) {
                            atomicAdd(error_counts, 1u);
                            valid = false;
                        }
                    }
                    if (valid) {
                        uint32_t shared_words =
                            uint32_t(__cvta_generic_to_shared(shared_tile)) + operand_a;
#if PHOENIX_STYLE_ARCH >= 800
                        asm volatile("cp.async.ca.shared.global [%0],[%1],4;" ::"r"(shared_words),
                                     "l"(global_address)
                                     : "memory");
#else
                        phoenix_architecture::copy_shared_word(shared_words, global_address);
#endif
                    }
                }
            }
        }
        matrix_accumulator_147 = threadIdx.x & 31u;

        matrix_fragment_23 = matrix_fragment_80 + uniform_index_word_7;
#if PHOENIX_STYLE_ARCH >= 800
        asm volatile("cp.async.commit_group;" ::: "memory");
#endif

        matrix_fragment_23 = matrix_accumulator_147 * 0x10u + matrix_fragment_23;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0x400u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0x200u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_90,
                                 matrix_fragment_91);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0xc00u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_60,
                                 matrix_accumulator_61);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0x800u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_32,
                                 matrix_fragment_33);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0x600u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_48,
                                 matrix_accumulator_49);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0xa00u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_36,
                                 matrix_accumulator_37);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_23 + 0xe00u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        condition_0 = (matrix_accumulator_134 & 0x1u) != 0;
#if PHOENIX_STYLE_ARCH >= 800
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif

        matrix_fragment_23 = matrix_fragment_22 * 0x80u + matrix_accumulator_135;

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_44,
                                 matrix_accumulator_45);

        matrix_fragment_82 = 0u;

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_28,
                                 matrix_fragment_29);

        matrix_fragment_92 = parameters[parameter_trained_weights_low_4];
        matrix_fragment_93 = parameters[parameter_trained_weights_high_5];

        matrix_fragment_80 = 0x400u;

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_81 = matrix_fragment_80 + 0x2000u;

        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_92, matrix_fragment_93));

        matrix_fragment_81 = (matrix_fragment_82 << 0x18u) + matrix_fragment_81;

        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_accumulator_147)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));

        if (!condition_0) {
            matrix_fragment_81 = matrix_fragment_81 + 0x8u;
        }

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x600ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_76,
                                 matrix_fragment_77);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_78,
                                 matrix_fragment_79);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_22, matrix_fragment_23),
                                            matrix_fragment_76, matrix_fragment_77,
                                            matrix_fragment_78, matrix_fragment_79, memory_ranges,
                                            error_counts);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_28,
                                 matrix_fragment_29);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x400ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x4000ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_110,
                                 matrix_accumulator_111);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x4200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x4400ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x200ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x4600ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = matrix_fragment_81;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_22, matrix_fragment_23,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_22 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_81,
                                        async_tokens[(((matrix_fragment_81) >> 3) & 1)]);
#else
        __syncthreads();
        condition_0 = true;
#endif
        if (!condition_0) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_81,
                                                async_tokens[(((matrix_fragment_81) >> 3) & 1)]);
#else
                __syncthreads();
                condition_0 = true;
#endif
            } while (!condition_0);
        }
    } while (!condition_2);
    matrix_fragment_83 = threadIdx.x & 31u;

    packed_half_0 = (matrix_fragment_82 << 0x18u) + matrix_fragment_80;

    uniform_index_word_4 = 0u;

    packed_half_0 = matrix_fragment_83 * 0x10u + packed_half_0;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1800u, matrix_fragment_80,
                                        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1000u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1400u, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1c00u, matrix_fragment_84,
                                        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1200u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_40,
                             matrix_accumulator_41);

    thread_scratch[20] = matrix_accumulator_38;
    thread_scratch[21] = matrix_accumulator_39;

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_90,
                             matrix_fragment_91);

    thread_scratch[18] = matrix_accumulator_40;
    thread_scratch[19] = matrix_accumulator_41;

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_62,
                             matrix_accumulator_63);

    thread_scratch[16] = matrix_accumulator_36;
    thread_scratch[17] = matrix_accumulator_37;

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_60,
                             matrix_accumulator_61);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1e00u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    thread_scratch[14] = matrix_accumulator_38;
    thread_scratch[15] = matrix_accumulator_39;

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_28,
                             matrix_fragment_29);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1600u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_58,
                             matrix_accumulator_59);

    thread_scratch[12] = matrix_accumulator_40;
    thread_scratch[13] = matrix_accumulator_41;

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_56,
                             matrix_accumulator_57);

    thread_scratch[10] = matrix_accumulator_36;
    thread_scratch[11] = matrix_accumulator_37;

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_54,
                             matrix_accumulator_55);

    thread_scratch[8] = matrix_accumulator_38;
    thread_scratch[9] = matrix_accumulator_39;

    multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_accumulator_38 = 0;
    matrix_accumulator_39 = 0;

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    multiply_accumulate_half(matrix_accumulator_162, matrix_accumulator_163, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_50,
                             matrix_accumulator_51);

    matrix_accumulator_50 = 0;
    matrix_accumulator_51 = 0;

    multiply_accumulate_half(matrix_accumulator_160, matrix_accumulator_161, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_48,
                             matrix_accumulator_49);

    matrix_accumulator_48 = 0;
    matrix_accumulator_49 = 0;

    multiply_accumulate_half(matrix_accumulator_158, matrix_accumulator_159, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_46,
                             matrix_accumulator_47);

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_44,
                             matrix_accumulator_45);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1a00u, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    packed_half_0 = blockIdx.z;

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_42,
                             matrix_accumulator_43);

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    thread_scratch[6] = matrix_fragment_32;
    thread_scratch[7] = matrix_fragment_33;

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_116,
                             matrix_accumulator_117);

    thread_scratch[4] = matrix_accumulator_36;
    thread_scratch[5] = matrix_accumulator_37;

    multiply_accumulate_half(matrix_accumulator_2, index_word_3, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_108,
                             matrix_accumulator_109);

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_104,
                             matrix_accumulator_105);

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    thread_scratch[2] = matrix_fragment_34;
    thread_scratch[3] = matrix_fragment_35;

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_134,
                             matrix_accumulator_135);

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    thread_scratch[0] = matrix_fragment_32;
    thread_scratch[1] = matrix_fragment_33;

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_132,
                             matrix_accumulator_133);

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_120,
                             matrix_accumulator_121);

    matrix_fragment_21 = packed_half_0 * 0x4u + matrix_accumulator_138;

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_accumulator_113 = matrix_fragment_21 * 0x20u;

    matrix_fragment_21 = matrix_fragment_21 * 0x1000u;

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        uint64_t(uint32_t(matrix_accumulator_113)) * uint64_t(uint32_t(0x200u)) +
                            combine_words(matrix_fragment_92, matrix_fragment_93));

    multiply_accumulate_half(matrix_accumulator_164, matrix_accumulator_165, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_164,
                             matrix_accumulator_165);

    split_address_words(matrix_fragment_92, matrix_fragment_93,
                        uint64_t(uint32_t(matrix_fragment_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_92, matrix_fragment_93));

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x42200u);

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    split_address_words(matrix_accumulator_114, matrix_accumulator_115,
                        combine_words(matrix_fragment_92, matrix_fragment_93) + 0x60600u);

    split_address_words(matrix_accumulator_116, matrix_accumulator_117,
                        combine_words(matrix_fragment_92, matrix_fragment_93) + 0x40200u);

    multiply_accumulate_half(matrix_accumulator_162, matrix_accumulator_163, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_162,
                             matrix_accumulator_163);

    multiply_accumulate_half(matrix_accumulator_160, matrix_accumulator_161, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_160,
                             matrix_accumulator_161);

    multiply_accumulate_half(matrix_accumulator_158, matrix_accumulator_159, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_158,
                             matrix_accumulator_159);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_136,
                             matrix_accumulator_137);

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    multiply_accumulate_half(matrix_accumulator_2, index_word_3, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_2,
                             index_word_3);

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_70,
                             matrix_fragment_71);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_64 = 0;
    matrix_fragment_65 = 0;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_66 = 0;
    matrix_fragment_67 = 0;
    do {
        matrix_fragment_76 = threadIdx.x & 31u;

        matrix_fragment_84 = thread_scratch[18];

        matrix_fragment_85 = thread_scratch[19];

        matrix_fragment_86 = thread_scratch[14];

        matrix_fragment_87 = thread_scratch[15];

        matrix_accumulator_139 = thread_scratch[20];

        matrix_accumulator_138 = thread_scratch[21];

        matrix_fragment_93 = thread_scratch[16];

        matrix_fragment_92 = thread_scratch[17];

        split_address_words(matrix_fragment_88, matrix_fragment_89,
                            int64_t(int32_t(matrix_fragment_76)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_116, matrix_accumulator_117));

        matrix_accumulator_99 = thread_scratch[2];

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_88, matrix_fragment_89) - 0x200ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_88, matrix_fragment_89),
                                            matrix_fragment_72, matrix_fragment_73,
                                            matrix_fragment_74, matrix_fragment_75, memory_ranges,
                                            error_counts);

        matrix_accumulator_98 = thread_scratch[3];

        matrix_accumulator_97 = thread_scratch[4];

        matrix_accumulator_96 = thread_scratch[5];

        matrix_fragment_95 = thread_scratch[0];

        matrix_fragment_94 = thread_scratch[1];

        split_address_words(matrix_fragment_90, matrix_fragment_91,
                            int64_t(int32_t(matrix_fragment_76)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_112, matrix_accumulator_113));

        packed_half_157 = thread_scratch[13];

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_90, matrix_fragment_91) - 0x200ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_90, matrix_fragment_91),
                                            matrix_fragment_80, matrix_fragment_81,
                                            matrix_fragment_82, matrix_fragment_83, memory_ranges,
                                            error_counts);

        matrix_accumulator_155 = thread_scratch[8];

        matrix_accumulator_154 = thread_scratch[9];

        matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                          select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                          select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_164),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_165),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_152, matrix_accumulator_153, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_3),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        matrix_fragment_68 = thread_scratch[10];

        matrix_fragment_69 = thread_scratch[11];

        multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_70 = thread_scratch[6];

        matrix_fragment_71 = thread_scratch[7];

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        matrix_fragment_87 = thread_scratch[12];

        matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_155),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_154),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                          select_half_operand<2, false, false>(matrix_fragment_68));

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_150,
                                 matrix_accumulator_151);

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_160),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_161),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_162),
                          select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_163),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_158),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_159),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_94,
                                 matrix_fragment_95);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_152, matrix_accumulator_153, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_152,
                                 matrix_accumulator_153);

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_138,
                                 matrix_accumulator_139);

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_154, matrix_accumulator_155, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_146,
                                 matrix_accumulator_147);

        matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                          select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        packed_half_157 = 0x2b28u;

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, packed_half_144,
                                 packed_half_145);

        matrix_fragment_85 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        packed_half_0 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                           0x44004400u);

        matrix_fragment_85 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_148,
                                 matrix_accumulator_149);

        packed_half_0 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_72,
                                 matrix_fragment_73);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_74,
                                 matrix_fragment_75);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0x44004400u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_85),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_0),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_88 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        packed_half_156 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_0),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_88 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0xc400c400u);

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_89 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

        matrix_fragment_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                select_half_operand<2, false, false>(packed_half_0));

        packed_half_156 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_156), 0xc400c400u);

        packed_half_0 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_fragment_92 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_89 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0xc400c400u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_88),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        packed_half_0 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_156),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        packed_half_156 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_156),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_89),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_0),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_89 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_69 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_154), 0x44004400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_0),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

        matrix_fragment_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(packed_half_0));

        packed_half_0 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_155), 0x44004400u);

        matrix_fragment_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                select_half_operand<2, false, false>(matrix_fragment_89));

        matrix_fragment_95 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0x44004400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        packed_half_0 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        matrix_fragment_95 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0xc400c400u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99), 0x44004400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_0),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        packed_half_156 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(packed_half_156));

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_95),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_accumulator_96 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_144), 0x44004400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_0),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        matrix_accumulator_97 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_145), 0x44004400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_accumulator_96 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0xc400c400u);

        matrix_accumulator_155 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_155),
                                select_half_operand<2, false, false>(packed_half_0));

        packed_half_0 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_148), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_accumulator_97 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0xc400c400u);

        matrix_accumulator_154 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_154),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        packed_half_0 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_fragment_95));

        matrix_accumulator_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_149), 0x44004400u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_96),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_71 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_accumulator_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_accumulator_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0xc400c400u);

        matrix_accumulator_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_70 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_150), 0x44004400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_0),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_97),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_71 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

        packed_half_0 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_0),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

        matrix_accumulator_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_69 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_98),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_71),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

        matrix_accumulator_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_151), 0x44004400u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71),
            select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_accumulator_150 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_70 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_152), 0x44004400u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_72 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_153), 0x44004400u);

        matrix_accumulator_151 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_75 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_142), 0x44004400u);

        matrix_accumulator_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_149),
                                select_half_operand<2, false, false>(matrix_accumulator_98));

        matrix_accumulator_152 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_152),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_accumulator_149 = threadIdx.x & 31u;

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_72),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_75 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0xc400c400u);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_146), 0x44004400u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        matrix_accumulator_153 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_153),
                                select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_75),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_73 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_143), 0x44004400u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_74),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_73 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0xc400c400u);

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_72 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_147), 0x44004400u);

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                select_half_operand<2, false, false>(matrix_accumulator_96));

        packed_half_144 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_86), 0x44004400u);

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        packed_half_145 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_87), 0x44004400u);

        matrix_fragment_76 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_73),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0xc400c400u);

        packed_half_144 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_144), 0xc400c400u);

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73),
            select_half_operand<2, false, false>(matrix_fragment_76), 0x3b283b28u);

        packed_half_145 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_145), 0xc400c400u);

        matrix_fragment_76 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_72),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        split_address_words(matrix_fragment_80, matrix_fragment_81,
                            int64_t(int32_t(matrix_accumulator_149)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_114, matrix_accumulator_115));

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_144),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_145),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_76), 0x3b283b28u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_80, matrix_fragment_81) - 0x200ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        packed_half_144 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_144),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        packed_half_145 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_145),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_accumulator_142 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_80, matrix_fragment_81) - 0x600ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        matrix_accumulator_146 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                                select_half_operand<2, false, false>(matrix_fragment_74));

        matrix_accumulator_143 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                                select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_accumulator_147 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                                select_half_operand<2, false, false>(matrix_fragment_72));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_80, matrix_fragment_81) - 0x400ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_80, matrix_fragment_81),
                                            matrix_fragment_80, matrix_fragment_81,
                                            matrix_fragment_82, matrix_fragment_83, memory_ranges,
                                            error_counts);

        matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                           select_half_operand<2, false, false>(0u));

        packed_half_144 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                select_half_operand<2, false, false>(packed_half_144));

        matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                          select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_93 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_155),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_150),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                          select_half_operand<2, false, false>(matrix_fragment_92));

        matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_93 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                          select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_86 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_fragment_86));

        matrix_fragment_92 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_154),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_152),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_99 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_138), 0x44004400u);

        packed_half_145 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                select_half_operand<2, false, false>(packed_half_145));

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                          select_half_operand<2, false, false>(matrix_fragment_94));

        matrix_fragment_92 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                          select_half_operand<2, false, false>(matrix_fragment_92));

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_151),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                          select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_94 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_146),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_99 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99), 0xc400c400u);

        matrix_fragment_95 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                          select_half_operand<2, false, false>(0u));

        packed_half_144 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_140), 0x44004400u);

        matrix_fragment_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        matrix_fragment_94 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                          select_half_operand<2, false, false>(matrix_fragment_94));

        matrix_accumulator_142 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_99),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_153),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_95 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                          select_half_operand<2, false, false>(matrix_fragment_95));

        packed_half_144 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_144), 0xc400c400u);

        matrix_accumulator_143 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_141), 0x44004400u);

        matrix_accumulator_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99),
            select_half_operand<2, false, false>(matrix_accumulator_142), 0x3b283b28u);

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                          select_half_operand<2, false, false>(matrix_fragment_91));

        packed_half_145 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_144),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_accumulator_142 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_139), 0x44004400u);

        matrix_accumulator_143 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_143), 0xc400c400u);

        packed_half_144 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_144),
            select_half_operand<2, false, false>(packed_half_145), 0x3b283b28u);

        matrix_accumulator_142 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_142), 0xc400c400u);

        packed_half_145 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_143),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        matrix_accumulator_146 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_142),
            select_half_operand<0, false, true>(packed_half_157), 0x37283728u);

        packed_half_145 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_143),
            select_half_operand<2, false, false>(packed_half_145), 0x3b283b28u);

        packed_half_144 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                select_half_operand<2, false, false>(packed_half_144));

        packed_half_0 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_148),
                                select_half_operand<2, false, false>(packed_half_0));

        matrix_accumulator_146 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_142),
            select_half_operand<2, false, false>(matrix_accumulator_146), 0x3b283b28u);

        packed_half_145 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                select_half_operand<2, false, false>(packed_half_145));

        matrix_accumulator_98 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                select_half_operand<2, false, false>(matrix_accumulator_99));

        packed_half_144 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_146 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                select_half_operand<2, false, false>(matrix_accumulator_146));

        packed_half_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                        select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(0u));

        packed_half_145 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_97 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(matrix_accumulator_98));

        matrix_accumulator_98 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                          select_half_operand<2, false, false>(packed_half_144));

        matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                           select_half_operand<2, false, false>(packed_half_156));

        matrix_accumulator_96 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(packed_half_0));

        matrix_accumulator_99 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_146),
                          select_half_operand<2, false, false>(packed_half_145));

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_index_word_5 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                            combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x400u);

        split_address_words(matrix_accumulator_116, matrix_accumulator_117,
                            combine_words(matrix_accumulator_116, matrix_accumulator_117) + 0x400u);

        split_address_words(matrix_accumulator_114, matrix_accumulator_115,
                            combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x800u);

        uniform_index_word_4 = uniform_index_word_5;

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_accumulator_96,
                                 matrix_accumulator_97, matrix_accumulator_98,
                                 matrix_accumulator_99, matrix_fragment_68, matrix_fragment_69,
                                 matrix_fragment_64, matrix_fragment_65);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_accumulator_96,
                                 matrix_accumulator_97, matrix_accumulator_98,
                                 matrix_accumulator_99, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_66, matrix_fragment_67);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(
            matrix_accumulator_100, matrix_accumulator_101, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_72,
            matrix_fragment_73, matrix_accumulator_100, matrix_accumulator_101);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(
            matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_74,
            matrix_fragment_75, matrix_accumulator_102, matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(
            matrix_accumulator_104, matrix_accumulator_105, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_76,
            matrix_fragment_77, matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(
            matrix_accumulator_106, matrix_accumulator_107, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_78,
            matrix_fragment_79, matrix_accumulator_106, matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(
            matrix_accumulator_108, matrix_accumulator_109, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_80,
            matrix_fragment_81, matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(
            matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_96,
            matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_82,
            matrix_fragment_83, matrix_accumulator_110, matrix_accumulator_111);
    } while (!uniform_condition_0);
    packed_half_0 = blockIdx.x;

    matrix_fragment_6 = parameters[parameter_tensor_height_6];
    matrix_fragment_7 = parameters[parameter_tensor_width_7];

    uniform_index_word_4 = uniform_block_index_6 << (0x1u & 31u);

    matrix_fragment_73 = threadIdx.y;

    matrix_accumulator_148 = blockIdx.z;

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    index_word_3 = parameters[parameter_output_high_3];

    matrix_fragment_74 = threadIdx.x;

    matrix_fragment_72 = threadIdx.z;

    matrix_fragment_4 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x2u >> 32) + matrix_fragment_7;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_6;

    matrix_fragment_70 = packed_half_0 * 0x2u;

    matrix_fragment_69 = uint32_t(int32_t(matrix_fragment_4) >> 0x2u);

    matrix_fragment_68 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    condition_0 = int32_t(matrix_fragment_70) >= int32_t(matrix_fragment_69);

    matrix_fragment_4 = matrix_fragment_69 * uniform_index_word_4 + matrix_fragment_70;

    matrix_fragment_6 = matrix_accumulator_148 * 0x4u + matrix_fragment_73;

    condition_0 = int32_t(matrix_fragment_68) <= int32_t(uniform_index_word_4) || condition_0;

    matrix_fragment_6 = matrix_fragment_6 * 0x40u;

    matrix_fragment_71 = matrix_fragment_6 * 0x4u;

    matrix_fragment_9 = matrix_fragment_4 * 0x800u + matrix_fragment_71;

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, index_word_3));
    if (!condition_0) {
        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(matrix_fragment_11));

        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_accumulator_149)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_21));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    matrix_fragment_14 = matrix_fragment_70 + 0x1u;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = matrix_fragment_69 * uniform_index_word_4 + matrix_fragment_69;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(0u));

    condition_0 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_69);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    condition_0 = int32_t(matrix_fragment_68) <= int32_t(uniform_index_word_4) || condition_0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_45 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_18, 0x5410u);

        matrix_fragment_16 = permute_signed(matrix_fragment_16, matrix_fragment_19, 0x5410u);

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_12, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_13, 0x5410u);

        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_accumulator_149)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        matrix_fragment_5 = permute_signed(matrix_fragment_5, matrix_fragment_10, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_11, 0x5410u);

        matrix_fragment_11 = permute_signed(matrix_accumulator_45, matrix_fragment_20, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_accumulator_44, matrix_fragment_21, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x2000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        matrix_fragment_8 = matrix_fragment_16;

        matrix_fragment_9 = matrix_fragment_15;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x2200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_index_word_5 = uniform_index_word_4 + 0x1u;

    matrix_fragment_4 = matrix_fragment_70 + matrix_fragment_17;

    matrix_fragment_73 = matrix_fragment_73 | matrix_fragment_74;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                                      select_half_operand<2, false, false>(0u));

    condition_0 = int32_t(matrix_fragment_68) <= int32_t(uniform_index_word_5);

    matrix_fragment_71 = matrix_fragment_4 * 0x800u + matrix_fragment_71;

    condition_2 = (matrix_fragment_72 | matrix_fragment_73) != 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_2, index_word_3,
                        int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, index_word_3));

    condition_1 = int32_t(matrix_fragment_70) >= int32_t(matrix_fragment_69) || condition_0;

    condition_0 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_69) || condition_0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        matrix_fragment_9 = permute_signed(matrix_fragment_9, matrix_fragment_4, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_5, 0x5410u);

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_accumulator_149)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, index_word_3));

        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_6, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_fragment_10, matrix_fragment_7, 0x5410u);

        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_18, 0x5410u);

        matrix_fragment_14 = permute_signed(matrix_fragment_14, matrix_fragment_19, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_16, 0x5410u);

        matrix_fragment_12 = permute_signed(matrix_fragment_12, matrix_fragment_17, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_104 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_10, 0x5410u);

        split_address_words(matrix_accumulator_2, index_word_3,
                            int64_t(int32_t(matrix_accumulator_149)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, index_word_3));

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_11, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, matrix_fragment_8, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_9, 0x5410u);

        matrix_fragment_11 = permute_signed(matrix_accumulator_109, matrix_fragment_14, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_accumulator_108, matrix_fragment_15, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, index_word_3) + 0x2000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        matrix_fragment_9 = permute_signed(matrix_accumulator_105, matrix_fragment_12, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_104, matrix_fragment_13, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, index_word_3) + 0x2200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    __syncthreads();

    if (condition_2) {
        return;
    }

    matrix_fragment_4 = thread_scratch[22];

    matrix_accumulator_2 = parameters[parameter_publication_low_12];
    index_word_3 = parameters[parameter_publication_high_13];

    matrix_fragment_5 = matrix_accumulator_148 + uniform_index_word_4;

    matrix_fragment_5 = matrix_fragment_5 * matrix_fragment_4 + packed_half_0;

    split_address_words(matrix_accumulator_2, index_word_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, index_word_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, index_word_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(176) void phoenix_global_attention_split_heads_512_feed_forward_input_view_tile_sync(
    KernelArguments parameters) {
    run_global_attention_split_heads_512_feed_forward_input_view_tile_sync<false>(parameters.words,
                                                                                  nullptr, nullptr);
}
