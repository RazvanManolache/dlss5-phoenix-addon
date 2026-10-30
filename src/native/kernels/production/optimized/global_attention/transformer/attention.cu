#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: global_attention/transformer/attention.cu
// Coarsest 1024-channel global-transformer stage. Attention spans the full latent token set.
// Projection and feed-forward modules retain their explicit publication and synchronization boundaries.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 9232
#include "../../../common/device_memory_scalar.cuh"
#include "../../../common/attention_tail.cuh"
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_q_low_0 = 0,
    parameter_input_q_high_1 = 1,
    parameter_input_k_low_2 = 2,
    parameter_input_k_high_3 = 3,
    parameter_input_v_low_4 = 4,
    parameter_input_v_high_5 = 5,
    parameter_output_low_6 = 6,
    parameter_output_high_7 = 7,
    parameter_wait_low_10 = 10,
    parameter_wait_high_11 = 11,
    parameter_publication_low_12 = 12,
    parameter_publication_high_13 = 13,
    parameter_LH_14 = 14,
    parameter_LW_15 = 15,
};

struct alignas(8) KernelArguments {
    uint32_t words[16];
};
static_assert(sizeof(KernelArguments) == 64, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_transformer_attention(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t index_word_3 = 0;
    uint32_t address_word_4 = 0;
    uint32_t index_word_5 = 0;
    uint32_t index_word_6 = 0;
    uint32_t index_word_7 = 0;
    uint32_t index_word_8 = 0;
    uint32_t address_word_9 = 0;
    uint32_t packed_half_10 = 0;
    uint32_t packed_half_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t matrix_fragment_16 = 0;
    uint32_t matrix_fragment_17 = 0;
    uint32_t matrix_fragment_18 = 0;
    uint32_t matrix_fragment_19 = 0;
    uint32_t packed_half_20 = 0;
    uint32_t packed_half_21 = 0;
    uint32_t packed_half_22 = 0;
    uint32_t packed_half_23 = 0;
    uint32_t packed_half_24 = 0;
    uint32_t packed_half_25 = 0;
    uint32_t packed_half_26 = 0;
    uint32_t packed_half_27 = 0;
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
    uint32_t packed_half_104 = 0;
    uint32_t packed_half_105 = 0;
    uint32_t packed_half_106 = 0;
    uint32_t packed_half_107 = 0;
    uint32_t packed_half_108 = 0;
    uint32_t packed_half_109 = 0;
    uint32_t packed_half_110 = 0;
    uint32_t packed_half_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
    uint32_t packed_half_114 = 0;
    uint32_t packed_half_115 = 0;
    uint32_t packed_half_116 = 0;
    uint32_t index_word_117 = 0;
    uint32_t index_word_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t packed_half_120 = 0;
    uint32_t packed_half_121 = 0;
    uint32_t packed_half_122 = 0;
    uint32_t packed_half_123 = 0;
    uint32_t index_word_124 = 0;
    uint32_t index_word_125 = 0;
    uint32_t tile_word_126 = 0;
    uint32_t packed_half_127 = 0;
    uint32_t packed_half_128 = 0;
    uint32_t packed_half_129 = 0;
    uint32_t packed_half_130 = 0;
    uint32_t packed_half_131 = 0;
    uint32_t packed_half_132 = 0;
    uint32_t packed_half_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t packed_half_136 = 0;
    uint32_t packed_half_137 = 0;
    uint32_t packed_half_138 = 0;
    uint32_t packed_half_139 = 0;
    uint32_t packed_half_140 = 0;
    uint32_t packed_half_141 = 0;
    uint32_t packed_half_142 = 0;
    uint32_t packed_half_143 = 0;
    uint32_t packed_half_144 = 0;
    uint32_t packed_half_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t packed_half_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t packed_half_150 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_reused_word_7 = 0;
    uint32_t uniform_reused_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
    uint32_t uniform_address_word_10 = 0;
    uint32_t uniform_address_word_11 = 0;
    uint32_t uniform_index_word_79 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    index_word_124 = parameters[parameter_LH_14];
    index_word_125 = parameters[parameter_LW_15];

    address_word_4 = threadIdx.x;

    packed_half_21 = threadIdx.y;

    index_word_8 = blockIdx.y;

    matrix_fragment_14 = blockIdx.x;

    address_word_9 = threadIdx.z;

    index_word_125 = index_word_125 * index_word_124;

    index_word_124 = 0u;

    condition_2 = uint32_t(index_word_125) != 0u;

    address_word_4 = address_word_4 | packed_half_21;
    condition_0 = address_word_4 != 0;

    if (condition_2) {
        address_word_2 = index_word_125 + 0xffffffffu;

        index_word_0 = index_word_125 + 0xffffffffu;

        index_word_6 = index_word_125 + 0xffffffffu;

        index_word_5 = uint32_t(int32_t(address_word_2) >> 0x1fu);

        index_word_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);

        address_word_2 = uint32_t(combine_words(index_word_5, 0u) << 0x7u >> 32) + address_word_2;
    }

    index_word_5 = index_word_8 * 0x2u;

    if (condition_2) {
        index_word_0 = uint32_t(combine_words(index_word_3, 0u) << 0x5u >> 32) + index_word_0;

        index_word_124 = uint32_t(int64_t(int32_t(address_word_2)) * (1ll << 0x19u) >> 32) + 0x1u;
    }

    index_word_3 = 0u;

    address_word_2 = index_word_5 & 0x1fffffeu;

    if (condition_2) {
        index_word_0 = index_word_0 & 0xffffffe0u;
    }
    if (!condition_0) {
        uniform_address_word_6 = 0u;

        uniform_reused_word_7 = blockDim.x;

        uniform_reused_word_8 = blockDim.y;

        uniform_address_word_4 = 0x400u;

        uniform_index_word_5 = uniform_address_word_4 + 0x2000u;

        uniform_address_word_4 = uniform_reused_word_7 * uniform_reused_word_8;

        uniform_address_word_6 = (uniform_address_word_6 << 0x18u) + uniform_index_word_5;

        uniform_address_word_4 = 0u - uniform_address_word_4 + 0x100000u;

        uniform_index_word_5 = uniform_address_word_4 << (0xbu & 31u);

        uniform_address_word_4 = uniform_address_word_4 << (0x1u & 31u);

        __threadfence_block();

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }
    index_word_7 = address_word_2 + 0x2u;

    phoenix_architecture::finish_compatibility_copies();

    condition_1 = (address_word_4 | address_word_9) != 0;

    matrix_fragment_12 = matrix_fragment_14 * 0x20u;

    if (condition_2) {
        index_word_3 = index_word_0 + 0x20u;
    }

    packed_half_10 = min(int32_t(index_word_7), int32_t(index_word_124));

    if (condition_2) {
        address_word_9 = uint32_t(int32_t(index_word_6) >> 0x1fu);
    }

    uniform_reused_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    address_word_2 = uint32_t(int32_t(index_word_3) >> 0x1fu);

    condition_0 = int32_t(index_word_5) >= int32_t(packed_half_10) || condition_1;

    if (condition_2) {
        index_word_0 = uint32_t(combine_words(address_word_9, 0u) << 0x6u >> 32) + index_word_6;
    }

    address_word_4 = uint32_t(combine_words(address_word_2, 0u) << 0x4u >> 32) + index_word_3;

    address_word_2 = uint32_t(combine_words(address_word_2, 0u) << 0x5u >> 32) + index_word_3;

    index_word_118 = 0u;

    if (condition_2) {
        index_word_118 = uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1au) >> 32) + 0x1u;
    }

    index_word_0 = uint32_t(int32_t(address_word_4) >> 0x4u);

    address_word_2 = uint32_t(int32_t(address_word_2) >> 0x5u);
    if (!condition_0) {
        index_word_6 = index_word_5;

        index_word_7 = uint32_t(uint32_t(matrix_fragment_14) >> 0x1u);
        do {
            address_word_4 = parameters[parameter_wait_low_10];
            index_word_5 = parameters[parameter_wait_high_11];

            address_word_9 = index_word_6 * 0x10u + index_word_7;

            split_address_words(address_word_4, index_word_5,
                                uint64_t(uint32_t(address_word_9)) * uint64_t(uint32_t(0x4u)) +
                                    combine_words(address_word_4, index_word_5));

            address_word_9 = pipeline_read<CheckBounds>(combine_words(address_word_4, index_word_5),
                                                        memory_ranges, error_counts);

            index_word_6 = index_word_6 + 0x1u;

            condition_2 = uint32_t(index_word_6) != uint32_t(packed_half_10);

            condition_0 = int32_t(address_word_9) > -1;
            if (!condition_0) {
                do {
                    __nanosleep(0x40u);

                    address_word_9 = pipeline_read<CheckBounds>(
                        combine_words(address_word_4, index_word_5), memory_ranges, error_counts);

                    condition_0 = int32_t(address_word_9) > -1;
                } while (!condition_0);
            }
        } while (condition_2);
    }
    matrix_fragment_13 = index_word_8 * 0x4u + packed_half_21;

    __syncwarp(__activemask());

    matrix_fragment_13 = matrix_fragment_13 * 0x4u;

    phoenix_architecture::finish_compatibility_copies();

    matrix_fragment_15 = matrix_fragment_13 + 0x1u;

    matrix_fragment_17 = matrix_fragment_13 + 0x2u;

    matrix_fragment_19 = matrix_fragment_13 + 0x3u;

    condition_0 = int32_t(matrix_fragment_13) >= int32_t(index_word_0);

    condition_2 = int32_t(matrix_fragment_15) >= int32_t(index_word_0);

    condition_3 = int32_t(matrix_fragment_17) >= int32_t(index_word_0);

    condition_4 = int32_t(matrix_fragment_19) >= int32_t(index_word_0);

    if (!condition_0) {
        packed_half_23 = threadIdx.x & 31u;

        address_word_4 = parameters[parameter_input_q_low_0];
        index_word_5 = parameters[parameter_input_q_high_1];
    }

    if (!condition_2) {
        matrix_fragment_15 = matrix_fragment_15 * 0x20u + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_2) {
        packed_half_25 = threadIdx.x & 31u;
    }

    if (!condition_3) {
        matrix_fragment_17 = matrix_fragment_17 * 0x20u + matrix_fragment_14;
    }

    if (condition_3) {
        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_4) {
        matrix_fragment_19 = matrix_fragment_19 * 0x20u + matrix_fragment_14;
    }

    if (!condition_3) {
        packed_half_27 = threadIdx.x & 31u;
    }

    if (!condition_2) {
        index_word_6 = parameters[parameter_input_q_low_0];
        index_word_7 = parameters[parameter_input_q_high_1];

        matrix_fragment_15 = matrix_fragment_15 * 0x80u;
    }

    if (condition_4) {
        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_4) {
        matrix_accumulator_29 = threadIdx.x & 31u;
    }

    if (!condition_3) {
        matrix_fragment_17 = matrix_fragment_17 * 0x80u;
    }

    if (condition_2) {
        matrix_fragment_39 = permute_signed(matrix_fragment_39, matrix_fragment_39, 0x5410u);
    }

    if (!condition_4) {
        matrix_fragment_19 = matrix_fragment_19 * 0x80u;
    }

    if (condition_3) {
        matrix_fragment_43 = permute_signed(matrix_fragment_43, matrix_fragment_43, 0x5410u);
    }

    if (!condition_3) {
        index_word_8 = parameters[parameter_input_q_low_0];
        address_word_9 = parameters[parameter_input_q_high_1];
    }

    if (condition_4) {
        matrix_fragment_47 = permute_signed(matrix_fragment_47, matrix_fragment_47, 0x5410u);
    }

    if (!condition_0) {
        matrix_fragment_13 = matrix_fragment_13 * 0x20u + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_36 = matrix_fragment_39;

        matrix_fragment_37 = matrix_fragment_39;

        matrix_fragment_38 = matrix_fragment_39;
    }

    if (condition_3) {
        matrix_fragment_40 = matrix_fragment_43;

        matrix_fragment_41 = matrix_fragment_43;
    }

    if (!condition_4) {
        packed_half_10 = parameters[parameter_input_q_low_0];
        packed_half_11 = parameters[parameter_input_q_high_1];
    }

    if (condition_3) {
        matrix_fragment_42 = matrix_fragment_43;
    }

    if (condition_4) {
        matrix_fragment_44 = matrix_fragment_47;

        matrix_fragment_45 = matrix_fragment_47;

        matrix_fragment_46 = matrix_fragment_47;
    }

    if (!condition_0) {
        matrix_fragment_13 = matrix_fragment_13 * 0x80u;
    }

    if (condition_0) {
        matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_2) {
        split_address_words(index_word_6, index_word_7,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(index_word_6, index_word_7));
    }

    if (condition_0) {
        matrix_fragment_35 = permute_signed(matrix_fragment_35, matrix_fragment_35, 0x5410u);
    }

    if (!condition_0) {
        split_address_words(address_word_4, index_word_5,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_4, index_word_5));
    }

    if (condition_0) {
        matrix_fragment_32 = matrix_fragment_35;
    }

    if (!condition_3) {
        split_address_words(index_word_8, address_word_9,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(index_word_8, address_word_9));
    }

    if (condition_0) {
        matrix_fragment_33 = matrix_fragment_35;

        matrix_fragment_34 = matrix_fragment_35;
    }

    if (!condition_2) {
        split_address_words(index_word_6, index_word_7,
                            int64_t(int32_t(packed_half_25)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_6, index_word_7));
    }

    if (!condition_4) {
        split_address_words(packed_half_10, packed_half_11,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_10, packed_half_11));
    }

    if (!condition_2) {
        load_four_words_cached<CheckBounds>(
            combine_words(index_word_6, index_word_7), matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);
    }

    if (!condition_3) {
        split_address_words(index_word_8, address_word_9,
                            int64_t(int32_t(packed_half_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_8, address_word_9));
    }

    if (!condition_4) {
        split_address_words(packed_half_10, packed_half_11,
                            int64_t(int32_t(matrix_accumulator_29)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_10, packed_half_11));
    }

    if (!condition_3) {
        load_four_words_cached<CheckBounds>(
            combine_words(index_word_8, address_word_9), matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);
    }

    if (!condition_0) {
        split_address_words(address_word_4, index_word_5,
                            int64_t(int32_t(packed_half_23)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_4, index_word_5));
    }

    if (!condition_4) {
        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_10, packed_half_11), matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);
    }

    if (!condition_0) {
        load_four_words_cached<CheckBounds>(
            combine_words(address_word_4, index_word_5), matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);
    }

    condition_0 = int32_t(index_word_118) >= int32_t(0x1u);
    if (condition_0) {
        uniform_address_word_6 = 0u;

        uniform_address_word_4 = 0x400u;

        address_word_4 = index_word_3 & 0xffffffe0u;

        uniform_index_word_5 = uniform_address_word_4 + 0x1000u;

        index_word_5 = packed_half_21 & 0x1u;

        matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_14) >> 0x1u);

        matrix_fragment_12 =
            uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1cu >> 32) + index_word_5;

        condition_2 = uint32_t(address_word_4) != uint32_t(0x20u);

        index_word_6 = uint32_t(uint32_t(packed_half_21) >> 0x1u);

        matrix_fragment_13 = min(int32_t(index_word_118), int32_t(0x2u));

        packed_half_10 = 0u;

        uniform_address_word_4 = (uniform_address_word_6 << 0x18u) + uniform_address_word_4;

        uniform_index_word_5 = (uniform_address_word_6 << 0x18u) + uniform_index_word_5;

        index_word_7 = (packed_half_21 << 0x9u) + uniform_address_word_4;

        address_word_9 = (packed_half_21 << 0x9u) + uniform_index_word_5;
        do {
            packed_half_11 = uint32_t(uint32_t(packed_half_10) >> 0x1u);

            condition_0 = int32_t(index_word_124) <= int32_t(packed_half_11) || condition_1;
            if (!condition_0) {
                address_word_4 = parameters[parameter_wait_low_10];
                index_word_5 = parameters[parameter_wait_high_11];

                packed_half_11 = packed_half_11 * 0x10u + matrix_fragment_18;

                split_address_words(address_word_4, index_word_5,
                                    uint64_t(uint32_t(packed_half_11)) * uint64_t(uint32_t(0x4u)) +
                                        combine_words(address_word_4, index_word_5));

                index_word_8 = pipeline_read<CheckBounds>(
                    combine_words(address_word_4, index_word_5), memory_ranges, error_counts);

                condition_0 = int32_t(index_word_8) > -1;
                if (!condition_0) {
                    do {
                        __nanosleep(0x40u);

                        index_word_8 =
                            pipeline_read<CheckBounds>(combine_words(address_word_4, index_word_5),
                                                       memory_ranges, error_counts);

                        condition_0 = int32_t(index_word_8) > -1;
                    } while (!condition_0);
                }
            }
            __syncwarp(__activemask());

            uniform_index_word_5 = 0u;

            index_word_5 = packed_half_10 * 0x4u + packed_half_21;

            uniform_address_word_4 = 0x400u;

            phoenix_architecture::finish_compatibility_copies();

            uniform_address_word_4 = uniform_address_word_4 + 0x2000u;

            address_word_4 = index_word_5 * 0x20u + matrix_fragment_14;

            condition_3 = int32_t(index_word_5) >= int32_t(index_word_0);

            condition_0 = int32_t(index_word_5) >= int32_t(index_word_0);

            index_word_5 = 0u;

            address_word_4 = address_word_4 * 0x80u;

            split_address_words(address_word_4, index_word_5,
                                !condition_0 ? combine_words(address_word_4, index_word_5)
                                             : combine_words(0u, 0u));

            uniform_address_word_4 = (uniform_index_word_5 << 0x18u) + uniform_address_word_4;

            packed_half_11 = (packed_half_10 << 0x3u) + uniform_address_word_4;
            if (condition_3) {
                address_word_4 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                index_word_5 = packed_half_10 * 0x800u + index_word_7;

                packed_half_24 = uniform_address_word_4;

                matrix_fragment_15 = uniform_address_word_4;

                packed_half_24 = permute_signed(packed_half_24, matrix_fragment_15, 0x5410u);

                packed_half_25 = packed_half_24;

                packed_half_26 = packed_half_24;

                packed_half_27 = packed_half_24;

                address_word_4 = address_word_4 * 0x10u + index_word_5;

                store_four_shared_words<CheckBounds>(shared_tile, address_word_4, packed_half_24,
                                                     packed_half_25, packed_half_26, packed_half_27,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    matrix_fragment_16 = parameters[parameter_input_k_low_2];
                    matrix_fragment_17 = parameters[parameter_input_k_high_3];

                    uniform_address_word_4 = packed_half_10;

                    uniform_address_word_6 = index_word_7;

                    uniform_index_word_5 = packed_half_11;

                    uniform_address_word_4 =
                        (uniform_address_word_4 << 0xbu) + uniform_address_word_6;

                    uniform_address_word_6 = 0x20u;

                    const uint64_t byte_address_0 =
                        scaled_word_address(combine_words(matrix_fragment_16, matrix_fragment_17),
                                            combine_words(address_word_4, index_word_5));
                    index_word_8 = uint32_t(byte_address_0);

                    address_word_4 = uint32_t(byte_address_0 >> 32);

                    uniform_address_word_10 = index_word_8;

                    uniform_address_word_11 = address_word_4;

                    address_word_4 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_6 * 16u, packed_half_11, shared_bytes, memory_ranges,
                        error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            index_word_5 = packed_half_10 * 0x2u + index_word_6;

            condition_0 = int32_t(index_word_5) < int32_t(address_word_2) || !condition_2;
            if (!condition_0) {
                address_word_4 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                index_word_5 = packed_half_10 * 0x800u + address_word_9;

                packed_half_24 = uniform_address_word_4;

                packed_half_11 = uniform_address_word_4;

                packed_half_24 = permute_signed(packed_half_24, packed_half_11, 0x5410u);

                packed_half_25 = packed_half_24;

                packed_half_26 = packed_half_24;

                packed_half_27 = packed_half_24;

                address_word_4 = address_word_4 * 0x10u + index_word_5;

                store_four_shared_words<CheckBounds>(shared_tile, address_word_4, packed_half_24,
                                                     packed_half_25, packed_half_26, packed_half_27,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                index_word_5 = index_word_5 * 0x2000u;

                uniform_address_word_4 = 0xffffffffu;

                index_word_5 = condition_2 ? index_word_5 : 0u;

                address_word_4 = matrix_fragment_12 * 0x80u + index_word_5;

                index_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

                split_address_words(address_word_4, index_word_5,
                                    condition_0 ? combine_words(address_word_4, index_word_5)
                                                : combine_words(0u, 0u));

                condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_0) {
                    matrix_fragment_16 = parameters[parameter_input_v_low_4];
                    matrix_fragment_17 = parameters[parameter_input_v_high_5];

                    uniform_address_word_4 = packed_half_10;

                    uniform_address_word_6 = address_word_9;

                    uniform_index_word_5 = packed_half_11;

                    uniform_address_word_4 =
                        (uniform_address_word_4 << 0xbu) + uniform_address_word_6;

                    uniform_address_word_6 = 0x20u;

                    const uint64_t byte_address_1 =
                        scaled_word_address(combine_words(matrix_fragment_16, matrix_fragment_17),
                                            combine_words(address_word_4, index_word_5));
                    index_word_8 = uint32_t(byte_address_1);

                    address_word_4 = uint32_t(byte_address_1 >> 32);

                    uniform_address_word_10 = index_word_8;

                    uniform_address_word_11 = address_word_4;

                    address_word_4 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_6 * 16u, packed_half_11, shared_bytes, memory_ranges,
                        error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_4,
                        combine_words(uniform_address_word_10, uniform_address_word_11),
                        uniform_address_word_6 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            packed_half_10 = packed_half_10 + 0x1u;

            condition_0 = uint32_t(packed_half_10) != uint32_t(matrix_fragment_13);
        } while (condition_0);
    }
    index_word_5 = 0u;

    address_word_4 = 0x400u;

    condition_1 = int32_t(index_word_118) >= int32_t(0x1u);

    packed_half_119 = 0u;

    address_word_4 = address_word_4 + 0x2000u;

    index_word_7 = (index_word_5 << 0x18u) + address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = index_word_7;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(address_word_4, index_word_5, async_tokens[operand_a >> 3 & 1]);
    }
#else
    address_word_4 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_0 =
        wait_copy_barrier(shared_tile, index_word_7, async_tokens[(((index_word_7) >> 3) & 1)]);
#else
    phoenix_architecture::finish_compatibility_copies();
    condition_0 = true;
#endif
    if (!condition_0) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_0 = wait_copy_barrier(shared_tile, index_word_7,
                                            async_tokens[(((index_word_7) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_0 = true;
#endif
        } while (!condition_0);
    }
    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_accumulator_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_accumulator_88 = 0;
    matrix_accumulator_89 = 0;

    matrix_accumulator_90 = 0;
    matrix_accumulator_91 = 0;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    matrix_accumulator_86 = 0;
    matrix_accumulator_87 = 0;

    matrix_accumulator_94 = 0;
    matrix_accumulator_95 = 0;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_accumulator_92 = 0;
    matrix_accumulator_93 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_76 = 0;
    matrix_accumulator_77 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;
    if (condition_1) {
        index_word_3 = index_word_3 & 0xffffffe0u;

        matrix_accumulator_96 = 0;
        matrix_accumulator_97 = 0;

        packed_half_119 = 0u;

        index_word_117 = 0u;

        matrix_accumulator_80 = 0;
        matrix_accumulator_81 = 0;

        matrix_accumulator_30 = 0;
        matrix_accumulator_31 = 0;

        matrix_accumulator_88 = 0;
        matrix_accumulator_89 = 0;

        matrix_accumulator_90 = 0;
        matrix_accumulator_91 = 0;

        matrix_accumulator_82 = 0;
        matrix_accumulator_83 = 0;

        matrix_accumulator_86 = 0;
        matrix_accumulator_87 = 0;

        matrix_accumulator_94 = 0;
        matrix_accumulator_95 = 0;

        matrix_accumulator_84 = 0;
        matrix_accumulator_85 = 0;

        matrix_accumulator_28 = 0;
        matrix_accumulator_29 = 0;

        matrix_accumulator_92 = 0;
        matrix_accumulator_93 = 0;

        matrix_accumulator_98 = 0;
        matrix_accumulator_99 = 0;

        matrix_accumulator_78 = 0;
        matrix_accumulator_79 = 0;

        matrix_accumulator_76 = 0;
        matrix_accumulator_77 = 0;

        matrix_accumulator_100 = 0;
        matrix_accumulator_101 = 0;

        matrix_accumulator_102 = 0;
        matrix_accumulator_103 = 0;
        do {
            uniform_index_word_5 = 0u;

            index_word_8 = threadIdx.x & 31u;

            uniform_address_word_4 = 0x400u;

            index_word_6 = index_word_117 * 0x800u;

            address_word_4 = uniform_address_word_4;

            packed_half_127 = 0x2dbbu;

            index_word_7 = index_word_6 & 0x800u;

            matrix_fragment_62 = address_word_4 + 0x1000u;

            index_word_5 = uniform_index_word_5;

            __syncwarp(__activemask());

            index_word_6 = (index_word_5 << 0x18u) + address_word_4;

            matrix_fragment_62 = (index_word_5 << 0x18u) + matrix_fragment_62;

            {
                const uint32_t bits = index_word_8;
                condition_0 = bool(bits & 1u);
                condition_1 = bool(bits & 2u);
                condition_3 = bool(bits & 8u);
                condition_4 = bool(bits & 16u);
            }

            index_word_6 = index_word_6 + index_word_7;

            index_word_7 = matrix_fragment_62 + index_word_7;

            packed_half_120 = index_word_8 * 0x10u + index_word_6;

            tile_word_126 = index_word_8 * 0x10u + index_word_7;

            load_four_shared_words<CheckBounds>(
                shared_tile, packed_half_120, matrix_fragment_16, matrix_fragment_17,
                matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

            load_four_shared_words<CheckBounds>(
                shared_tile, packed_half_120 + 0x200u, matrix_fragment_12, matrix_fragment_13,
                matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

            load_four_shared_words<CheckBounds>(
                shared_tile, tile_word_126, matrix_fragment_72, matrix_fragment_73,
                matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

            multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_16, matrix_fragment_17, 0u, 0u);

            multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_16, matrix_fragment_17, 0u, 0u);

            matrix_fragment_54 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_55 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_16, matrix_fragment_17, 0u, 0u);

            matrix_fragment_54 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3dc23dc2u);

            matrix_fragment_55 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3dc23dc2u);

            packed_half_115 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3fe93fe9u);

            packed_half_116 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3fe93fe9u);

            packed_half_115 = (packed_half_115 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_52 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_116 = (packed_half_116 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_53 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_18, matrix_fragment_19, 0u, 0u);

            matrix_fragment_52 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3dc23dc2u);

            matrix_fragment_53 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3dc23dc2u);

            matrix_fragment_54 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_50),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_50 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3fe93fe9u);

            matrix_fragment_55 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_51),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_51 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3fe93fe9u);

            multiply_accumulate_half(packed_half_20, packed_half_21, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_14, matrix_fragment_15, 0u, 0u);

            matrix_fragment_54 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3dc23dc2u);

            matrix_fragment_55 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3dc23dc2u);

            matrix_fragment_52 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_48 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_16, matrix_fragment_17, 0u, 0u);

            matrix_fragment_52 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3dc23dc2u);

            matrix_fragment_49 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48), 0x3dc23dc2u);

            matrix_fragment_48 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3fe93fe9u);

            matrix_fragment_49 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49), 0x3fe93fe9u);

            packed_half_20 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_20),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_21 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(packed_half_24, packed_half_25, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_12, matrix_fragment_13, 0u, 0u);

            packed_half_20 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_20), 0x3dc23dc2u);

            packed_half_21 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_21), 0x3dc23dc2u);

            packed_half_20 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_20), 0x3fe93fe9u);

            packed_half_21 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_21), 0x3fe93fe9u);

            matrix_fragment_58 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_59 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(packed_half_26, packed_half_27, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_18, matrix_fragment_19, 0u, 0u);

            matrix_fragment_58 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58), 0x3dc23dc2u);

            matrix_fragment_59 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59), 0x3dc23dc2u);

            packed_half_107 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58), 0x3fe93fe9u);

            packed_half_108 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59), 0x3fe93fe9u);

            packed_half_107 = (packed_half_107 << 0x4u) + 0x3ffc4000u;

            packed_half_24 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_24),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_108 = (packed_half_108 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_12, matrix_fragment_13, 0u, 0u);

            matrix_fragment_53 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_26),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_26 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3fe93fe9u);

            matrix_fragment_55 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_27),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_27 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3fe93fe9u);

            multiply_accumulate_half(packed_half_22, packed_half_23, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_14, matrix_fragment_15, 0u, 0u);

            matrix_fragment_52 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_24), 0x3dc23dc2u);

            matrix_fragment_54 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3dc23dc2u);

            matrix_fragment_53 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_25),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_111 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3fe93fe9u);

            packed_half_25 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3fe93fe9u);

            matrix_fragment_55 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3dc23dc2u);

            matrix_fragment_53 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3dc23dc2u);

            matrix_fragment_17 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_17),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_24 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3fe93fe9u);

            matrix_fragment_16 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_16),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_53 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3fe93fe9u);

            multiply_accumulate_half(packed_half_10, packed_half_11, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_14, matrix_fragment_15, 0u, 0u);

            matrix_fragment_17 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_17), 0x3dc23dc2u);

            matrix_fragment_16 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_16), 0x3dc23dc2u);

            matrix_fragment_17 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_17), 0x3fe93fe9u);

            matrix_fragment_16 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_16), 0x3fe93fe9u);

            packed_half_25 = (packed_half_25 << 0x4u) + 0x3ffc4000u;

            packed_half_24 = (packed_half_24 << 0x4u) + 0x3ffc4000u;

            packed_half_23 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_111 = (packed_half_111 << 0x4u) + 0x3ffc4000u;

            packed_half_22 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_112 = (matrix_fragment_53 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_14, matrix_fragment_15, 0u, 0u);

            packed_half_23 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_23), 0x3dc23dc2u);

            packed_half_22 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_22), 0x3dc23dc2u);

            packed_half_23 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_23), 0x3fe93fe9u);

            packed_half_22 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_22), 0x3fe93fe9u);

            packed_half_109 = (packed_half_23 << 0x4u) + 0x3ffc4000u;

            packed_half_10 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_10),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_110 = (packed_half_22 << 0x4u) + 0x3ffc4000u;

            packed_half_11 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_11),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_71 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                              select_half_operand<2, false, false>(0u));

            multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_18, matrix_fragment_19, 0u, 0u);

            packed_half_10 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_10), 0x3dc23dc2u);

            packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                             select_half_operand<2, false, false>(packed_half_109));

            packed_half_11 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_11), 0x3dc23dc2u);

            matrix_fragment_55 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_10), 0x3fe93fe9u);

            packed_half_10 = (matrix_fragment_50 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_70 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                              select_half_operand<2, false, false>(0u));

            packed_half_110 = add_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                             select_half_operand<2, false, false>(packed_half_110));

            packed_half_22 = (matrix_fragment_55 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_15 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_15),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_70 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                              select_half_operand<2, false, false>(matrix_fragment_70));

            matrix_fragment_14 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_14),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_22),
                                               select_half_operand<2, false, false>(0u));

            multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_18, matrix_fragment_19, 0u, 0u);

            matrix_fragment_15 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_15), 0x3dc23dc2u);

            matrix_fragment_14 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_14), 0x3dc23dc2u);

            packed_half_104 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_15), 0x3fe93fe9u);

            matrix_fragment_15 = (packed_half_20 << 0x4u) + 0x3ffc4000u;

            packed_half_105 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_14), 0x3fe93fe9u);

            matrix_fragment_14 = (packed_half_21 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_57 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_104 = (packed_half_104 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_56 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_105 = (packed_half_105 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_12, matrix_fragment_13, 0u, 0u);

            matrix_fragment_57 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57), 0x3dc23dc2u);

            matrix_fragment_56 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56), 0x3dc23dc2u);

            packed_half_113 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57), 0x3fe93fe9u);

            packed_half_114 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56), 0x3fe93fe9u);

            matrix_fragment_56 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_11), 0x3fe93fe9u);

            packed_half_11 = (matrix_fragment_51 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_60 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_114 = (packed_half_114 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_61 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_113 = (packed_half_113 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_12, matrix_fragment_13, 0u, 0u);

            matrix_fragment_60 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60), 0x3dc23dc2u);

            address_word_9 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61), 0x3dc23dc2u);

            packed_half_106 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60), 0x3fe93fe9u);

            address_word_9 = minimum_half_pairs(
                select_half_operand<2, false, false>(address_word_9), 0x3fe93fe9u);

            packed_half_21 = (matrix_fragment_56 << 0x4u) + 0x3ffc4000u;

            packed_half_106 = (packed_half_106 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_19 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_19),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            address_word_9 = (address_word_9 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_18 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_18),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_68 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                              select_half_operand<2, false, false>(0u));

            packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                             select_half_operand<2, false, false>(packed_half_114));

            matrix_fragment_69 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                              select_half_operand<2, false, false>(0u));

            packed_half_113 = add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                             select_half_operand<2, false, false>(packed_half_113));

            matrix_fragment_18 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_18), 0x3dc23dc2u);

            packed_half_110 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                             select_half_operand<2, false, false>(packed_half_110));

            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                              select_half_operand<2, false, false>(0u));

            packed_half_106 = add_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                             select_half_operand<2, false, false>(packed_half_106));

            matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_9),
                                               select_half_operand<2, false, false>(0u));

            address_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_108),
                                            select_half_operand<2, false, false>(address_word_9));

            matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_21),
                                               select_half_operand<2, false, false>(0u));

            packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                             select_half_operand<2, false, false>(packed_half_109));

            matrix_fragment_51 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_13 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_13),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_50 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_52 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_12),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_12 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_19), 0x3dc23dc2u);

            matrix_fragment_13 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_13), 0x3dc23dc2u);

            matrix_fragment_54 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_12), 0x3fe93fe9u);

            matrix_fragment_57 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_13), 0x3fe93fe9u);

            matrix_fragment_13 = (matrix_fragment_17 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_12 = (matrix_fragment_16 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_16 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_66 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                              select_half_operand<2, false, false>(matrix_fragment_17));

            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                               select_half_operand<2, false, false>(matrix_fragment_15));

            matrix_fragment_19 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_18), 0x3fe93fe9u);

            matrix_fragment_17 = (matrix_fragment_49 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                               select_half_operand<2, false, false>(matrix_fragment_12));

            matrix_fragment_52 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3dc23dc2u);

            matrix_fragment_18 = (packed_half_27 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_67 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                              select_half_operand<2, false, false>(matrix_fragment_16));

            matrix_fragment_13 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                               select_half_operand<2, false, false>(matrix_fragment_14));

            matrix_fragment_16 = (matrix_fragment_48 << 0x4u) + 0x3ffc4000u;

            packed_half_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

            address_word_9 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                               select_half_operand<2, false, false>(matrix_fragment_13));

            matrix_fragment_52 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3fe93fe9u);

            matrix_fragment_48 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_60 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                              select_half_operand<2, false, false>(packed_half_23));

            packed_half_10 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                               select_half_operand<2, false, false>(matrix_fragment_17));

            matrix_fragment_61 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                              select_half_operand<2, false, false>(matrix_fragment_48));

            packed_half_11 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                               select_half_operand<2, false, false>(matrix_fragment_18));

            packed_half_23 = (packed_half_26 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_19 = (matrix_fragment_19 << 0x4u) + 0x3ffc4000u;

            packed_half_20 = (matrix_fragment_54 << 0x4u) + 0x3ffc4000u;

            packed_half_26 = (matrix_fragment_52 << 0x4u) + 0x3ffc4000u;

            packed_half_27 = (matrix_fragment_57 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_25),
                                               select_half_operand<2, false, false>(0u));

            load_four_shared_words<CheckBounds>(
                shared_tile, tile_word_126 + 0x200u, matrix_fragment_56, matrix_fragment_57,
                matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

            matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_24),
                                               select_half_operand<2, false, false>(0u));

            matrix_fragment_68 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                              select_half_operand<2, false, false>(matrix_fragment_68));

            matrix_fragment_69 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                              select_half_operand<2, false, false>(matrix_fragment_69));

            matrix_fragment_71 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                              select_half_operand<2, false, false>(matrix_fragment_71));

            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                              select_half_operand<2, false, false>(matrix_fragment_64));

            matrix_fragment_65 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                              select_half_operand<2, false, false>(matrix_fragment_65));

            matrix_fragment_62 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                              select_half_operand<2, false, false>(matrix_fragment_62));

            multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97,
                                     matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                     matrix_fragment_71, matrix_fragment_72, matrix_fragment_73,
                                     matrix_accumulator_96, matrix_accumulator_97);

            matrix_fragment_63 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_20),
                              select_half_operand<2, false, false>(matrix_fragment_63));

            matrix_fragment_19 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                               select_half_operand<2, false, false>(packed_half_22));

            matrix_fragment_52 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                              select_half_operand<2, false, false>(matrix_fragment_49));

            matrix_fragment_16 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                               select_half_operand<2, false, false>(packed_half_25));

            matrix_fragment_53 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_23),
                              select_half_operand<2, false, false>(matrix_fragment_48));

            packed_half_23 = add_half_pairs(select_half_operand<2, false, false>(packed_half_23),
                                            select_half_operand<2, false, false>(packed_half_24));

            matrix_fragment_54 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                              select_half_operand<2, false, false>(matrix_fragment_51));

            packed_half_20 = add_half_pairs(select_half_operand<2, false, false>(packed_half_20),
                                            select_half_operand<2, false, false>(packed_half_21));

            matrix_fragment_55 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_27),
                              select_half_operand<2, false, false>(matrix_fragment_50));

            packed_half_26 = add_half_pairs(select_half_operand<2, false, false>(packed_half_26),
                                            select_half_operand<2, false, false>(packed_half_105));

            load_four_shared_words<CheckBounds>(
                shared_tile, packed_half_120 + 0x400u, matrix_fragment_48, matrix_fragment_49,
                matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

            packed_half_27 = add_half_pairs(select_half_operand<2, false, false>(packed_half_27),
                                            select_half_operand<2, false, false>(packed_half_104));

            packed_half_10 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                               select_half_operand<2, false, false>(matrix_fragment_19));

            multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_72, matrix_fragment_73,
                                     matrix_accumulator_90, matrix_accumulator_91);

            packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                            select_half_operand<2, false, false>(packed_half_20));

            matrix_fragment_16 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                               select_half_operand<2, false, false>(packed_half_26));

            packed_half_23 = add_half_pairs(select_half_operand<2, false, false>(packed_half_23),
                                            select_half_operand<2, false, false>(packed_half_27));

            multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_72, matrix_fragment_73,
                                     matrix_accumulator_84, matrix_accumulator_85);

            multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_72, matrix_fragment_73,
                                     matrix_accumulator_78, matrix_accumulator_79);

            multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81,
                                     matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                     matrix_fragment_71, matrix_fragment_74, matrix_fragment_75,
                                     matrix_accumulator_80, matrix_accumulator_81);

            multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_74, matrix_fragment_75,
                                     matrix_accumulator_82, matrix_accumulator_83);

            multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_74, matrix_fragment_75,
                                     matrix_accumulator_28, matrix_accumulator_29);

            multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_74, matrix_fragment_75,
                                     matrix_accumulator_76, matrix_accumulator_77);

            load_four_shared_words<CheckBounds>(
                shared_tile, packed_half_120 + 0x600u, matrix_fragment_72, matrix_fragment_73,
                matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

            multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31,
                                     matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                     matrix_fragment_71, matrix_fragment_56, matrix_fragment_57,
                                     matrix_accumulator_30, matrix_accumulator_31);

            multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89,
                                     matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                     matrix_fragment_71, matrix_fragment_58, matrix_fragment_59,
                                     matrix_accumulator_88, matrix_accumulator_89);

            multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_56, matrix_fragment_57,
                                     matrix_accumulator_86, matrix_accumulator_87);

            multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_56, matrix_fragment_57,
                                     matrix_accumulator_92, matrix_accumulator_93);

            multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_56, matrix_fragment_57,
                                     matrix_accumulator_100, matrix_accumulator_101);

            multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_58, matrix_fragment_59,
                                     matrix_accumulator_102, matrix_accumulator_103);

            multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_48, matrix_fragment_49, 0u, 0u);

            multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_48, matrix_fragment_49, 0u, 0u);

            matrix_fragment_53 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_52 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_48, matrix_fragment_49, 0u, 0u);

            matrix_fragment_53 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3dc23dc2u);

            matrix_fragment_52 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3dc23dc2u);

            matrix_fragment_52 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52), 0x3fe93fe9u);

            matrix_fragment_56 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_57 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_72, matrix_fragment_73, 0u, 0u);

            matrix_fragment_56 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56), 0x3dc23dc2u);

            matrix_fragment_57 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57), 0x3dc23dc2u);

            packed_half_130 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56), 0x3fe93fe9u);

            packed_half_131 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_57), 0x3fe93fe9u);

            packed_half_130 = (packed_half_130 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_54 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_131 = (packed_half_131 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_55 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_74, matrix_fragment_75, 0u, 0u);

            matrix_fragment_54 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3dc23dc2u);

            matrix_fragment_55 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3dc23dc2u);

            packed_half_134 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54), 0x3fe93fe9u);

            packed_half_135 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_55), 0x3fe93fe9u);

            packed_half_134 = (packed_half_134 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_68 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_68),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_135 = (packed_half_135 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_69 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_69),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_58, matrix_fragment_59,
                                     matrix_accumulator_94, matrix_accumulator_95);

            matrix_fragment_68 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_68), 0x3dc23dc2u);

            matrix_fragment_69 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_69), 0x3dc23dc2u);

            matrix_fragment_54 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_69), 0x3fe93fe9u);

            matrix_fragment_71 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_71),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_70 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_70),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_58, matrix_fragment_59,
                                     matrix_accumulator_98, matrix_accumulator_99);

            matrix_fragment_71 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_71), 0x3dc23dc2u);

            matrix_fragment_56 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_70), 0x3dc23dc2u);

            packed_half_136 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_71), 0x3fe93fe9u);

            matrix_fragment_56 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_56), 0x3fe93fe9u);

            packed_half_136 = (packed_half_136 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_50, matrix_fragment_51, 0u, 0u);

            multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_50, matrix_fragment_51, 0u, 0u);

            matrix_fragment_58 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_59 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_50, matrix_fragment_51, 0u, 0u);

            matrix_fragment_58 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58), 0x3dc23dc2u);

            matrix_fragment_59 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59), 0x3dc23dc2u);

            packed_half_129 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_58), 0x3fe93fe9u);

            packed_half_128 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_59), 0x3fe93fe9u);

            packed_half_129 = (packed_half_129 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_60 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_128 = (packed_half_128 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_61 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_50, matrix_fragment_51, 0u, 0u);

            matrix_fragment_60 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60), 0x3dc23dc2u);

            matrix_fragment_61 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61), 0x3dc23dc2u);

            packed_half_133 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_60), 0x3fe93fe9u);

            packed_half_132 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61), 0x3fe93fe9u);

            matrix_fragment_60 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                              select_half_operand<2, false, false>(0u));

            packed_half_133 = (packed_half_133 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_62 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_62),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_132 = (packed_half_132 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_63 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_63),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_36,
                                     matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                     matrix_fragment_72, matrix_fragment_73, 0u, 0u);

            matrix_fragment_62 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_62), 0x3dc23dc2u);

            matrix_fragment_63 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_63), 0x3dc23dc2u);

            packed_half_141 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_62), 0x3fe93fe9u);

            packed_half_142 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_63), 0x3fe93fe9u);

            packed_half_141 = (packed_half_141 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_66 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_66),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_142 = (packed_half_142 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_67 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_67),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_72, matrix_fragment_73, 0u, 0u);

            matrix_fragment_66 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_66), 0x3dc23dc2u);

            matrix_fragment_62 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                              select_half_operand<2, false, false>(0u));

            packed_half_148 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_66), 0x3fe93fe9u);

            matrix_fragment_61 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                              select_half_operand<2, false, false>(matrix_fragment_62));

            packed_half_135 = add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                             select_half_operand<2, false, false>(packed_half_142));

            packed_half_148 = (packed_half_148 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_64 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_64),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_65 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_65),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_32,
                                     matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_74, matrix_fragment_75, 0u, 0u);

            matrix_fragment_64 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_64), 0x3dc23dc2u);

            packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                            select_half_operand<2, false, false>(packed_half_135));

            matrix_fragment_65 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_65), 0x3dc23dc2u);

            matrix_fragment_50 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_50),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_51 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_51),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(packed_half_122, packed_half_123, matrix_fragment_40,
                                     matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                     matrix_fragment_74, matrix_fragment_75, 0u, 0u);

            matrix_fragment_50 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_50), 0x3dc23dc2u);

            matrix_fragment_51 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_51), 0x3dc23dc2u);

            packed_half_137 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_50), 0x3fe93fe9u);

            packed_half_138 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_51), 0x3fe93fe9u);

            matrix_fragment_51 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                              select_half_operand<2, false, false>(0u));

            packed_half_137 = (packed_half_137 << 0x4u) + 0x3ffc4000u;

            packed_half_120 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_120),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_138 = (packed_half_138 << 0x4u) + 0x3ffc4000u;

            packed_half_121 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_121),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_48, matrix_fragment_49, 0u, 0u);

            packed_half_120 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_120), 0x3dc23dc2u);

            packed_half_121 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_121), 0x3dc23dc2u);

            matrix_fragment_55 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_120), 0x3fe93fe9u);

            matrix_fragment_57 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_121), 0x3fe93fe9u);

            packed_half_120 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_64), 0x3fe93fe9u);

            packed_half_121 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_65), 0x3fe93fe9u);

            packed_half_122 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_122),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_123 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_123),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_120 = (packed_half_120 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_72, matrix_fragment_73, 0u, 0u);

            packed_half_122 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_122), 0x3dc23dc2u);

            packed_half_123 = maximum_half_pairs(
                select_half_operand<2, false, false>(packed_half_123), 0x3dc23dc2u);

            packed_half_140 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_122), 0x3fe93fe9u);

            packed_half_122 = (matrix_fragment_52 << 0x4u) + 0x3ffc4000u;

            packed_half_143 = minimum_half_pairs(
                select_half_operand<2, false, false>(packed_half_123), 0x3fe93fe9u);

            packed_half_140 = (packed_half_140 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_48 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_52 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                              select_half_operand<2, false, false>(matrix_fragment_51));

            matrix_fragment_49 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_123 = (matrix_fragment_56 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_44,
                                     matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                     matrix_fragment_74, matrix_fragment_75, 0u, 0u);

            matrix_fragment_48 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48), 0x3dc23dc2u);

            packed_half_122 = add_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                             select_half_operand<2, false, false>(packed_half_129));

            matrix_fragment_49 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49), 0x3dc23dc2u);

            packed_half_139 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48), 0x3fe93fe9u);

            packed_half_110 = add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_122));

            packed_half_144 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49), 0x3fe93fe9u);

            matrix_fragment_48 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_67), 0x3dc23dc2u);

            matrix_fragment_51 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_72 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_72),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_48 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_48), 0x3fe93fe9u);

            matrix_fragment_73 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_73),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_72 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_72), 0x3dc23dc2u);

            matrix_fragment_73 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_73), 0x3dc23dc2u);

            packed_half_146 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_72), 0x3fe93fe9u);

            packed_half_145 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_73), 0x3fe93fe9u);

            matrix_fragment_73 = (matrix_fragment_54 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_54 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_74 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_74),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_147 = (matrix_fragment_48 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_75 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_75),
                select_half_operand<0, false, false>(packed_half_127), 0x3ed63ed6u);

            packed_half_127 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53), 0x3fe93fe9u);

            matrix_fragment_53 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_68), 0x3fe93fe9u);

            load_four_shared_words<CheckBounds>(
                shared_tile, tile_word_126 + 0x400u, matrix_fragment_68, matrix_fragment_69,
                matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

            matrix_fragment_49 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_75), 0x3dc23dc2u);

            matrix_fragment_50 = maximum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_74), 0x3dc23dc2u);

            matrix_fragment_75 = (matrix_fragment_55 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_74 = (matrix_fragment_57 << 0x4u) + 0x3ffc4000u;

            packed_half_127 = (packed_half_127 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_72 = (matrix_fragment_53 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_55 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_58 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_53 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_127),
                              select_half_operand<2, false, false>(matrix_fragment_54));

            packed_half_127 = add_half_pairs(select_half_operand<2, false, false>(packed_half_127),
                                             select_half_operand<2, false, false>(packed_half_128));

            matrix_fragment_50 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_50), 0x3fe93fe9u);

            matrix_fragment_49 = minimum_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_49), 0x3fe93fe9u);

            packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_109),
                                             select_half_operand<2, false, false>(packed_half_127));

            matrix_fragment_54 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                              select_half_operand<2, false, false>(matrix_fragment_55));

            matrix_fragment_72 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                               select_half_operand<2, false, false>(matrix_fragment_75));

            matrix_fragment_55 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                              select_half_operand<2, false, false>(matrix_fragment_58));

            matrix_fragment_73 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                               select_half_operand<2, false, false>(matrix_fragment_74));

            matrix_fragment_58 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_132),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_72 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                               select_half_operand<2, false, false>(matrix_fragment_72));

            matrix_fragment_59 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_73 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_109),
                               select_half_operand<2, false, false>(matrix_fragment_73));

            packed_half_121 = (packed_half_121 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_56 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                              select_half_operand<2, false, false>(matrix_fragment_51));

            packed_half_130 = add_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                             select_half_operand<2, false, false>(packed_half_133));

            matrix_fragment_62 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                              select_half_operand<2, false, false>(matrix_fragment_64));

            packed_half_137 = add_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                             select_half_operand<2, false, false>(packed_half_140));

            packed_half_139 = (packed_half_139 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                               select_half_operand<2, false, false>(packed_half_130));

            packed_half_143 = (packed_half_143 << 0x4u) + 0x3ffc4000u;

            multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_68, matrix_fragment_69,
                                     matrix_accumulator_96, matrix_accumulator_97);

            matrix_fragment_51 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                              select_half_operand<2, false, false>(0u));

            packed_half_144 = (packed_half_144 << 0x4u) + 0x3ffc4000u;

            packed_half_149 = (matrix_fragment_50 << 0x4u) + 0x3ffc4000u;

            packed_half_150 = (matrix_fragment_49 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_65 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_57 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                              select_half_operand<2, false, false>(matrix_fragment_58));

            packed_half_131 = add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                             select_half_operand<2, false, false>(packed_half_132));

            matrix_fragment_58 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_120),
                              select_half_operand<2, false, false>(matrix_fragment_59));

            packed_half_120 = add_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                             select_half_operand<2, false, false>(packed_half_123));

            matrix_fragment_59 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_121),
                              select_half_operand<2, false, false>(matrix_fragment_60));

            address_word_9 = add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                                            select_half_operand<2, false, false>(packed_half_131));

            matrix_fragment_63 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                              select_half_operand<2, false, false>(0u));

            packed_half_121 = add_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                             select_half_operand<2, false, false>(packed_half_136));

            matrix_fragment_60 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_134),
                              select_half_operand<2, false, false>(matrix_fragment_51));

            packed_half_134 = add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                             select_half_operand<2, false, false>(packed_half_141));

            packed_half_146 = (packed_half_146 << 0x4u) + 0x3ffc4000u;

            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                               select_half_operand<2, false, false>(packed_half_120));

            packed_half_145 = (packed_half_145 << 0x4u) + 0x3ffc4000u;

            packed_half_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                            select_half_operand<2, false, false>(packed_half_134));

            matrix_fragment_49 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                              select_half_operand<2, false, false>(0u));

            address_word_9 = add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                                            select_half_operand<2, false, false>(packed_half_121));

            matrix_fragment_48 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                              select_half_operand<2, false, false>(0u));

            packed_half_137 = add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                             select_half_operand<2, false, false>(packed_half_137));

            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                              select_half_operand<2, false, false>(matrix_fragment_64));

            packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                             select_half_operand<2, false, false>(packed_half_148));

            matrix_fragment_65 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                              select_half_operand<2, false, false>(matrix_fragment_65));

            packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                             select_half_operand<2, false, false>(packed_half_147));

            matrix_fragment_63 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                              select_half_operand<2, false, false>(matrix_fragment_63));

            matrix_fragment_16 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                               select_half_operand<2, false, false>(packed_half_139));

            matrix_fragment_66 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                              select_half_operand<2, false, false>(matrix_fragment_49));

            packed_half_23 = add_half_pairs(select_half_operand<2, false, false>(packed_half_23),
                                            select_half_operand<2, false, false>(packed_half_144));

            matrix_fragment_67 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                              select_half_operand<2, false, false>(matrix_fragment_48));

            packed_half_138 = add_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                             select_half_operand<2, false, false>(packed_half_143));

            matrix_fragment_48 = index_word_8 * 0x4u;

            packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                             select_half_operand<2, false, false>(packed_half_149));

            multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91,
                                     matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                     matrix_fragment_59, matrix_fragment_68, matrix_fragment_69,
                                     matrix_accumulator_90, matrix_accumulator_91);

            packed_half_145 = add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                             select_half_operand<2, false, false>(packed_half_150));

            packed_half_10 = condition_0 ? matrix_fragment_73 : matrix_fragment_72;

            packed_half_138 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                             select_half_operand<2, false, false>(packed_half_138));

            packed_half_11 = condition_0 ? address_word_9 : matrix_fragment_12;

            matrix_fragment_16 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                               select_half_operand<2, false, false>(packed_half_146));

            address_word_9 = condition_0 ? matrix_fragment_12 : address_word_9;

            packed_half_23 = add_half_pairs(select_half_operand<2, false, false>(packed_half_23),
                                            select_half_operand<2, false, false>(packed_half_145));

            matrix_fragment_12 = condition_0 ? packed_half_138 : packed_half_137;

            matrix_fragment_72 = condition_0 ? matrix_fragment_72 : matrix_fragment_73;

            packed_half_137 = condition_0 ? packed_half_137 : packed_half_138;

            matrix_fragment_13 = !condition_1 ? packed_half_10 : packed_half_11;

            matrix_fragment_15 = condition_0 ? packed_half_23 : matrix_fragment_16;

            matrix_fragment_16 = condition_0 ? matrix_fragment_16 : packed_half_23;

            packed_half_10 = !condition_1 ? packed_half_11 : packed_half_10;

            packed_half_11 = !condition_1 ? matrix_fragment_72 : address_word_9;

            multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_68, matrix_fragment_69,
                                     matrix_accumulator_84, matrix_accumulator_85);

            matrix_fragment_19 = !condition_1 ? matrix_fragment_12 : matrix_fragment_15;

            matrix_fragment_18 = !condition_1 ? matrix_fragment_15 : matrix_fragment_12;

            packed_half_21 = !condition_1 ? packed_half_137 : matrix_fragment_16;

            address_word_9 = !condition_1 ? address_word_9 : matrix_fragment_72;

            packed_half_137 = !condition_1 ? matrix_fragment_16 : packed_half_137;

            multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_68, matrix_fragment_69,
                                     matrix_accumulator_78, matrix_accumulator_79);

            matrix_fragment_69 = matrix_fragment_48 & 0x1cu;

            load_four_shared_words<CheckBounds>(
                shared_tile, tile_word_126 + 0x600u, matrix_fragment_48, matrix_fragment_49,
                matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

            index_word_8 = uint32_t(int64_t(int32_t(index_word_8)) * (1ll << 0x1du) >> 32) +
                           matrix_fragment_69;

            matrix_fragment_12 = 0x1u ^ index_word_8;

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_13), "r"(index_word_8), "r"(0x1fu), "r"(__activemask()));
                matrix_fragment_13 = result_word;
            }

            matrix_fragment_15 = 0x2u ^ index_word_8;

            multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_70, matrix_fragment_71,
                                     matrix_accumulator_80, matrix_accumulator_81);

            matrix_fragment_16 = 0x3u ^ index_word_8;

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_11), "r"(matrix_fragment_12), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_14 = result_word;
            }

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_19), "r"(index_word_8), "r"(0x1fu), "r"(__activemask()));
                matrix_fragment_19 = result_word;
            }

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_10), "r"(matrix_fragment_15), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_10 = result_word;
            }

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(address_word_9), "r"(matrix_fragment_16), "r"(0x1fu),
                      "r"(__activemask()));
                address_word_9 = result_word;
            }

            multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83,
                                     matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                     matrix_fragment_59, matrix_fragment_70, matrix_fragment_71,
                                     matrix_accumulator_82, matrix_accumulator_83);

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_18), "r"(matrix_fragment_15), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_18 = result_word;
            }

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_21), "r"(matrix_fragment_12), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_20 = result_word;
            }

            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_137), "r"(matrix_fragment_16), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_137 = result_word;
            }

            matrix_fragment_17 = !condition_3 ? matrix_fragment_13 : matrix_fragment_14;

            matrix_fragment_13 = !condition_3 ? matrix_fragment_14 : matrix_fragment_13;

            multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_70, matrix_fragment_71,
                                     matrix_accumulator_28, matrix_accumulator_29);

            index_word_8 = !condition_3 ? packed_half_10 : address_word_9;

            packed_half_10 = !condition_3 ? address_word_9 : packed_half_10;

            address_word_9 = !condition_4 ? matrix_fragment_17 : index_word_8;

            matrix_fragment_12 = !condition_4 ? matrix_fragment_13 : packed_half_10;

            index_word_8 = !condition_4 ? index_word_8 : matrix_fragment_17;

            packed_half_11 = !condition_3 ? matrix_fragment_19 : packed_half_20;

            matrix_fragment_19 = !condition_3 ? packed_half_20 : matrix_fragment_19;

            packed_half_10 = !condition_4 ? packed_half_10 : matrix_fragment_13;

            address_word_9 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                               select_half_operand<2, false, false>(matrix_fragment_12));

            matrix_fragment_14 = !condition_3 ? matrix_fragment_18 : packed_half_137;

            matrix_fragment_18 = !condition_3 ? packed_half_137 : matrix_fragment_18;

            matrix_fragment_13 = index_word_117 + 0x2u;

            address_word_9 = add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                                            select_half_operand<2, false, false>(index_word_8));

            multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_70, matrix_fragment_71,
                                     matrix_accumulator_76, matrix_accumulator_77);

            matrix_fragment_15 = !condition_4 ? packed_half_11 : matrix_fragment_14;

            matrix_fragment_16 = !condition_4 ? matrix_fragment_19 : matrix_fragment_18;

            packed_half_11 = !condition_4 ? matrix_fragment_14 : packed_half_11;

            matrix_fragment_18 = !condition_4 ? matrix_fragment_18 : matrix_fragment_19;

            condition_0 = int32_t(matrix_fragment_13) >= int32_t(index_word_118);

            address_word_9 = add_half_pairs(select_half_operand<2, false, false>(address_word_9),
                                            select_half_operand<2, false, false>(packed_half_10));

            matrix_fragment_15 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                               select_half_operand<2, false, false>(matrix_fragment_16));

            address_word_9 = add_half_pairs(select_half_operand<0, false, false>(address_word_9),
                                            select_half_operand<1, false, false>(address_word_9));

            matrix_fragment_15 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                               select_half_operand<2, false, false>(packed_half_11));

            matrix_fragment_15 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                               select_half_operand<2, false, false>(matrix_fragment_18));

            multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_48, matrix_fragment_49,
                                     matrix_accumulator_30, matrix_accumulator_31);

            index_word_8 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_15),
                                          select_half_operand<1, false, false>(matrix_fragment_15));

            index_word_8 = permute_signed(address_word_9, index_word_8, 0x5410u);

            packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                             select_half_operand<2, false, false>(index_word_8));

            multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89,
                                     matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                     matrix_fragment_55, matrix_fragment_50, matrix_fragment_51,
                                     matrix_accumulator_88, matrix_accumulator_89);

            multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87,
                                     matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                     matrix_fragment_59, matrix_fragment_48, matrix_fragment_49,
                                     matrix_accumulator_86, matrix_accumulator_87);

            multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95,
                                     matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                     matrix_fragment_59, matrix_fragment_50, matrix_fragment_51,
                                     matrix_accumulator_94, matrix_accumulator_95);

            multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_48, matrix_fragment_49,
                                     matrix_accumulator_92, matrix_accumulator_93);

            multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99,
                                     matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                     matrix_fragment_63, matrix_fragment_50, matrix_fragment_51,
                                     matrix_accumulator_98, matrix_accumulator_99);

            multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_48, matrix_fragment_49,
                                     matrix_accumulator_100, matrix_accumulator_101);

            multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103,
                                     matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                     matrix_fragment_67, matrix_fragment_50, matrix_fragment_51,
                                     matrix_accumulator_102, matrix_accumulator_103);
            if (!condition_0) {
                index_word_8 = threadIdx.x;

                address_word_4 = uint32_t(uint32_t(matrix_fragment_13) >> 0x1u);

                packed_half_10 = index_word_117 & 0x1u;

                matrix_fragment_15 = threadIdx.y;

                index_word_5 = address_word_4 & 0x1ffffffu;

                address_word_9 = threadIdx.z;

                index_word_5 = index_word_5 + 0x1u;

                matrix_fragment_14 = blockIdx.x;

                packed_half_11 = min(int32_t(index_word_5), int32_t(index_word_124));

                condition_0 = (address_word_9 | index_word_8 | matrix_fragment_15) != 0;

                condition_0 = int32_t(address_word_4) >= int32_t(packed_half_11) || condition_0;
                if (!condition_0) {
                    index_word_8 = address_word_4;
                    do {
                        address_word_4 = parameters[parameter_wait_low_10];
                        index_word_5 = parameters[parameter_wait_high_11];

                        address_word_9 = uint32_t(uint32_t(matrix_fragment_14) >> 0x1u);

                        address_word_9 = index_word_8 * 0x10u + address_word_9;

                        split_address_words(address_word_4, index_word_5,
                                            uint64_t(uint32_t(address_word_9)) *
                                                    uint64_t(uint32_t(0x4u)) +
                                                combine_words(address_word_4, index_word_5));

                        address_word_9 =
                            pipeline_read<CheckBounds>(combine_words(address_word_4, index_word_5),
                                                       memory_ranges, error_counts);

                        index_word_8 = index_word_8 + 0x1u;

                        condition_1 = int32_t(index_word_8) >= int32_t(packed_half_11);

                        condition_0 = int32_t(address_word_9) > -1;
                        if (!condition_0) {
                            do {
                                __nanosleep(0x40u);

                                address_word_9 = pipeline_read<CheckBounds>(
                                    combine_words(address_word_4, index_word_5), memory_ranges,
                                    error_counts);

                                condition_0 = int32_t(address_word_9) > -1;
                            } while (!condition_0);
                        }
                    } while (!condition_1);
                }
                __syncwarp(__activemask());

                uniform_index_word_5 = 0u;

                address_word_9 = matrix_fragment_13 * 0x4u + matrix_fragment_15;

                uniform_address_word_4 = 0x400u;

                phoenix_architecture::finish_compatibility_copies();

                address_word_4 = uniform_address_word_4;

                index_word_8 = address_word_9 * 0x20u + matrix_fragment_14;

                condition_1 = int32_t(address_word_9) >= int32_t(index_word_0);

                condition_0 = int32_t(address_word_9) >= int32_t(index_word_0);

                matrix_fragment_12 = address_word_4 + 0x2000u;

                split_address_words(index_word_8, address_word_9,
                                    int64_t(int32_t(index_word_8)) * int64_t(int32_t(0x80u)) +
                                        0ull);

                index_word_5 = uniform_index_word_5;

                packed_half_11 = (index_word_5 << 0x18u) + matrix_fragment_12;

                matrix_fragment_12 = packed_half_10 * 0x8u + packed_half_11;

                split_address_words(packed_half_10, packed_half_11,
                                    !condition_0 ? combine_words(index_word_8, address_word_9)
                                                 : combine_words(0u, 0u));

                address_word_9 = matrix_fragment_15 * 0x80u;
                if (condition_1) {
                    index_word_8 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    packed_half_11 = matrix_fragment_15 * 0x200u + index_word_6;

                    matrix_fragment_16 = uniform_address_word_4;

                    matrix_fragment_17 = uniform_address_word_4;

                    matrix_fragment_16 =
                        permute_signed(matrix_fragment_16, matrix_fragment_17, 0x5410u);

                    matrix_fragment_17 = matrix_fragment_16;

                    matrix_fragment_18 = matrix_fragment_16;

                    matrix_fragment_19 = matrix_fragment_16;

                    index_word_6 = index_word_8 * 0x10u + packed_half_11;

                    store_four_shared_words<CheckBounds>(shared_tile, index_word_6,
                                                         matrix_fragment_16, matrix_fragment_17,
                                                         matrix_fragment_18, matrix_fragment_19,
                                                         shared_bytes, memory_ranges, error_counts);
                } else {
                    uniform_address_word_4 = 0xffffffffu;

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        matrix_fragment_16 = parameters[parameter_input_k_low_2];
                        matrix_fragment_17 = parameters[parameter_input_k_high_3];

                        uniform_address_word_6 = index_word_6;

                        uniform_address_word_4 = matrix_fragment_15;

                        uniform_index_word_5 = matrix_fragment_12;

                        uniform_address_word_4 =
                            (uniform_address_word_4 << 0x9u) + uniform_address_word_6;

                        uniform_address_word_6 = 0x20u;

                        const uint64_t byte_address_2 = scaled_word_address(
                            combine_words(matrix_fragment_16, matrix_fragment_17),
                            combine_words(packed_half_10, packed_half_11));
                        index_word_6 = uint32_t(byte_address_2);

                        packed_half_10 = uint32_t(byte_address_2 >> 32);

                        uniform_address_word_10 = index_word_6;

                        uniform_address_word_11 = packed_half_10;

                        packed_half_11 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_10, uniform_address_word_11),
                            uniform_address_word_6 * 16u, matrix_fragment_12, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_10, uniform_address_word_11),
                            uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
                index_word_6 = uint32_t(uint32_t(matrix_fragment_15) >> 0x1u);

                condition_1 = uint32_t(index_word_3) != uint32_t(0x20u);

                packed_half_11 = matrix_fragment_13 * 0x2u + index_word_6;

                condition_0 = int32_t(packed_half_11) < int32_t(address_word_2) || !condition_1;
                if (!condition_0) {
                    index_word_6 = threadIdx.x & 31u;

                    uniform_address_word_4 =
                        pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));

                    index_word_7 = matrix_fragment_15 * 0x200u + index_word_7;

                    index_word_8 = uniform_address_word_4;

                    address_word_9 = uniform_address_word_4;

                    index_word_8 = permute_signed(index_word_8, address_word_9, 0x5410u);

                    address_word_9 = index_word_8;

                    packed_half_10 = index_word_8;

                    packed_half_11 = index_word_8;

                    index_word_6 = index_word_6 * 0x10u + index_word_7;

                    store_four_shared_words<CheckBounds>(
                        shared_tile, index_word_6, index_word_8, address_word_9, packed_half_10,
                        packed_half_11, shared_bytes, memory_ranges, error_counts);
                } else {
                    packed_half_11 = packed_half_11 * 0x2000u;

                    address_word_9 = address_word_9 & 0x80u;

                    uniform_address_word_4 = 0xffffffffu;

                    packed_half_11 = condition_1 ? packed_half_11 : 0u;

                    index_word_8 = matrix_fragment_14 * 0x100u + address_word_9;

                    index_word_8 = packed_half_11 + index_word_8;

                    address_word_9 = uint32_t(int32_t(index_word_8) >> 0x1fu);

                    split_address_words(packed_half_10, packed_half_11,
                                        condition_0 ? combine_words(index_word_8, address_word_9)
                                                    : combine_words(0u, 0u));

                    condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                    if (condition_0) {
                        index_word_8 = parameters[parameter_input_v_low_4];
                        address_word_9 = parameters[parameter_input_v_high_5];

                        uniform_address_word_4 = matrix_fragment_15;

                        uniform_address_word_6 = index_word_7;

                        uniform_index_word_5 = matrix_fragment_12;

                        index_word_7 = 0x200u;

                        uniform_address_word_4 =
                            (uniform_address_word_4 << 0x9u) + uniform_address_word_6;

                        uniform_address_word_6 = 0x20u;

                        const uint64_t byte_address_3 =
                            scaled_word_address(combine_words(index_word_8, address_word_9),
                                                combine_words(packed_half_10, packed_half_11));
                        index_word_6 = uint32_t(byte_address_3);

                        packed_half_10 = uint32_t(byte_address_3 >> 32);

                        uniform_address_word_10 = index_word_6;

                        uniform_address_word_11 = packed_half_10;

#if DLSS5_ENABLE_ASYNC_COPIES
                        copy_global_to_shared<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_10, uniform_address_word_11),
                            uniform_address_word_6 * 16u, matrix_fragment_12, shared_bytes,
                            memory_ranges, error_counts);
#else
                        bulk_copy<CheckBounds>(
                            shared_tile, uniform_address_word_4,
                            combine_words(uniform_address_word_10, uniform_address_word_11),
                            uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                            error_counts);
#endif
                    }
                }
            }
            index_word_117 = index_word_117 + 0x1u;

            condition_0 = int32_t(index_word_117) >= int32_t(index_word_118);
            if (!condition_0) {
                index_word_6 = address_word_4 + 0x2000u;

                address_word_4 = index_word_117 * 0x8u;

                index_word_7 = (index_word_5 << 0x18u) + index_word_6;

                address_word_4 = address_word_4 & 0x8u;

                index_word_7 = index_word_7 + address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                {
                    const uint32_t operand_a = index_word_7;
                    async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
                    split_address_words(address_word_4, index_word_5,
                                        async_tokens[operand_a >> 3 & 1]);
                }
#else
                address_word_4 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
                condition_0 = wait_copy_barrier(shared_tile, index_word_7,
                                                async_tokens[(((index_word_7) >> 3) & 1)]);
#else
                phoenix_architecture::finish_compatibility_copies();
                condition_0 = true;
#endif
                if (!condition_0) {
                    do {
#if DLSS5_ENABLE_ASYNC_COPIES
                        condition_0 = wait_copy_barrier(shared_tile, index_word_7,
                                                        async_tokens[(((index_word_7) >> 3) & 1)]);
#else
                        phoenix_architecture::finish_compatibility_copies();
                        condition_0 = true;
#endif
                    } while (!condition_0);
                }
            }
            condition_0 = uint32_t(index_word_117) != uint32_t(index_word_118);
        } while (condition_0);
    }
    address_word_2 = threadIdx.x & 31u;

    index_word_6 = address_word_2;

    address_word_9 = uint32_t(int32_t(index_word_6) >> 0x2u);

    address_word_4 = uint32_t(int32_t(index_word_6) >> 0x1fu);

    address_word_4 = uint32_t(combine_words(address_word_4, 0u) << 0x6u >> 32) + address_word_9;

    address_word_4 = address_word_4 & 0xffc0u;

    index_word_5 = address_word_9 + (0u - address_word_4);

    address_word_4 = index_word_118 * 0x40u + (0u - index_word_125);

    index_word_3 = permute_signed(index_word_5, 0u, 0x8880u);

    condition_0 = int32_t(address_word_4) >= int32_t(0x1u);

    index_word_3 = permute_signed(index_word_3, 0u, 0x7710u);

    index_word_3 = uint32_t(uint32_t(index_word_3) >> 0xau);

    index_word_8 = index_word_3 & 0x1fu;
    if (condition_0) {
        packed_half_119 = subtract_attention_tail(packed_half_119, address_word_4);
    }
    index_word_8 = index_word_5 + index_word_8;

    packed_half_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_119), 0x4100410u);

    uniform_address_word_4 = 0xffffffffu;

    address_word_4 = index_word_8 & 0xe0u;

    index_word_3 = half_to_float_bits((packed_half_119 & 65535u) * 65537u);

    index_word_8 = index_word_8 & 0xffffu;

    packed_half_119 = half_to_float_bits((packed_half_119 >> 16) * 65537u);

    packed_half_10 = 0u - address_word_4;

    index_word_3 = reciprocal_bits(index_word_3);

    packed_half_10 = permute_signed(packed_half_10, 0u, 0x7710u);

    address_word_4 = reciprocal_bits(packed_half_119);

    index_word_5 = index_word_5 + packed_half_10;

    index_word_5 = index_word_5 & 0xffffu;

    packed_half_11 = permute_signed(index_word_5, 0u, 0x8880u);

    index_word_5 = uint32_t(uint32_t(index_word_8) >> 0x5u);

    address_word_4 = pack_half(address_word_4, index_word_3);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        index_word_7 = result_word;
    }

    condition_6 = (index_word_5 & 0x1u) != 0;

    matrix_fragment_14 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x18u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_12 = result_word;
    }

    matrix_fragment_16 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x20u;

    matrix_fragment_17 = uint32_t(int32_t(matrix_fragment_14) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        index_word_3 = result_word;
    }

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_16) >> 0x1fu);

    matrix_fragment_17 =
        uint32_t(combine_words(matrix_fragment_17, 0u) << 0x6u >> 32) + matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_19 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x6u >> 32) + matrix_fragment_16;

    matrix_fragment_17 = matrix_fragment_17 & 0xffc0u;

    matrix_fragment_19 = matrix_fragment_19 & 0xffc0u;

    matrix_fragment_17 = matrix_fragment_14 + (0u - matrix_fragment_17);

    matrix_fragment_14 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x30u;

    matrix_fragment_19 = matrix_fragment_16 + (0u - matrix_fragment_19);

    packed_half_10 = permute_signed(index_word_7, packed_half_10, 0x7632u);

    matrix_fragment_16 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x38u;

    matrix_fragment_13 = permute_signed(matrix_fragment_12, matrix_fragment_13, 0x7632u);

    index_word_5 = condition_6 ? packed_half_10 : index_word_7;

    packed_half_21 = uint32_t(int32_t(matrix_fragment_14) >> 0x1fu);

    index_word_8 = permute_signed(index_word_3, index_word_8, 0x7632u);

    index_word_7 = condition_6 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_12 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x10u;

    index_word_3 = condition_6 ? index_word_8 : index_word_3;

    index_word_8 = address_word_9 + 0x8u;

    packed_half_11 = uint32_t(int32_t(matrix_fragment_12) >> 0x1fu);

    packed_half_10 = permute_signed(matrix_fragment_15, packed_half_10, 0x7632u);

    address_word_9 = uint32_t(int32_t(index_word_8) >> 0x1fu);

    matrix_fragment_13 =
        uint32_t(combine_words(packed_half_11, 0u) << 0x6u >> 32) + matrix_fragment_12;

    address_word_9 = uint32_t(combine_words(address_word_9, 0u) << 0x6u >> 32) + index_word_8;

    matrix_fragment_13 = matrix_fragment_13 & 0xffc0u;

    packed_half_11 = address_word_9 & 0xffc0u;

    address_word_9 = condition_6 ? packed_half_10 : matrix_fragment_15;

    matrix_fragment_13 = matrix_fragment_12 + (0u - matrix_fragment_13);

    matrix_fragment_12 = uint32_t(int64_t(int32_t(index_word_6)) * (1ll << 0x1eu) >> 32) + 0x28u;

    packed_half_11 = index_word_8 + (0u - packed_half_11);

    packed_half_21 = uint32_t(combine_words(packed_half_21, 0u) << 0x6u >> 32) + matrix_fragment_14;

    index_word_8 = permute_signed(matrix_fragment_13, 0u, 0x8880u);

    index_word_6 = permute_signed(packed_half_11, 0u, 0x8880u);

    index_word_8 = permute_signed(index_word_8, 0u, 0x7710u);

    index_word_6 = permute_signed(index_word_6, 0u, 0x7710u);

    matrix_fragment_15 = uint32_t(int32_t(matrix_fragment_12) >> 0x1fu);

    index_word_8 = uint32_t(uint32_t(index_word_8) >> 0xau);

    index_word_6 = uint32_t(uint32_t(index_word_6) >> 0xau);

    matrix_fragment_15 =
        uint32_t(combine_words(matrix_fragment_15, 0u) << 0x6u >> 32) + matrix_fragment_12;

    index_word_8 = index_word_8 & 0x1fu;

    index_word_6 = index_word_6 & 0x1fu;

    matrix_fragment_15 = matrix_fragment_15 & 0xffc0u;

    packed_half_21 = packed_half_21 & 0xffc0u;

    packed_half_10 = matrix_fragment_13 + index_word_8;

    packed_half_23 = uint32_t(int32_t(matrix_fragment_16) >> 0x1fu);

    index_word_6 = packed_half_11 + index_word_6;

    matrix_fragment_18 = matrix_fragment_12 + (0u - matrix_fragment_15);

    packed_half_21 = matrix_fragment_14 + (0u - packed_half_21);

    packed_half_23 = uint32_t(combine_words(packed_half_23, 0u) << 0x6u >> 32) + matrix_fragment_16;

    matrix_fragment_12 = packed_half_10 & 0xffffu;

    matrix_fragment_14 = permute_signed(matrix_fragment_17, 0u, 0x8880u);

    matrix_fragment_15 = permute_signed(matrix_fragment_19, 0u, 0x8880u);

    index_word_8 = index_word_6 & 0xffffu;

    packed_half_23 = packed_half_23 & 0xffc0u;

    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_12) >> 0x5u);

    matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_15 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    index_word_8 = uint32_t(uint32_t(index_word_8) >> 0x5u);

    condition_4 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_14) >> 0xau);

    matrix_fragment_15 = uint32_t(uint32_t(matrix_fragment_15) >> 0xau);

    packed_half_23 = matrix_fragment_16 + (0u - packed_half_23);

    matrix_fragment_12 = permute_signed(matrix_fragment_18, 0u, 0x8880u);

    condition_5 = (index_word_8 & 0x1u) != 0;

    matrix_fragment_14 = matrix_fragment_14 & 0x1fu;

    index_word_8 = matrix_fragment_15 & 0x1fu;

    index_word_6 = index_word_6 & 0xe0u;

    matrix_fragment_16 = permute_signed(packed_half_21, 0u, 0x8880u);

    matrix_fragment_15 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    packed_half_20 = permute_signed(packed_half_23, 0u, 0x8880u);

    packed_half_10 = packed_half_10 & 0xe0u;

    matrix_fragment_16 = permute_signed(matrix_fragment_16, 0u, 0x7710u);

    matrix_fragment_14 = matrix_fragment_17 + matrix_fragment_14;

    matrix_fragment_12 = matrix_fragment_19 + index_word_8;

    matrix_fragment_15 = uint32_t(uint32_t(matrix_fragment_15) >> 0xau);

    index_word_6 = 0u - index_word_6;

    index_word_8 = permute_signed(packed_half_20, 0u, 0x7710u);

    packed_half_20 = uint32_t(uint32_t(matrix_fragment_16) >> 0xau);

    matrix_fragment_15 = matrix_fragment_15 & 0x1fu;

    packed_half_10 = 0u - packed_half_10;

    matrix_fragment_16 = matrix_fragment_14 & 0xffffu;

    packed_half_22 = uint32_t(uint32_t(index_word_8) >> 0xau);

    index_word_6 = permute_signed(index_word_6, 0u, 0x7710u);

    matrix_fragment_14 = matrix_fragment_14 & 0xe0u;

    index_word_8 = uint32_t(uint32_t(matrix_fragment_16) >> 0x5u);

    packed_half_20 = packed_half_20 & 0x1fu;

    packed_half_22 = packed_half_22 & 0x1fu;

    matrix_fragment_15 = matrix_fragment_18 + matrix_fragment_15;

    packed_half_10 = permute_signed(packed_half_10, 0u, 0x7710u);

    packed_half_24 = matrix_fragment_12 & 0xffffu;

    packed_half_11 = packed_half_11 + index_word_6;

    index_word_6 = 0u - matrix_fragment_14;

    condition_3 = (index_word_8 & 0x1u) != 0;

    matrix_fragment_16 = uint32_t(uint32_t(packed_half_24) >> 0x5u);

    packed_half_20 = packed_half_21 + packed_half_20;

    packed_half_22 = packed_half_23 + packed_half_22;

    index_word_8 = matrix_fragment_15 & 0xffffu;

    matrix_fragment_13 = matrix_fragment_13 + packed_half_10;

    matrix_fragment_15 = matrix_fragment_15 & 0xe0u;

    packed_half_11 = packed_half_11 & 0xffffu;

    index_word_6 = permute_signed(index_word_6, 0u, 0x7710u);

    condition_2 = (matrix_fragment_16 & 0x1u) != 0;

    index_word_8 = uint32_t(uint32_t(index_word_8) >> 0x5u);

    matrix_fragment_12 = matrix_fragment_12 & 0xe0u;

    matrix_fragment_16 = packed_half_20 & 0xffffu;

    packed_half_24 = packed_half_22 & 0xffffu;

    matrix_fragment_13 = matrix_fragment_13 & 0xffffu;

    packed_half_11 = permute_signed(packed_half_11, 0u, 0x8880u);

    packed_half_20 = packed_half_20 & 0xe0u;

    packed_half_22 = packed_half_22 & 0xe0u;

    matrix_fragment_15 = 0u - matrix_fragment_15;

    condition_1 = (index_word_8 & 0x1u) != 0;

    matrix_fragment_17 = matrix_fragment_17 + index_word_6;

    index_word_8 = 0u - matrix_fragment_12;

    index_word_6 = permute_signed(matrix_fragment_13, 0u, 0x8880u);

    matrix_fragment_13 = packed_half_11;

    packed_half_10 = 0u - packed_half_20;

    matrix_fragment_12 = 0u - packed_half_22;

    matrix_fragment_15 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    index_word_8 = permute_signed(index_word_8, 0u, 0x7710u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    packed_half_11 = index_word_6;

    packed_half_10 = permute_signed(packed_half_10, 0u, 0x7710u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        index_word_6 = result_word;
    }

    matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    matrix_fragment_18 = matrix_fragment_18 + matrix_fragment_15;

    matrix_fragment_17 = matrix_fragment_17 & 0xffffu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_19 = matrix_fragment_19 + index_word_8;

    packed_half_21 = packed_half_21 + packed_half_10;

    packed_half_23 = packed_half_23 + matrix_fragment_12;

    matrix_fragment_18 = matrix_fragment_18 & 0xffffu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        packed_half_10 = result_word;
    }

    matrix_fragment_17 = permute_signed(matrix_fragment_17, 0u, 0x8880u);

    matrix_fragment_19 = matrix_fragment_19 & 0xffffu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        packed_half_22 = result_word;
    }

    matrix_fragment_16 = uint32_t(uint32_t(matrix_fragment_16) >> 0x5u);

    packed_half_21 = packed_half_21 & 0xffffu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        packed_half_25 = result_word;
    }

    packed_half_23 = packed_half_23 & 0xffffu;

    matrix_fragment_33 = permute_signed(matrix_fragment_18, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    packed_half_27 = permute_signed(matrix_fragment_19, 0u, 0x8880u);

    condition_0 = (matrix_fragment_16 & 0x1u) != 0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    matrix_fragment_41 = permute_signed(packed_half_21, 0u, 0x8880u);

    matrix_fragment_43 = permute_signed(packed_half_23, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    packed_half_24 = uint32_t(uint32_t(packed_half_24) >> 0x5u);

    matrix_fragment_13 = permute_signed(packed_half_10, matrix_fragment_13, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        packed_half_21 = result_word;
    }

    condition_6 = (packed_half_24 & 0x1u) != 0;

    index_word_8 = permute_signed(matrix_fragment_15, index_word_8, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        packed_half_23 = result_word;
    }

    packed_half_10 = condition_5 ? matrix_fragment_13 : packed_half_10;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_27), "r"(0x1fu), "r"(__activemask()));
        packed_half_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_11), "r"(0x1fu), "r"(__activemask()));
        packed_half_24 = result_word;
    }

    matrix_fragment_17 = permute_signed(packed_half_22, matrix_fragment_17, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_33), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_32 = result_word;
    }

    packed_half_20 = permute_signed(matrix_fragment_19, packed_half_20, 0x7632u);

    matrix_fragment_17 = condition_4 ? matrix_fragment_17 : packed_half_22;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_27), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_34 = result_word;
    }

    matrix_fragment_13 = permute_signed(matrix_fragment_16, matrix_fragment_13, 0x7632u);

    packed_half_11 = permute_signed(index_word_6, packed_half_11, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_27), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_36 = result_word;
    }

    packed_half_22 = permute_signed(packed_half_25, packed_half_22, 0x7632u);

    matrix_fragment_13 = condition_4 ? matrix_fragment_13 : matrix_fragment_16;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_33), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    index_word_6 = condition_5 ? packed_half_11 : index_word_6;

    packed_half_11 = permute_signed(matrix_fragment_14, packed_half_11, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_33), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_37 = result_word;
    }

    matrix_fragment_19 = condition_4 ? packed_half_20 : matrix_fragment_19;

    packed_half_22 = condition_3 ? packed_half_22 : packed_half_25;

    matrix_fragment_12 = condition_5 ? packed_half_11 : matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_41), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    packed_half_11 = permute_signed(matrix_fragment_18, packed_half_11, 0x7632u);

    matrix_fragment_14 = condition_5 ? index_word_8 : matrix_fragment_15;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(packed_half_27), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    index_word_8 = permute_signed(packed_half_23, index_word_8, 0x7632u);

    matrix_fragment_18 = condition_3 ? packed_half_11 : matrix_fragment_18;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_33), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    packed_half_11 = permute_signed(packed_half_26, packed_half_11, 0x7632u);

    packed_half_20 = condition_3 ? index_word_8 : packed_half_23;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_41), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    matrix_fragment_15 = permute_signed(packed_half_24, matrix_fragment_15, 0x7632u);

    packed_half_23 = permute_signed(matrix_fragment_32, packed_half_23, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_41), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    packed_half_25 = permute_signed(matrix_fragment_34, packed_half_25, 0x7632u);

    packed_half_27 = permute_signed(matrix_fragment_36, packed_half_27, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_41), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    index_word_8 = permute_signed(matrix_fragment_37, index_word_8, 0x7632u);

    matrix_fragment_15 = condition_4 ? matrix_fragment_15 : packed_half_24;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_43), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_fragment_32 = condition_1 ? packed_half_23 : matrix_fragment_32;

    packed_half_25 = condition_2 ? packed_half_25 : matrix_fragment_34;

    packed_half_27 = condition_2 ? packed_half_27 : matrix_fragment_36;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_43), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    matrix_fragment_33 = permute_signed(matrix_fragment_38, matrix_fragment_33, 0x7632u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_43), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_fragment_34 = permute_signed(matrix_fragment_39, matrix_fragment_34, 0x7632u);

    matrix_fragment_33 = condition_2 ? matrix_fragment_33 : matrix_fragment_38;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_4), "r"(matrix_fragment_43), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_41 = permute_signed(matrix_fragment_44, matrix_fragment_41, 0x7632u);

    matrix_fragment_34 = condition_1 ? matrix_fragment_34 : matrix_fragment_39;

    matrix_fragment_41 = condition_0 ? matrix_fragment_41 : matrix_fragment_44;

    address_word_4 = permute_signed(packed_half_21, address_word_4, 0x7632u);

    matrix_fragment_16 = condition_3 ? address_word_4 : packed_half_21;

    packed_half_21 = condition_2 ? packed_half_11 : packed_half_26;

    address_word_4 = permute_signed(matrix_fragment_35, address_word_4, 0x7632u);

    packed_half_11 = permute_signed(matrix_fragment_40, packed_half_11, 0x7632u);

    packed_half_26 = condition_1 ? index_word_8 : matrix_fragment_37;

    packed_half_24 = condition_1 ? address_word_4 : matrix_fragment_35;

    matrix_fragment_35 = permute_signed(matrix_fragment_46, matrix_fragment_35, 0x7632u);

    packed_half_23 = condition_0 ? packed_half_11 : matrix_fragment_40;

    matrix_fragment_37 = permute_signed(matrix_fragment_48, matrix_fragment_37, 0x7632u);

    packed_half_11 = permute_signed(matrix_fragment_42, packed_half_11, 0x7632u);

    address_word_4 = permute_signed(matrix_fragment_45, address_word_4, 0x7632u);

    matrix_fragment_36 = permute_signed(matrix_fragment_47, matrix_fragment_36, 0x7632u);

    matrix_fragment_35 = condition_0 ? matrix_fragment_35 : matrix_fragment_46;

    matrix_fragment_37 = condition_0 ? matrix_fragment_37 : matrix_fragment_48;

    matrix_fragment_42 = condition_6 ? packed_half_11 : matrix_fragment_42;

    index_word_8 = condition_6 ? address_word_4 : matrix_fragment_45;

    matrix_fragment_36 = condition_6 ? matrix_fragment_36 : matrix_fragment_47;

    address_word_4 = threadIdx.y;

    packed_half_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<0, false, false>(packed_half_25));

    condition_4 = uint32_t(index_word_125) != 0u;

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<0, false, false>(matrix_fragment_12));

    packed_half_11 = blockIdx.y;

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(index_word_6));

    index_word_6 = permute_signed(matrix_fragment_49, index_word_6, 0x7632u);

    index_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<0, false, false>(index_word_5));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    index_word_7 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       select_half_operand<0, false, false>(index_word_7));

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<0, false, false>(packed_half_10));

    index_word_6 = condition_6 ? index_word_6 : matrix_fragment_49;

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<0, false, false>(matrix_fragment_15));

    index_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<0, false, false>(index_word_3));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<0, false, false>(matrix_fragment_17));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(index_word_6));

    index_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_5),
                                 select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<0, false, false>(matrix_fragment_14));

    index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_7),
                                 select_half_operand<2, false, false>(0u));

    packed_half_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<0, false, false>(packed_half_24));

    index_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                 select_half_operand<2, false, false>(0u));

    address_word_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<0, false, false>(address_word_9));

    index_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_3),
                                 select_half_operand<2, false, false>(0u));

    packed_half_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<0, false, false>(packed_half_20));

    matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

    packed_half_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<0, false, false>(packed_half_22));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<0, false, false>(matrix_fragment_32));

    packed_half_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_20),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<0, false, false>(matrix_fragment_16));

    packed_half_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<0, false, false>(packed_half_27));

    matrix_accumulator_28 = packed_half_11 * 0x4u + address_word_4;

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_accumulator_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<0, false, false>(matrix_fragment_13));

    matrix_accumulator_31 = matrix_accumulator_28 * 0x4u;

    if (condition_4) {
        matrix_accumulator_28 = index_word_125 + 0xffffffffu;
    }

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<0, false, false>(matrix_fragment_18));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<0, false, false>(matrix_fragment_19));

    matrix_fragment_39 = matrix_accumulator_31 + 0x1u;

    matrix_fragment_43 = matrix_accumulator_31 + 0x2u;

    matrix_fragment_45 = matrix_accumulator_31 + 0x3u;

    condition_0 = int32_t(matrix_accumulator_31) >= int32_t(index_word_0);

    if (condition_4) {
        matrix_fragment_47 = uint32_t(int32_t(matrix_accumulator_28) >> 0x1fu);
    }

    index_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<0, false, false>(index_word_8));

    condition_1 = int32_t(matrix_fragment_39) >= int32_t(index_word_0);

    condition_2 = int32_t(matrix_fragment_43) >= int32_t(index_word_0);

    condition_3 = int32_t(matrix_fragment_45) >= int32_t(index_word_0);

    if (condition_4) {
        matrix_fragment_47 =
            uint32_t(combine_words(matrix_fragment_47, 0u) << 0x5u >> 32) + matrix_accumulator_28;
    }

    index_word_0 = blockIdx.x;

    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(0u));

    packed_half_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<0, false, false>(packed_half_21));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    packed_half_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<0, false, false>(packed_half_26));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<0, false, false>(matrix_fragment_34));

    if (!condition_0) {
        matrix_fragment_17 = permute_signed(matrix_accumulator_97, packed_half_10, 0x5410u);
    }

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<0, false, false>(matrix_fragment_42));

    if (!condition_0) {
        packed_half_10 = permute_signed(index_word_3, index_word_6, 0x5410u);
    }

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(matrix_fragment_36));

    if (!condition_1) {
        matrix_accumulator_28 = permute_signed(matrix_accumulator_28, index_word_7, 0x5410u);
    }

    packed_half_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<0, false, false>(packed_half_23));

    index_word_6 = parameters[parameter_output_low_6];
    index_word_7 = parameters[parameter_output_high_7];

    if (!condition_0) {
        matrix_fragment_14 = permute_signed(matrix_fragment_12, matrix_accumulator_29, 0x5410u);
    }

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<0, false, false>(matrix_fragment_41));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_9),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    address_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_22),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(matrix_fragment_37));

    index_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                 select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_15 = permute_signed(packed_half_20, address_word_9, 0x5410u);
    }

    address_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_24),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_27),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        index_word_5 = permute_signed(index_word_5, matrix_fragment_12, 0x5410u);
    }

    matrix_accumulator_31 = matrix_accumulator_31 * 0x20u + index_word_0;

    if (!condition_2) {
        matrix_accumulator_30 = permute_signed(matrix_accumulator_30, address_word_9, 0x5410u);
    }

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    address_word_9 = matrix_accumulator_31 * 0x80u;

    if (!condition_2) {
        matrix_accumulator_29 = permute_signed(matrix_accumulator_29, matrix_fragment_16, 0x5410u);
    }

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(index_word_8),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(index_word_8, address_word_9,
                        int64_t(int32_t(address_word_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(index_word_6, index_word_7));

    if (!condition_1) {
        packed_half_24 = permute_signed(matrix_fragment_18, index_word_3, 0x5410u);

        packed_half_27 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x5410u);
    }

    packed_half_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_26),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    packed_half_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_21),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        packed_half_26 = permute_signed(packed_half_26, matrix_fragment_19, 0x5410u);

        packed_half_25 = permute_signed(packed_half_21, matrix_fragment_12, 0x5410u);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_8, address_word_9));
    }

    if (!condition_3) {
        matrix_accumulator_31 = permute_signed(matrix_accumulator_31, matrix_fragment_18, 0x5410u);

        matrix_fragment_33 = permute_signed(matrix_fragment_33, matrix_fragment_16, 0x5410u);
    }

    if (!condition_1) {
        split_address_words(packed_half_20, packed_half_21,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_8, address_word_9));
    }

    if (!condition_0) {
        matrix_fragment_16 = packed_half_10;

        matrix_fragment_18 = index_word_5;

        matrix_fragment_19 = matrix_fragment_14;
    }

    if (!condition_1) {
        matrix_fragment_14 = matrix_accumulator_28;
    }

    packed_half_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_23),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_fragment_34 = permute_signed(packed_half_23, matrix_fragment_34, 0x5410u);
    }

    if (!condition_2) {
        split_address_words(packed_half_22, packed_half_23,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_8, address_word_9));
    }

    if (!condition_3) {
        matrix_fragment_32 = permute_signed(matrix_fragment_32, matrix_fragment_37, 0x5410u);
    }

    if (condition_4) {
        matrix_fragment_47 = matrix_fragment_47 & 0xffffffe0u;
    }

    if (!condition_3) {
        split_address_words(address_word_2, index_word_3,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(index_word_8, address_word_9));
    }

    index_word_8 = 0u;

    if (condition_4) {
        index_word_8 = matrix_fragment_47 + 0x20u;
    }

    if (!condition_1) {
        matrix_fragment_12 = packed_half_27;

        matrix_fragment_13 = packed_half_24;
    }

    condition_0 = uint32_t(index_word_8) != uint32_t(index_word_125);

    if (!condition_1) {
        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_20, packed_half_21) + 0x4000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (!condition_2) {
        matrix_fragment_12 = packed_half_25;

        matrix_fragment_13 = matrix_accumulator_30;

        matrix_fragment_14 = matrix_accumulator_29;

        matrix_fragment_15 = packed_half_26;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_22, packed_half_23) + 0x8000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (!condition_3) {
        matrix_fragment_12 = matrix_fragment_34;

        matrix_fragment_13 = matrix_fragment_33;

        matrix_fragment_14 = matrix_fragment_32;

        matrix_fragment_15 = matrix_accumulator_31;

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_2, index_word_3) + 0xc000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    if (condition_0) {
        address_word_2 = index_word_125 + 0x7fu;

        __syncwarp(__activemask());

        index_word_3 = uint32_t(int32_t(address_word_2) >> 0x1fu);

        phoenix_architecture::finish_compatibility_copies();

        index_word_3 = uint32_t(combine_words(index_word_3, 0u) << 0x7u >> 32) + address_word_2;

        index_word_3 = uint32_t(int64_t(int32_t(index_word_3)) * (1ll << 0x19u) >> 32) + 0x1u;

        index_word_3 = uint32_t(combine_words(index_word_3, 0u) << 0x1u >> 32) + index_word_3;

        address_word_2 =
            uint32_t(int64_t(int32_t(index_word_3)) * (1ll << 0x1fu) >> 32) + 0xffffffffu;

        condition_0 = uint32_t(packed_half_11) != uint32_t(address_word_2);
        if (!condition_0) {
            address_word_9 = threadIdx.z;

            uniform_index_word_5 = blockDim.y;

            index_word_8 = 0u - index_word_125 + index_word_8;

            matrix_fragment_15 = threadIdx.x;

            uniform_address_word_6 = blockDim.x;

            index_word_3 = blockDim.z;

            packed_half_20 = index_word_8 * 0x8u;

            address_word_2 = address_word_9 * uniform_index_word_5 + address_word_4;

            index_word_5 = address_word_2 * uniform_address_word_6 + matrix_fragment_15;

            condition_0 = int32_t(index_word_5) >= int32_t(packed_half_20);
            if (!condition_0) {
                uniform_address_word_4 = uniform_index_word_5 * uniform_address_word_6;

                packed_half_10 = index_word_3 * uniform_address_word_4;

                index_word_3 = address_word_9 + index_word_3;

                matrix_fragment_13 = __float_as_uint(__uint2float_ru(packed_half_10));

                condition_2 = uint32_t(packed_half_10) != 0u;

                address_word_2 = index_word_3 * uniform_index_word_5 + address_word_4;

                index_word_8 = address_word_2 * uniform_address_word_6 + matrix_fragment_15;

                address_word_2 = 0u - packed_half_10;

                condition_0 = int32_t(index_word_8) >= int32_t(packed_half_20);

                address_word_9 = max(int32_t(packed_half_20), int32_t(index_word_8));

                matrix_fragment_13 = reciprocal_bits(matrix_fragment_13);

                matrix_fragment_12 = condition_0 ? 0u : 0x1u;

                address_word_9 = address_word_9 + (0u - index_word_8) + (0u - matrix_fragment_12);

                matrix_fragment_14 = matrix_fragment_13 + 0xffffffeu;

                index_word_3 = __float2uint_rz(__uint_as_float(matrix_fragment_14));

                matrix_fragment_13 = address_word_2 * index_word_3;

                address_word_2 = 0u;

                address_word_2 = uint32_t(uint64_t(index_word_3) * matrix_fragment_13 +
                                              combine_words(address_word_2, index_word_3) >>
                                          32);

                index_word_3 = uint32_t(uint64_t(address_word_2) * address_word_9 >> 32);

                address_word_2 = 0u - index_word_3;

                address_word_9 = packed_half_10 * address_word_2 + address_word_9;

                condition_0 = uint32_t(address_word_9) >= uint32_t(packed_half_10);

                if (condition_0) {
                    address_word_9 = 0u - packed_half_10 + address_word_9;

                    index_word_3 = index_word_3 + 0x1u;
                }

                condition_1 = uint32_t(address_word_9) >= uint32_t(packed_half_10);

                if (condition_1) {
                    index_word_3 = index_word_3 + 0x1u;
                }

                if (!condition_2) {
                    index_word_3 = ~packed_half_10;
                }

                index_word_3 = matrix_fragment_12 + index_word_3;

                condition_0 = (index_word_3 & 0x1u) != 0;

                condition_1 = uint32_t(index_word_3) != 0u;
                if (!condition_0) {
                    address_word_2 =
                        uint32_t(combine_words(index_word_5, 0u) << 0x1du >> 32) + index_word_125;

                    index_word_3 = uint32_t(int32_t(address_word_2) >> 0x1fu);

                    index_word_8 =
                        uint32_t(combine_words(index_word_3, 0u) << 0x4u >> 32) + address_word_2;

                    index_word_3 = index_word_8 & 0xfff0u;

                    address_word_9 = address_word_2 + (0u - index_word_3);

                    address_word_2 = permute_signed(address_word_9, 0u, 0x8880u);

                    address_word_2 = permute_signed(address_word_2, 0u, 0x7710u);

                    address_word_2 = uint32_t(uint32_t(address_word_2) >> 0xcu);

                    address_word_2 = address_word_2 & 0x7u;

                    index_word_3 = address_word_9 + address_word_2;

                    address_word_2 = index_word_5 * 0x4u;

                    matrix_fragment_12 = index_word_3 & 0xf8u;

                    matrix_fragment_13 = permute_signed(index_word_3, 0u, 0x8880u);

                    index_word_3 = address_word_2 & 0x1cu;

                    matrix_fragment_14 = 0u - matrix_fragment_12;

                    matrix_fragment_12 = matrix_fragment_13;

                    matrix_fragment_13 = index_word_8 * 0x100u;

                    address_word_2 = address_word_2 & 0xcu;

                    index_word_3 = index_word_0 * 0x20u + index_word_3;

                    matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

                    matrix_fragment_12 = uint32_t(int32_t(matrix_fragment_12) >> 0x3u);

                    index_word_8 = uint32_t(uint32_t(index_word_5) >> 0x1u);

                    address_word_9 = address_word_9 + matrix_fragment_14;

                    matrix_fragment_15 = permute_signed(matrix_fragment_12, 0u, 0x9910u);

                    matrix_fragment_14 = index_word_3 * 0x4u;

                    matrix_fragment_12 = matrix_fragment_13 & 0xfffff000u;

                    address_word_9 = permute_signed(address_word_9, 0u, 0x8880u);

                    index_word_3 = matrix_fragment_15;

                    matrix_fragment_14 = matrix_fragment_14 & 0xffffff80u;

                    index_word_8 = index_word_8 & 0x2u;

                    matrix_fragment_12 = index_word_3 + matrix_fragment_12 + matrix_fragment_14;

                    index_word_3 = address_word_9 * 0x10u + address_word_2;

                    index_word_5 = packed_half_10 + index_word_5;

                    index_word_3 = index_word_3 + matrix_fragment_12 + index_word_8;

                    split_address_words(address_word_2, index_word_3,
                                        int64_t(int32_t(index_word_3)) * int64_t(int32_t(0x4u)) +
                                            combine_words(index_word_6, index_word_7));

                    pipeline_write<CheckBounds>(combine_words(address_word_2, index_word_3), 0u,
                                                memory_ranges, error_counts);
                }
                if (condition_1) {
                    do {
                        matrix_fragment_13 = packed_half_10 + index_word_5;

                        address_word_2 = uint32_t(combine_words(index_word_5, 0u) << 0x1du >> 32) +
                                         index_word_125;

                        matrix_fragment_12 =
                            uint32_t(combine_words(matrix_fragment_13, 0u) << 0x1du >> 32) +
                            index_word_125;

                        index_word_3 = uint32_t(int32_t(address_word_2) >> 0x1fu);

                        address_word_9 = uint32_t(int32_t(matrix_fragment_12) >> 0x1fu);

                        index_word_8 = uint32_t(combine_words(index_word_3, 0u) << 0x4u >> 32) +
                                       address_word_2;

                        matrix_fragment_16 =
                            uint32_t(combine_words(address_word_9, 0u) << 0x4u >> 32) +
                            matrix_fragment_12;

                        index_word_3 = index_word_8 & 0xfff0u;

                        index_word_8 = index_word_8 * 0x100u;

                        matrix_fragment_15 = matrix_fragment_16 & 0xfff0u;

                        matrix_fragment_16 = matrix_fragment_16 * 0x100u;

                        address_word_9 = address_word_2 + (0u - index_word_3);

                        index_word_8 = index_word_8 & 0xfffff000u;

                        matrix_fragment_15 = matrix_fragment_12 + (0u - matrix_fragment_15);

                        matrix_fragment_16 = matrix_fragment_16 & 0xfffff000u;

                        address_word_2 = permute_signed(address_word_9, 0u, 0x8880u);

                        index_word_3 = permute_signed(matrix_fragment_15, 0u, 0x8880u);

                        address_word_2 = permute_signed(address_word_2, 0u, 0x7710u);

                        index_word_3 = permute_signed(index_word_3, 0u, 0x7710u);

                        address_word_2 = uint32_t(uint32_t(address_word_2) >> 0xcu);

                        index_word_3 = uint32_t(uint32_t(index_word_3) >> 0xcu);

                        address_word_2 = address_word_2 & 0x7u;

                        matrix_fragment_14 = index_word_3 & 0x7u;

                        matrix_fragment_12 = address_word_9 + address_word_2;

                        matrix_fragment_17 = matrix_fragment_15 + matrix_fragment_14;

                        address_word_2 = matrix_fragment_12 & 0xf8u;

                        index_word_3 = matrix_fragment_17 & 0xf8u;

                        matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x8880u);

                        matrix_fragment_14 = 0u - address_word_2;

                        matrix_fragment_18 = 0u - index_word_3;

                        address_word_2 = index_word_5 * 0x4u;

                        matrix_fragment_19 = permute_signed(matrix_fragment_17, 0u, 0x8880u);

                        matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

                        matrix_fragment_18 = permute_signed(matrix_fragment_18, 0u, 0x7710u);

                        index_word_3 = address_word_2 & 0x1cu;

                        address_word_9 = address_word_9 + matrix_fragment_14;

                        matrix_fragment_14 = matrix_fragment_13 * 0x4u;

                        matrix_fragment_17 = matrix_fragment_15 + matrix_fragment_18;

                        index_word_3 = index_word_0 * 0x20u + index_word_3;

                        matrix_fragment_18 = matrix_fragment_12;

                        address_word_9 = permute_signed(address_word_9, 0u, 0x8880u);

                        matrix_fragment_15 = matrix_fragment_14 & 0x1cu;

                        matrix_fragment_12 = index_word_3 * 0x4u;

                        index_word_3 = uint32_t(int32_t(matrix_fragment_18) >> 0x3u);

                        matrix_fragment_18 = uint32_t(int32_t(matrix_fragment_19) >> 0x3u);

                        matrix_fragment_15 = index_word_0 * 0x20u + matrix_fragment_15;

                        index_word_3 = permute_signed(index_word_3, 0u, 0x9910u);

                        matrix_fragment_19 = permute_signed(matrix_fragment_18, 0u, 0x9910u);

                        matrix_fragment_18 = matrix_fragment_15 * 0x4u;

                        matrix_fragment_12 = matrix_fragment_12 & 0xffffff80u;

                        address_word_2 = address_word_2 & 0xcu;

                        matrix_fragment_17 = permute_signed(matrix_fragment_17, 0u, 0x8880u);

                        matrix_fragment_14 = matrix_fragment_14 & 0xcu;

                        address_word_2 = address_word_9 * 0x10u + address_word_2;

                        matrix_fragment_15 = matrix_fragment_19;

                        index_word_3 = index_word_3 + index_word_8 + matrix_fragment_12;

                        index_word_5 = uint32_t(uint32_t(index_word_5) >> 0x1u);

                        address_word_9 = matrix_fragment_17 * 0x10u + matrix_fragment_14;

                        matrix_fragment_18 = matrix_fragment_18 & 0xffffff80u;

                        index_word_8 = uint32_t(uint32_t(matrix_fragment_13) >> 0x1u);

                        index_word_5 = index_word_5 & 0x2u;

                        matrix_fragment_15 =
                            matrix_fragment_15 + matrix_fragment_16 + matrix_fragment_18;

                        index_word_8 = index_word_8 & 0x2u;

                        index_word_3 = address_word_2 + index_word_3 + index_word_5;

                        address_word_9 = address_word_9 + matrix_fragment_15 + index_word_8;

                        index_word_5 = packed_half_10 + matrix_fragment_13;

                        split_address_words(address_word_2, index_word_3,
                                            int64_t(int32_t(index_word_3)) *
                                                    int64_t(int32_t(0x4u)) +
                                                combine_words(index_word_6, index_word_7));

                        split_address_words(index_word_8, address_word_9,
                                            int64_t(int32_t(address_word_9)) *
                                                    int64_t(int32_t(0x4u)) +
                                                combine_words(index_word_6, index_word_7));

                        pipeline_write<CheckBounds>(combine_words(address_word_2, index_word_3), 0u,
                                                    memory_ranges, error_counts);

                        condition_0 = int32_t(index_word_5) >= int32_t(packed_half_20);

                        pipeline_write<CheckBounds>(combine_words(index_word_8, address_word_9), 0u,
                                                    memory_ranges, error_counts);
                    } while (!condition_0);
                }
            }
        }
    }
    __syncwarp(__activemask());

    index_word_6 = parameters[parameter_publication_low_12];
    index_word_7 = parameters[parameter_publication_high_13];

    phoenix_architecture::finish_compatibility_copies();

    condition_0 = combine_words(index_word_6, index_word_7) != 0ull;

    if (!condition_0) {
        return;
    }

    index_word_3 = threadIdx.x;

    index_word_125 = index_word_125 + 0x7fu;

    address_word_2 = threadIdx.z;

    condition_0 = (address_word_2 | address_word_4 | index_word_3) != 0;

    address_word_2 = uint32_t(int32_t(index_word_125) >> 0x1fu);

    address_word_2 = uint32_t(combine_words(address_word_2, 0u) << 0x7u >> 32) + index_word_125;

    if (condition_0) {
        return;
    }

    address_word_2 = uint32_t(int32_t(address_word_2) >> 0x7u);

    index_word_5 = packed_half_11 * 0x2u;

    condition_0 = int32_t(index_word_5) >= int32_t(address_word_2);

    index_word_5 = index_word_5 + 0x1u;

    condition_1 = int32_t(index_word_5) >= int32_t(address_word_2);

    if (!condition_0) {
        index_word_3 = packed_half_11 * 0x40u + index_word_0;

        split_address_words(address_word_2, index_word_3,
                            uint64_t(uint32_t(index_word_3)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(index_word_6, index_word_7));

        __threadfence();

        pipeline_write<CheckBounds>(combine_words(address_word_2, index_word_3), 0u, memory_ranges,
                                    error_counts);
    }

    if (condition_1) {
        return;
    }

    index_word_3 = index_word_5 * 0x20u + index_word_0;

    split_address_words(address_word_2, index_word_3,
                        uint64_t(uint32_t(index_word_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(index_word_6, index_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(address_word_2, index_word_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__
__maxnreg__(181) void phoenix_global_attention_transformer_attention(KernelArguments parameters) {
    run_global_attention_transformer_attention<false>(parameters.words, nullptr, nullptr);
}
// Large inputs have enough blocks to benefit from higher residency. Keep this
// separately compiled: forcing its register budget on small inputs loses speed.
extern "C" __global__ __maxnreg__(128) void
phoenix_global_attention_transformer_attention_large_input(KernelArguments parameters) {
    run_global_attention_transformer_attention<false>(parameters.words, nullptr, nullptr);
}
