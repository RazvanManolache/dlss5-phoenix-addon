// DLSS5 Phoenix optimized kernel: global_attention/transformer/query_key_value.cu
// Coarsest 1024-channel global-transformer stage. Attention spans the full latent token set.
// Projection and feed-forward modules retain their explicit publication and synchronization boundaries.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 9232
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
    parameter_output_q_low_2 = 2,
    parameter_output_q_high_3 = 3,
    parameter_output_k_low_4 = 4,
    parameter_output_k_high_5 = 5,
    parameter_output_v_low_6 = 6,
    parameter_output_v_high_7 = 7,
    parameter_trained_weights_low_8 = 8,
    parameter_trained_weights_high_9 = 9,
    parameter_reduction_counter_low_10 = 10,
    parameter_reduction_counter_high_11 = 11,
    parameter_reduction_scratch_low_12 = 12,
    parameter_reduction_scratch_high_13 = 13,
    parameter_wait_low_14 = 14,
    parameter_wait_high_15 = 15,
    parameter_publication_low_16 = 16,
    parameter_publication_high_17 = 17,
    parameter_LH_18 = 18,
    parameter_LW_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[20];
};
static_assert(sizeof(KernelArguments) == 80, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_transformer_query_key_value(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t matrix_fragment_8 = 0;
    uint32_t matrix_fragment_9 = 0;
    uint32_t matrix_fragment_10 = 0;
    uint32_t matrix_fragment_11 = 0;
    uint32_t matrix_accumulator_12 = 0;
    uint32_t matrix_accumulator_13 = 0;
    uint32_t matrix_accumulator_14 = 0;
    uint32_t matrix_accumulator_15 = 0;
    uint32_t matrix_fragment_16 = 0;
    uint32_t matrix_fragment_17 = 0;
    uint32_t matrix_fragment_18 = 0;
    uint32_t matrix_fragment_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_fragment_24 = 0;
    uint32_t matrix_fragment_25 = 0;
    uint32_t matrix_fragment_26 = 0;
    uint32_t matrix_fragment_27 = 0;
    uint32_t matrix_fragment_28 = 0;
    uint32_t matrix_fragment_29 = 0;
    uint32_t matrix_fragment_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_accumulator_32 = 0;
    uint32_t matrix_accumulator_33 = 0;
    uint32_t packed_half_34 = 0;
    uint32_t packed_half_35 = 0;
    uint32_t address_word_36 = 0;
    uint32_t address_word_37 = 0;
    uint32_t packed_half_38 = 0;
    uint32_t packed_half_39 = 0;
    uint32_t matrix_accumulator_40 = 0;
    uint32_t packed_half_41 = 0;
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
    uint32_t address_word_86 = 0;
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
    uint32_t address_word_100 = 0;
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
    uint32_t packed_half_133 = 0;
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
    uint32_t index_word_144 = 0;
    uint32_t index_word_145 = 0;
    uint32_t index_word_146 = 0;
    uint32_t index_word_147 = 0;
    uint32_t index_word_148 = 0;
    uint32_t index_word_149 = 0;
    uint32_t index_word_150 = 0;
    uint32_t tile_word_151 = 0;
    uint32_t index_word_152 = 0;
    uint32_t index_word_153 = 0;
    uint32_t index_word_154 = 0;
    uint32_t index_word_155 = 0;
    uint32_t index_word_156 = 0;
    uint32_t index_word_157 = 0;
    uint32_t index_word_158 = 0;
    uint32_t index_word_159 = 0;
    uint32_t tile_word_160 = 0;
    uint32_t index_word_161 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_address_word_10 = 0;
    uint32_t uniform_address_word_11 = 0;
    uint32_t uniform_address_word_12 = 0;
    uint32_t uniform_address_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_float_bits_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_index_word_17 = 0;
    uint32_t uniform_address_word_18 = 0;
    uint32_t uniform_address_word_19 = 0;
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

    uniform_address_word_6 = parameters[parameter_LH_18];
    uniform_parameter_word_7 = parameters[parameter_LW_19];

    matrix_fragment_17 = threadIdx.y;

    uniform_address_word_12 = blockIdx.x;

    matrix_fragment_68 = threadIdx.x;

    matrix_fragment_69 = blockIdx.z;

    matrix_accumulator_12 = parameters[parameter_trained_weights_low_8];
    matrix_accumulator_13 = parameters[parameter_trained_weights_high_9];

    matrix_fragment_19 = threadIdx.x & 31u;

    uniform_parameter_word_7 = uniform_parameter_word_7 * uniform_address_word_6;

    uniform_address_word_6 = uniform_parameter_word_7 + 0xffffffffu;

    uniform_address_word_10 = uint32_t(int32_t(uniform_address_word_6) >> 0x1fu);

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_address_word_10, 0u) << 0x7u >> 32) + uniform_address_word_6;

    condition_1 = (matrix_fragment_17 & 0x1u) != 0;

    uniform_address_word_11 = uint32_t(int32_t(uniform_index_word_4) >> 0x7u);

    uniform_index_word_14 = uniform_address_word_11 + 0x1u;

    matrix_fragment_70 = !condition_1 ? 0u : 0x60u;

    uniform_condition_2 = int32_t(uniform_index_word_14) != int32_t(0u);

    index_word_0 = uniform_index_word_14;

    matrix_accumulator_2 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    matrix_fragment_4 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    uniform_float_bits_15 = matrix_accumulator_2;

    matrix_accumulator_2 = blockIdx.x;

    matrix_fragment_4 = 0u - matrix_fragment_4;

    uniform_index_word_4 = __float_as_uint(__int2float_ru(uniform_float_bits_15));

    matrix_accumulator_3 = reciprocal_bits(uniform_index_word_4);

    matrix_accumulator_2 =
        int32_t(matrix_accumulator_2) < 0 ? 0u - matrix_accumulator_2 : matrix_accumulator_2;

    index_word_0 = matrix_accumulator_2;

    uniform_index_word_4 = matrix_accumulator_3;

    uniform_address_word_9 = index_word_0;

    index_word_0 = matrix_fragment_4;

    uniform_address_word_13 = index_word_0;

    uniform_index_word_4 = uniform_index_word_4 + 0xffffffeu;

    index_word_0 = matrix_fragment_17 | matrix_fragment_68;
    condition_0 = index_word_0 != 0;

    uniform_index_word_5 = __float2uint_rz(__uint_as_float(uniform_index_word_4));

    uniform_index_word_4 = 0u;

    matrix_accumulator_3 = 0u - uniform_index_word_5;

    if (!condition_0) {
        matrix_accumulator_2 = blockDim.x;
    }

    uniform_address_word_8 = matrix_accumulator_3;

    if (!condition_0) {
        matrix_accumulator_3 = blockDim.y;
    }

    uniform_address_word_8 = uniform_address_word_8 * uniform_float_bits_15;

    uniform_index_word_4 = uint32_t(uint64_t(uniform_index_word_5) * uniform_address_word_8 +
                                        combine_words(uniform_index_word_4, uniform_index_word_5) >>
                                    32);

    split_address_words(uniform_index_word_4, uniform_index_word_5,
                        uint64_t(uint32_t(uniform_index_word_4)) *
                            uint64_t(uint32_t(uniform_address_word_9)));

    uniform_index_word_4 = uniform_index_word_5 * uniform_address_word_13 + uniform_address_word_9;

    uniform_condition_0 = uint32_t(uniform_float_bits_15) > uint32_t(uniform_index_word_4);

    if (!condition_0) {
        matrix_accumulator_2 = matrix_accumulator_2 * matrix_accumulator_3;
    }

    uniform_address_word_8 =
        broadcast_warp_word(matrix_accumulator_2, !condition_0, uniform_address_word_8);

    matrix_accumulator_2 = matrix_fragment_69 * 0xc00u;

    if (!uniform_condition_0) {
        uniform_index_word_4 = uniform_index_word_4 + (0u - uniform_float_bits_15);

        uniform_index_word_5 = uniform_index_word_5 + 0x1u;
    }

    uniform_condition_1 = uint32_t(uniform_index_word_4) >= uint32_t(uniform_float_bits_15);

    uniform_index_word_4 = uniform_address_word_12 ^ uniform_index_word_14;

    uniform_condition_0 = int32_t(uniform_index_word_4) >= int32_t(0u);

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_5 + 0x1u;
    }

    uniform_condition_1 = vote_warp_predicate<true>(condition_0);

    if (!uniform_condition_0) {
        uniform_index_word_5 = 0u - uniform_index_word_5;
    }

    if (!uniform_condition_1) {
        uniform_address_word_9 = 0u;

        uniform_index_word_4 = 0x400u;
    }

    if (!uniform_condition_2) {
        uniform_index_word_5 = ~uniform_index_word_14;
    }

    if (!uniform_condition_1) {
        uniform_index_word_4 = uniform_index_word_4 + 0x2000u;
    }

    uniform_condition_0 = vote_warp_predicate<true>(condition_0);

    uniform_condition_2 = vote_warp_predicate<true>(condition_0);

    uniform_index_word_14 = 0u;
    uniform_float_bits_15 = 0u;

    matrix_accumulator_3 = uniform_index_word_5;

    matrix_fragment_70 = matrix_accumulator_3 * 0xc0u + matrix_fragment_70;

    matrix_fragment_71 = matrix_fragment_70 + 0x10u;

    index_word_146 = matrix_fragment_70 + 0x20u;

    index_word_147 = matrix_fragment_70 + 0x30u;

    index_word_148 = matrix_fragment_70 + 0x40u;

    index_word_149 = matrix_fragment_70 + 0x50u;

    matrix_accumulator_3 = uint32_t(int64_t(int32_t(matrix_fragment_70)) * (1ll << 0x1cu) >> 32) +
                           matrix_accumulator_2;

    if (!uniform_condition_1) {
        uniform_index_word_4 = (uniform_address_word_9 << 0x18u) + uniform_index_word_4;

        uniform_address_word_8 = 0u - uniform_address_word_8 + 0x100000u;

        uniform_address_word_9 = uniform_address_word_8 << (0xbu & 31u);

        uniform_address_word_8 = uniform_address_word_8 << (0x1u & 31u);
    }

    matrix_fragment_4 = uint32_t(int64_t(int32_t(matrix_fragment_71)) * (1ll << 0x1cu) >> 32) +
                        matrix_accumulator_2;

    matrix_fragment_6 =
        uint32_t(int64_t(int32_t(index_word_146)) * (1ll << 0x1cu) >> 32) + matrix_accumulator_2;

    matrix_accumulator_3 = matrix_accumulator_3 * 0x80u;

    matrix_fragment_8 =
        uint32_t(int64_t(int32_t(index_word_147)) * (1ll << 0x1cu) >> 32) + matrix_accumulator_2;

    matrix_fragment_5 = matrix_fragment_4 * 0x80u;

    matrix_fragment_10 =
        uint32_t(int64_t(int32_t(index_word_148)) * (1ll << 0x1cu) >> 32) + matrix_accumulator_2;

    matrix_fragment_7 = matrix_fragment_6 * 0x80u;

    matrix_accumulator_2 =
        uint32_t(int64_t(int32_t(index_word_149)) * (1ll << 0x1cu) >> 32) + matrix_accumulator_2;

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    matrix_fragment_9 = matrix_fragment_8 << (0x7u & 31u);

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_4);
#endif
    }

    matrix_fragment_11 = matrix_fragment_10 * 0x80u;

    matrix_accumulator_15 = matrix_accumulator_2 * 0x80u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    if (!uniform_condition_2) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_index_word_4 + 0x8u);
#endif
    }

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        int64_t(int32_t(matrix_accumulator_15)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __syncthreads();

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_6, matrix_fragment_7));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_10, matrix_fragment_11));

    split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                        int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_12, matrix_accumulator_13));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x80ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x80ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x80ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x80ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x80ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x80ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    uniform_condition_0 = uint32_t(uniform_parameter_word_7) != 0u;

    uniform_parameter_word_7 =
        uniform_address_word_11 * uniform_index_word_5 + uniform_index_word_5;

    uniform_index_word_4 = 0u;

    matrix_accumulator_15 = threadIdx.z;

    address_word_36 = 0u;

    uniform_parameter_word_7 = 0u - uniform_parameter_word_7 + uniform_address_word_12;

    uniform_index_word_17 = uniform_parameter_word_7 << (0x3u & 31u);

    if (uniform_condition_0) {
        uniform_address_word_10 =
            uint32_t(combine_words(uniform_address_word_10, 0u) << 0x5u >> 32) +
            uniform_address_word_6;

        uniform_address_word_10 = uniform_address_word_10 & 0xffffffe0u;

        uniform_address_word_10 = uniform_address_word_10 + 0x20u;

        uniform_index_word_4 = uint32_t(int32_t(uniform_address_word_10) >> 0x4u);
    }

    condition_0 = (index_word_0 | matrix_accumulator_15) != 0;
    if (!condition_0) {
        matrix_accumulator_2 = parameters[parameter_wait_low_14];
        matrix_accumulator_3 = parameters[parameter_wait_high_15];

        matrix_fragment_5 = (matrix_fragment_69 << 0x2u) + uniform_index_word_17;

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        index_word_0 = pipeline_read<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges, error_counts);

        condition_0 = int32_t(index_word_0) > -1;
        if (!condition_0) {
            do {
                __nanosleep(0x40u);

                index_word_0 = pipeline_read<CheckBounds>(
                    combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                    error_counts);

                condition_0 = int32_t(index_word_0) > -1;
            } while (!condition_0);
        }
    }
    __syncwarp(__activemask());

    uniform_index_word_16 = 0u;

    index_word_155 = matrix_fragment_17 + uniform_index_word_17;

    uniform_address_word_13 = 0x400u;

    __syncthreads();

    condition_1 = int32_t(index_word_155) >= int32_t(uniform_index_word_4);

    index_word_152 = index_word_155 * 0x1000u;

    condition_0 = int32_t(index_word_155) >= int32_t(uniform_index_word_4);

    matrix_accumulator_2 = (matrix_fragment_69 << 0xbu) + index_word_152;

    index_word_0 = !condition_0 ? matrix_accumulator_2 : 0u;

    uniform_address_word_8 = (uniform_index_word_16 << 0x18u) + uniform_address_word_13;

    index_word_153 = (matrix_fragment_17 << 0x9u) + uniform_address_word_8;
    if (condition_1) {
        uniform_address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        index_word_0 = matrix_fragment_19 * 0x10u + index_word_153;

        matrix_fragment_4 = uniform_address_word_8;

        matrix_accumulator_3 = uniform_address_word_8;

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_accumulator_3, 0x5410u);

        matrix_fragment_5 = matrix_fragment_4;

        matrix_fragment_6 = matrix_fragment_4;

        matrix_fragment_7 = matrix_fragment_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, index_word_0, matrix_fragment_4, matrix_fragment_5, matrix_fragment_6,
            matrix_fragment_7, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_8 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_8 = parameters[parameter_input_low_0];
            uniform_address_word_9 = parameters[parameter_input_high_1];

            uniform_address_word_11 = index_word_0;

            uniform_address_word_18 = index_word_153;

            uniform_address_word_10 = uniform_address_word_13 + 0x2000u;

            index_word_0 = 0x200u;

            uniform_address_word_19 = (uniform_index_word_16 << 0x18u) + uniform_address_word_10;

            uniform_address_word_10 = 0x20u;

            split_address_words(uniform_address_word_8, uniform_address_word_9,
                                int64_t(int32_t(uniform_address_word_11)) * int64_t(int32_t(0x4u)) +
                                    combine_words(uniform_address_word_8, uniform_address_word_9));

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_18,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_10 * 16u, uniform_address_word_19, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_18,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_10 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    index_word_156 = index_word_155 + 0x4u;

    index_word_0 = matrix_accumulator_2 + 0x4000u;

    condition_1 = int32_t(index_word_156) >= int32_t(uniform_index_word_4);

    condition_0 = int32_t(index_word_156) >= int32_t(uniform_index_word_4);

    index_word_0 = !condition_0 ? index_word_0 : 0u;
    if (condition_1) {
        uniform_address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = matrix_fragment_19 * 0x10u + index_word_153;

        matrix_fragment_4 = uniform_address_word_8;

        matrix_accumulator_3 = uniform_address_word_8;

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_accumulator_3, 0x5410u);

        matrix_fragment_5 = matrix_fragment_4;

        matrix_fragment_6 = matrix_fragment_4;

        matrix_fragment_7 = matrix_fragment_4;

        store_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_19 + 0x800u, matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_address_word_8 = 0xffffffffu;

        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            uniform_address_word_8 = parameters[parameter_input_low_0];
            uniform_address_word_9 = parameters[parameter_input_high_1];

            uniform_address_word_10 = index_word_0;

            uniform_address_word_12 = index_word_153;

            uniform_address_word_11 = uniform_address_word_13 + 0x2000u;

            index_word_0 = 0x200u;

            uniform_address_word_11 = (uniform_index_word_16 << 0x18u) + uniform_address_word_11;

            split_address_words(uniform_address_word_8, uniform_address_word_9,
                                int64_t(int32_t(uniform_address_word_10)) * int64_t(int32_t(0x4u)) +
                                    combine_words(uniform_address_word_8, uniform_address_word_9));

            uniform_address_word_10 = uniform_address_word_12 + 0x800u;

            uniform_address_word_12 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_10,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_12 * 16u, uniform_address_word_11, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_10,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_12 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_fragment_5 = 0u;

    uniform_address_word_8 = uniform_address_word_13 + 0x2000u;

    index_word_145 = matrix_fragment_17 * 0x2u;

    index_word_150 = matrix_fragment_17 << (0xau & 31u);

    index_word_0 = uniform_address_word_8;

    matrix_fragment_5 = (matrix_fragment_5 << 0x18u) + index_word_0;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = matrix_fragment_5;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_2 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_5,
                                    async_tokens[(((matrix_fragment_5) >> 3) & 1)]);
#else
    __syncthreads();
    condition_0 = true;
#endif
    if (!condition_0) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_0 = wait_copy_barrier(shared_tile, matrix_fragment_5,
                                            async_tokens[(((matrix_fragment_5) >> 3) & 1)]);
#else
            __syncthreads();
            condition_0 = true;
#endif
        } while (!condition_0);
    }
    index_word_154 = 0u;

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    address_word_86 = 0;
    matrix_accumulator_87 = 0;

    matrix_accumulator_88 = 0;
    matrix_accumulator_89 = 0;

    matrix_accumulator_90 = 0;
    matrix_accumulator_91 = 0;

    matrix_accumulator_92 = 0;
    matrix_accumulator_93 = 0;

    matrix_accumulator_94 = 0;
    matrix_accumulator_95 = 0;

    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_132 = 0;
    packed_half_133 = 0;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    address_word_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_14 = 0;
    matrix_accumulator_15 = 0;

    matrix_accumulator_12 = 0;
    matrix_accumulator_13 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_50 = 0;
    matrix_accumulator_51 = 0;

    matrix_accumulator_48 = 0;
    matrix_accumulator_49 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    matrix_accumulator_140 = 0;
    matrix_accumulator_141 = 0;

    matrix_accumulator_142 = 0;
    matrix_accumulator_143 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    packed_half_34 = 0;
    packed_half_35 = 0;

    packed_half_38 = 0;
    packed_half_39 = 0;

    matrix_accumulator_40 = 0;
    packed_half_41 = 0;

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    index_word_145 = index_word_145 & 0x7fcu;

    index_word_150 = index_word_150 & 0xff800u;

#pragma unroll 1
    do {
        matrix_fragment_4 = 0u;

        index_word_158 = index_word_154 * 0x80u;

        index_word_157 = 0x400u;

        index_word_0 = threadIdx.x & 31u;

        index_word_158 = index_word_158 & 0x1000u;

        __syncwarp(__activemask());

        address_word_37 = threadIdx.y;

        index_word_161 = index_word_154;

        index_word_159 = uint32_t(uint32_t(index_word_154) >> 0x5u);

        index_word_144 = threadIdx.z;

        index_word_154 = index_word_154 + 0x20u;

        tile_word_151 = (matrix_fragment_4 << 0x18u) + index_word_157;

        matrix_fragment_5 = tile_word_151 + index_word_158 + index_word_150;

        tile_word_160 = index_word_0 * 0x10u + matrix_fragment_5;

        load_four_shared_words<CheckBounds>(shared_tile, tile_word_160, matrix_fragment_4,
                                            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_160 + 0x200u, matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

        index_word_144 = address_word_37 | index_word_144 | matrix_fragment_68;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_160 + 0x400u, matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_160 + 0x600u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        condition_0 = (index_word_154 & 0x60u) != 0;

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_130,
                                 matrix_accumulator_131);

        condition_1 = uint32_t(index_word_161) >= uint32_t(0x1c0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_40, packed_half_41, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_40,
                                 packed_half_41);

        multiply_accumulate_half(packed_half_38, packed_half_39, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_64, matrix_fragment_65, packed_half_38,
                                 packed_half_39);

        multiply_accumulate_half(packed_half_34, packed_half_35, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_66, matrix_fragment_67, packed_half_34,
                                 packed_half_35);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(address_word_100, matrix_accumulator_101, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_78, matrix_fragment_79, address_word_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_132, packed_half_133, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_132,
                                 packed_half_133);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_30, matrix_fragment_31, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_90,
                                 matrix_accumulator_91);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_28, matrix_fragment_29, matrix_accumulator_88,
                                 matrix_accumulator_89);

        multiply_accumulate_half(address_word_86, matrix_accumulator_87, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_30, matrix_fragment_31, address_word_86,
                                 matrix_accumulator_87);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_32,
                                 matrix_accumulator_33);
        if (!condition_0) {
            condition_0 = uint32_t(index_word_144) != 0u;
            if (!condition_0) {
                matrix_fragment_4 = parameters[parameter_wait_low_14];
                matrix_fragment_5 = parameters[parameter_wait_high_15];

                matrix_fragment_6 = matrix_fragment_69 * 0x200u + index_word_154;

                matrix_fragment_7 = uint32_t(combine_words(matrix_fragment_6, 0u) << 0x19u >> 32) +
                                    uniform_index_word_17;

                split_address_words(matrix_fragment_4, matrix_fragment_5,
                                    uint64_t(uint32_t(matrix_fragment_7)) *
                                            uint64_t(uint32_t(0x4u)) +
                                        combine_words(matrix_fragment_4, matrix_fragment_5));

                matrix_fragment_6 =
                    pipeline_read<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                               memory_ranges, error_counts);

                condition_0 = int32_t(matrix_fragment_6) > -1;
                if (!condition_0) {
                    do {
                        __nanosleep(0x40u);

                        matrix_fragment_6 = pipeline_read<CheckBounds>(
                            combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges,
                            error_counts);

                        condition_0 = int32_t(matrix_fragment_6) > -1;
                    } while (!condition_0);
                }
            }
            __syncwarp(__activemask());

            __syncthreads();
        }
        uniform_address_word_9 = 0u;

        uniform_address_word_10 = index_word_159;

        uniform_address_word_8 = index_word_157;

        condition_0 = int32_t(index_word_155) >= int32_t(uniform_index_word_4);

        matrix_fragment_17 = (matrix_fragment_69 << 0x9u) + index_word_154;

        condition_2 = int32_t(index_word_155) >= int32_t(uniform_index_word_4);

        index_word_158 = 0x1000u ^ index_word_158;

        matrix_fragment_9 = matrix_fragment_17 * 0x4u + index_word_152;

        {
            uint32_t result_word;
            result_word = uniform_address_word_10 & 0x1u;
            uniform_condition_0 = result_word != 0;
        }

        matrix_fragment_11 = index_word_153 + index_word_158;

        uniform_address_word_8 = uniform_address_word_8 + 0x2000u;

        matrix_fragment_4 = !condition_2 ? matrix_fragment_9 : 0u;

        uniform_address_word_19 = (uniform_address_word_9 << 0x18u) + uniform_address_word_8;

        if (!uniform_condition_0) {
            uniform_address_word_19 = uniform_address_word_19 + 0x8u;
        }
        if (condition_0) {
            uniform_address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_fragment_8 = index_word_0 * 0x10u + matrix_fragment_11;

            matrix_fragment_4 = uniform_address_word_8;

            matrix_fragment_5 = uniform_address_word_8;

            matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_5, 0x5410u);

            matrix_fragment_5 = matrix_fragment_4;

            matrix_fragment_6 = matrix_fragment_4;

            matrix_fragment_7 = matrix_fragment_4;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_8, matrix_fragment_4, matrix_fragment_5,
                matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_8 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_8 = parameters[parameter_input_low_0];
                uniform_address_word_9 = parameters[parameter_input_high_1];

                uniform_address_word_10 = matrix_fragment_4;

                uniform_address_word_18 = matrix_fragment_11;

                matrix_fragment_4 = 0x200u;

                split_address_words(
                    uniform_address_word_8, uniform_address_word_9,
                    int64_t(int32_t(uniform_address_word_10)) * int64_t(int32_t(0x4u)) +
                        combine_words(uniform_address_word_8, uniform_address_word_9));

                uniform_address_word_10 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_18,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_10 * 16u, uniform_address_word_19, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_18,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_10 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        condition_0 = int32_t(index_word_156) >= int32_t(uniform_index_word_4);

        condition_2 = int32_t(index_word_156) >= int32_t(uniform_index_word_4);

        matrix_fragment_4 = matrix_fragment_9 + 0x4000u;

        matrix_fragment_4 = !condition_2 ? matrix_fragment_4 : 0u;
        if (condition_0) {
            uniform_address_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_fragment_11 = index_word_0 * 0x10u + matrix_fragment_11;

            matrix_fragment_4 = uniform_address_word_8;

            matrix_fragment_5 = uniform_address_word_8;

            matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_5, 0x5410u);

            matrix_fragment_5 = matrix_fragment_4;

            matrix_fragment_6 = matrix_fragment_4;

            matrix_fragment_7 = matrix_fragment_4;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_11 + 0x800u, matrix_fragment_4, matrix_fragment_5,
                matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_address_word_8 = 0xffffffffu;

            condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_0) {
                uniform_address_word_8 = parameters[parameter_input_low_0];
                uniform_address_word_9 = parameters[parameter_input_high_1];

                uniform_address_word_10 = matrix_fragment_4;

                uniform_address_word_11 = matrix_fragment_11;

                uniform_address_word_12 = 0x20u;

                matrix_fragment_4 = 0x200u;

                split_address_words(
                    uniform_address_word_8, uniform_address_word_9,
                    int64_t(int32_t(uniform_address_word_10)) * int64_t(int32_t(0x4u)) +
                        combine_words(uniform_address_word_8, uniform_address_word_9));

                uniform_address_word_10 = uniform_address_word_11 + 0x800u;

                uniform_address_word_11 = uniform_address_word_19;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_12 * 16u, uniform_address_word_19, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_10,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_12 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        matrix_fragment_4 = parameters[parameter_trained_weights_low_8];
        matrix_fragment_5 = parameters[parameter_trained_weights_high_9];

        matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_17) >> 0x5u);

        matrix_fragment_6 = matrix_fragment_6 * 0xc0u;

        matrix_fragment_7 = uint32_t(int64_t(int32_t(matrix_fragment_70)) * (1ll << 0x1cu) >> 32) +
                            matrix_fragment_6;

        matrix_fragment_8 = uint32_t(int64_t(int32_t(matrix_fragment_71)) * (1ll << 0x1cu) >> 32) +
                            matrix_fragment_6;

        matrix_fragment_10 =
            uint32_t(int64_t(int32_t(index_word_146)) * (1ll << 0x1cu) >> 32) + matrix_fragment_6;

        matrix_fragment_16 =
            uint32_t(int64_t(int32_t(index_word_147)) * (1ll << 0x1cu) >> 32) + matrix_fragment_6;

        matrix_fragment_9 = matrix_fragment_8 * 0x80u;

        matrix_fragment_18 =
            uint32_t(int64_t(int32_t(index_word_148)) * (1ll << 0x1cu) >> 32) + matrix_fragment_6;

        matrix_fragment_11 = matrix_fragment_10 * 0x80u;

        matrix_fragment_6 =
            uint32_t(int64_t(int32_t(index_word_149)) * (1ll << 0x1cu) >> 32) + matrix_fragment_6;

        matrix_fragment_17 = matrix_fragment_16 * 0x80u;

        matrix_fragment_29 = matrix_fragment_18 << (0x7u & 31u);

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        matrix_fragment_7 = matrix_fragment_7 << (0x7u & 31u);

        matrix_fragment_73 = matrix_fragment_6 * 0x80u;

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(matrix_fragment_73)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_28, matrix_fragment_29));

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_72, matrix_fragment_73));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x80ull, matrix_fragment_28,
            matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x80ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7) + 0x80ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9) + 0x80ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x80ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x80ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = uniform_address_word_19;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_4 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_0 = wait_copy_barrier(shared_tile, uniform_address_word_19,
                                        async_tokens[(((uniform_address_word_19) >> 3) & 1)]);
#else
        __syncthreads();
        condition_0 = true;
#endif
        if (!condition_0) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_0 =
                    wait_copy_barrier(shared_tile, uniform_address_word_19,
                                      async_tokens[(((uniform_address_word_19) >> 3) & 1)]);
#else
                __syncthreads();
                condition_0 = true;
#endif
            } while (!condition_0);
        }
    } while (!condition_1);
    tile_word_151 = tile_word_151 + index_word_150;

    __syncwarp(__activemask());

    tile_word_151 = index_word_0 * 0x10u + tile_word_151;

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_151 + 0x1000u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_151 + 0x1200u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_151 + 0x1400u, matrix_fragment_24,
                                        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_151 + 0x1600u, matrix_fragment_68,
                                        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_130,
                             matrix_accumulator_131);

    uniform_address_word_8 = uint32_t(int32_t(uniform_address_word_6) >> 0x1fu);

    uniform_address_word_18 = parameters[parameter_reduction_scratch_low_12];
    uniform_address_word_19 = parameters[parameter_reduction_scratch_high_13];

    uniform_address_word_8 =
        uint32_t(combine_words(uniform_address_word_8, 0u) << 0x5u >> 32) + uniform_address_word_6;

    uniform_address_word_12 = parameters[parameter_reduction_counter_low_10];
    uniform_address_word_13 = parameters[parameter_reduction_counter_high_11];

    uniform_address_word_8 = uniform_address_word_8 & 0xffffffe0u;

    uniform_address_word_6 = (uniform_parameter_word_7 << 0x4u) + uniform_index_word_5;

    uniform_address_word_8 = uniform_address_word_8 + 0x20u;

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_128,
                             matrix_accumulator_129);

    uniform_index_word_16 = uint32_t(int32_t(uniform_address_word_8) >> 0x4u);

    uniform_condition_0 = combine_words(uniform_address_word_18, uniform_address_word_19) != 0ull;

    split_address_words(uniform_address_word_8, uniform_address_word_9,
                        int64_t(int32_t(uniform_index_word_16)) * int64_t(int32_t(0x8000u)) +
                            combine_words(uniform_address_word_18, uniform_address_word_19));

    split_address_words(uniform_address_word_12, uniform_address_word_13,
                        int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_12, uniform_address_word_13));

    split_address_words(uniform_address_word_8, uniform_address_word_9,
                        uniform_condition_0
                            ? combine_words(uniform_address_word_8, uniform_address_word_9)
                            : combine_words(0u, 0u));

    split_address_words(uniform_address_word_10, uniform_address_word_11,
                        int64_t(int32_t(uniform_index_word_16)) * int64_t(int32_t(0x8000u)) +
                            combine_words(uniform_address_word_8, uniform_address_word_9));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_124,
                             matrix_accumulator_125);

    split_address_words(uniform_address_word_10, uniform_address_word_11,
                        uniform_condition_0
                            ? combine_words(uniform_address_word_10, uniform_address_word_11)
                            : combine_words(0u, 0u));

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_accumulator_40, packed_half_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_40,
                             packed_half_41);

    multiply_accumulate_half(packed_half_38, packed_half_39, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_64,
                             matrix_fragment_65, packed_half_38, packed_half_39);

    multiply_accumulate_half(packed_half_34, packed_half_35, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_66,
                             matrix_fragment_67, packed_half_34, packed_half_35);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_142,
                             matrix_accumulator_143);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_140,
                             matrix_accumulator_141);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_50,
                             matrix_accumulator_51);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_12,
                             matrix_accumulator_13);

    multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_14,
                             matrix_accumulator_15);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_134,
                             matrix_accumulator_135);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_132,
                             packed_half_133);

    matrix_accumulator_132 = blockIdx.z;

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_104,
                             matrix_accumulator_105);

    condition_0 = uint32_t(matrix_accumulator_132) != 0u;

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(address_word_100, matrix_accumulator_101, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_78, matrix_fragment_79, address_word_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_30, matrix_fragment_31, address_word_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_32,
                             matrix_accumulator_33);
    if (!condition_0) {
        index_word_0 = threadIdx.y;

        index_word_145 = index_word_145 + uniform_index_word_17;

        address_word_37 = uniform_index_word_5;

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            combine_words(uniform_address_word_18, uniform_address_word_19));

        address_word_86 = threadIdx.x & 31u;

        condition_0 = int32_t(index_word_145) >= int32_t(uniform_index_word_16);

        address_word_36 = index_word_145 + 0x1u;

        matrix_fragment_68 = index_word_145 + 0x3u;

        matrix_fragment_80 = 0x4u;

        condition_1 = int32_t(address_word_36) >= int32_t(uniform_index_word_16);

        condition_3 = int32_t(matrix_fragment_68) >= int32_t(uniform_index_word_16);

        index_word_0 = index_word_0 << (0x5u & 31u);

        index_word_0 = index_word_0 & 0x20u;

        index_word_0 = address_word_37 * 0x40u + index_word_0;

        address_word_37 = index_word_145 + 0x2u;

        index_word_0 = index_word_145 * 0x400u + index_word_0;

        condition_2 = int32_t(address_word_37) >= int32_t(uniform_index_word_16);

        matrix_fragment_81 = index_word_0 << (0x3u & 31u);

        index_word_0 = 0x4u;

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_76, matrix_fragment_77));
        if (!condition_0) {
            split_address_words(address_word_36, address_word_37,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_76, matrix_fragment_77));

            matrix_fragment_68 = matrix_accumulator_130;

            matrix_fragment_69 = matrix_accumulator_131;

            matrix_fragment_70 = matrix_accumulator_128;

            matrix_fragment_71 = matrix_accumulator_129;

            split_address_words(matrix_fragment_78, matrix_fragment_79,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_76, matrix_fragment_77));

            matrix_fragment_72 = matrix_accumulator_124;

            matrix_fragment_73 = matrix_accumulator_125;

            matrix_fragment_74 = matrix_accumulator_122;

            matrix_fragment_75 = matrix_accumulator_123;

            store_four_words_streaming<CheckBounds>(combine_words(address_word_36, address_word_37),
                                                    matrix_fragment_68, matrix_fragment_69,
                                                    matrix_fragment_70, matrix_fragment_71,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_78, matrix_fragment_79) + 0x200ull,
                matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                memory_ranges, error_counts);
        }
        split_address_words(address_word_36, address_word_37,
                            int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(index_word_0)) +
                                combine_words(uniform_address_word_8, uniform_address_word_9));

        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x8000u);

        split_address_words(matrix_fragment_70, matrix_fragment_71,
                            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x10000u);

        split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                            combine_words(address_word_36, address_word_37) + 0x8000u);

        if (!condition_0) {
            split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_36, address_word_37));
        }

        split_address_words(matrix_fragment_82, matrix_fragment_83,
                            combine_words(address_word_36, address_word_37) + 0x10000u);

        if (!condition_1) {
            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_36, address_word_37));

            split_address_words(address_word_36, address_word_37,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_68, matrix_fragment_69));
        }

        if (!condition_2) {
            split_address_words(matrix_accumulator_124, matrix_accumulator_125,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_68, matrix_fragment_69));
        }

        if (!condition_1) {
            matrix_fragment_68 = matrix_accumulator_114;

            matrix_fragment_69 = matrix_accumulator_115;
        }

        if (!condition_2) {
            split_address_words(matrix_accumulator_96, matrix_accumulator_97,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_70, matrix_fragment_71));
        }

        if (!condition_3) {
            split_address_words(matrix_accumulator_128, matrix_accumulator_129,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_70, matrix_fragment_71));
        }

        if (!condition_1) {
            matrix_fragment_70 = matrix_accumulator_112;

            matrix_fragment_71 = matrix_accumulator_113;

            split_address_words(matrix_accumulator_122, matrix_accumulator_123,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_76, matrix_fragment_77));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_36, address_word_37),
                                                    matrix_fragment_68, matrix_fragment_69,
                                                    matrix_fragment_70, matrix_fragment_71,
                                                    memory_ranges, error_counts);
        }

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(matrix_fragment_81)) *
                                    int64_t(int32_t(matrix_fragment_80)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        if (!condition_1) {
            split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_84, matrix_accumulator_85));
        }

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x8000u);

        split_address_words(matrix_accumulator_98, matrix_accumulator_99,
                            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x10000u);

        if (!condition_2) {
            split_address_words(matrix_accumulator_90, matrix_accumulator_91,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));

            split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_84, matrix_accumulator_85));
        }

        if (!condition_1) {
            matrix_fragment_68 = matrix_accumulator_110;

            matrix_fragment_69 = matrix_accumulator_111;

            matrix_fragment_70 = matrix_accumulator_108;

            matrix_fragment_71 = matrix_accumulator_109;
        }

        if (!condition_0) {
            address_word_36 = packed_half_38;

            address_word_37 = packed_half_39;

            packed_half_38 = packed_half_34;

            packed_half_39 = packed_half_35;
        }

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_122, matrix_accumulator_123) + 0x8200ull,
                matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                memory_ranges, error_counts);
        }

        if (!condition_3) {
            split_address_words(matrix_fragment_82, matrix_fragment_83,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_82, matrix_fragment_83));
        }

        if (!condition_0) {
            split_address_words(matrix_fragment_80, matrix_fragment_81,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_72, matrix_fragment_73));
        }

        if (!condition_1) {
            split_address_words(matrix_fragment_78, matrix_fragment_79,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_74, matrix_fragment_75));

            split_address_words(matrix_fragment_76, matrix_fragment_77,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_72, matrix_fragment_73));
        }

        if (!condition_2) {
            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_74, matrix_fragment_75));

            matrix_fragment_68 = matrix_accumulator_106;

            matrix_fragment_69 = matrix_accumulator_107;

            matrix_fragment_70 = matrix_accumulator_104;

            matrix_fragment_71 = matrix_accumulator_105;
        }

        if (!condition_3) {
            matrix_accumulator_122 = matrix_accumulator_126;

            matrix_accumulator_123 = matrix_accumulator_127;
        }

        if (!condition_2) {
            split_address_words(matrix_fragment_74, matrix_fragment_75,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_98, matrix_accumulator_99));

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_96, matrix_accumulator_97), matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);
        }

        if (!condition_3) {
            split_address_words(address_word_86, matrix_accumulator_87,
                                int64_t(int32_t(address_word_86)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_98, matrix_accumulator_99));
        }

        if (!condition_2) {
            matrix_fragment_68 = matrix_accumulator_102;

            matrix_fragment_69 = matrix_accumulator_103;

            matrix_fragment_70 = address_word_100;

            matrix_fragment_71 = matrix_accumulator_101;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_124, matrix_accumulator_125) + 0x8200ull,
                matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                memory_ranges, error_counts);
        }

        if (!condition_3) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_128, matrix_accumulator_129) + 0x8000ull,
                matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_118,
                matrix_accumulator_119, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_128, matrix_accumulator_129) + 0x8200ull,
                matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_122,
                matrix_accumulator_123, memory_ranges, error_counts);
        }

        if (!condition_0) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_92, matrix_accumulator_93) + 0x200ull,
                address_word_36, address_word_37, packed_half_38, packed_half_39, memory_ranges,
                error_counts);

            matrix_fragment_68 = matrix_accumulator_42;

            matrix_fragment_69 = matrix_accumulator_43;

            matrix_fragment_70 = matrix_accumulator_40;

            matrix_fragment_71 = packed_half_41;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_92, matrix_accumulator_93), matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);

            packed_half_38 = matrix_fragment_4;

            packed_half_39 = matrix_fragment_5;

            matrix_fragment_4 = matrix_fragment_6;

            matrix_fragment_5 = matrix_fragment_7;
        }

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_94, matrix_accumulator_95), matrix_accumulator_44,
                matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47, memory_ranges,
                error_counts);
        }

        if (!condition_0) {
            matrix_fragment_6 = matrix_fragment_8;

            matrix_fragment_7 = matrix_fragment_9;

            address_word_36 = matrix_accumulator_2;
        }

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_88, matrix_accumulator_89) + 0x8200ull,
                matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
                matrix_accumulator_51, memory_ranges, error_counts);
        }

        if (!condition_0) {
            address_word_37 = matrix_accumulator_3;
        }

        if (!condition_3) {
            matrix_fragment_8 = matrix_fragment_26;

            matrix_fragment_9 = matrix_fragment_27;
        }

        if (!condition_2) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_90, matrix_accumulator_91), matrix_accumulator_52,
                matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_84, matrix_accumulator_85) + 0x8200ull,
                matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58,
                matrix_accumulator_59, memory_ranges, error_counts);
        }

        if (!condition_3) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_82, matrix_fragment_83) + 0x8000ull,
                matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_82, matrix_fragment_83) + 0x8200ull,
                matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                memory_ranges, error_counts);
        }

        if (!condition_0) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81) + 0x200ull, matrix_fragment_4,
                matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_80, matrix_fragment_81), address_word_36,
                address_word_37, packed_half_38, packed_half_39, memory_ranges, error_counts);
        }

        if (!condition_1) {
            matrix_fragment_4 = matrix_fragment_10;

            matrix_fragment_5 = matrix_fragment_11;

            matrix_fragment_6 = matrix_accumulator_12;

            matrix_fragment_7 = matrix_accumulator_13;
        }

        if (!condition_3) {
            matrix_fragment_10 = matrix_fragment_28;

            matrix_fragment_11 = matrix_fragment_29;
        }

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_78, matrix_fragment_79), matrix_fragment_4,
                matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                error_counts);

            matrix_fragment_4 = matrix_accumulator_14;

            matrix_fragment_5 = matrix_accumulator_15;

            matrix_fragment_6 = matrix_fragment_16;

            matrix_fragment_7 = matrix_fragment_17;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_76, matrix_fragment_77) + 0x8200ull,
                matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                memory_ranges, error_counts);
        }

        if (!condition_2) {
            matrix_fragment_4 = matrix_fragment_18;

            matrix_fragment_5 = matrix_fragment_19;

            matrix_fragment_6 = matrix_accumulator_20;

            matrix_fragment_7 = matrix_accumulator_21;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_74, matrix_fragment_75), matrix_fragment_4,
                matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                error_counts);

            matrix_fragment_4 = matrix_accumulator_22;

            matrix_fragment_5 = matrix_accumulator_23;

            matrix_fragment_6 = matrix_fragment_24;

            matrix_fragment_7 = matrix_fragment_25;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x8200ull,
                matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                memory_ranges, error_counts);
        }

        if (!condition_3) {
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_86, matrix_accumulator_87) + 0x8000ull,
                matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                memory_ranges, error_counts);

            matrix_fragment_4 = matrix_fragment_30;

            matrix_fragment_5 = matrix_fragment_31;

            matrix_fragment_6 = matrix_accumulator_32;

            matrix_fragment_7 = matrix_accumulator_33;

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_86, matrix_accumulator_87) + 0x8200ull,
                matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                memory_ranges, error_counts);
        }
    } else {
        condition_0 = uint32_t(index_word_144) != 0u;
        if (!condition_0) {
            split_address_words(matrix_fragment_68, matrix_fragment_69,
                                combine_words(uniform_address_word_12, uniform_address_word_13));

            matrix_fragment_69 = pipeline_read<CheckBounds>(
                combine_words(matrix_fragment_68, matrix_fragment_69), memory_ranges, error_counts);

            matrix_fragment_70 = matrix_accumulator_132 + 0xffffffffu;

            condition_0 = int32_t(matrix_fragment_69) >= int32_t(matrix_fragment_70);
            if (!condition_0) {
                do {
                    split_address_words(
                        matrix_fragment_68, matrix_fragment_69,
                        combine_words(uniform_address_word_12, uniform_address_word_13));

                    __nanosleep(0x40u);

                    matrix_fragment_69 = pipeline_read<CheckBounds>(
                        combine_words(matrix_fragment_68, matrix_fragment_69), memory_ranges,
                        error_counts);

                    condition_0 = int32_t(matrix_fragment_69) >= int32_t(matrix_fragment_70);
                } while (!condition_0);
            }
        }
        __syncwarp(__activemask());

        address_word_37 = address_word_37 << (0x5u & 31u);

        __syncthreads();

        matrix_fragment_68 = uniform_index_word_5;

        index_word_145 = index_word_145 + uniform_index_word_17;

        address_word_37 = address_word_37 & 0x20u;

        matrix_accumulator_84 = 0;
        matrix_accumulator_85 = 0;

        split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                            combine_words(uniform_address_word_18, uniform_address_word_19));

        address_word_86 = 0;
        matrix_accumulator_87 = 0;

        condition_1 = int32_t(index_word_145) >= int32_t(uniform_index_word_16);

        address_word_37 = matrix_fragment_68 * 0x40u + address_word_37;

        condition_0 = int32_t(index_word_145) >= int32_t(uniform_index_word_16);

        matrix_fragment_72 = 0;
        matrix_fragment_73 = 0;

        matrix_fragment_74 = 0;
        matrix_fragment_75 = 0;

        matrix_fragment_68 = address_word_37 * 0x8u;

        address_word_37 = (index_word_145 << 0xdu) + matrix_fragment_68;

        split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                            int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_94, matrix_accumulator_95));

        split_address_words(matrix_fragment_78, matrix_fragment_79,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x200u);

        if (!condition_0) {
            split_address_words(matrix_fragment_76, matrix_fragment_77,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_94, matrix_accumulator_95));
        }

        if (!condition_1) {
            split_address_words(matrix_fragment_78, matrix_fragment_79,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_78, matrix_fragment_79));
        }

        address_word_37 = index_word_145 + 0x1u;

        if (!condition_0) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_76, matrix_fragment_77), matrix_fragment_72,
                matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
                error_counts);
        }

        if (!condition_1) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_78, matrix_fragment_79), matrix_accumulator_84,
                matrix_accumulator_85, address_word_86, matrix_accumulator_87, memory_ranges,
                error_counts);
        }

        condition_0 = int32_t(address_word_37) >= int32_t(uniform_index_word_16);

        split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x8200u);

        split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x8000u);

        matrix_fragment_80 = 0;
        matrix_fragment_81 = 0;

        matrix_fragment_82 = 0;
        matrix_fragment_83 = 0;

        matrix_fragment_68 = 0;
        matrix_fragment_69 = 0;

        matrix_fragment_70 = 0;
        matrix_fragment_71 = 0;

        if (!condition_0) {
            split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_92, matrix_accumulator_93));

            split_address_words(matrix_fragment_78, matrix_fragment_79,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_88, matrix_accumulator_89));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_92, matrix_accumulator_93), matrix_fragment_80,
                matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
                error_counts);
        }

        address_word_37 = index_word_145 + 0x2u;

        if (!condition_0) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_78, matrix_fragment_79), matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);
        }

        split_address_words(matrix_accumulator_98, matrix_accumulator_99,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x10200u);

        condition_0 = int32_t(address_word_37) >= int32_t(uniform_index_word_16);

        split_address_words(matrix_accumulator_134, matrix_accumulator_135,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x10000u);

        matrix_fragment_76 = 0;
        matrix_fragment_77 = 0;

        matrix_accumulator_88 = 0;
        matrix_accumulator_89 = 0;

        matrix_accumulator_90 = 0;
        matrix_accumulator_91 = 0;

        address_word_37 = index_word_145 + 0x3u;

        matrix_fragment_78 = 0;
        matrix_fragment_79 = 0;

        if (!condition_0) {
            split_address_words(matrix_accumulator_98, matrix_accumulator_99,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_98, matrix_accumulator_99));

            split_address_words(matrix_accumulator_134, matrix_accumulator_135,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_134, matrix_accumulator_135));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_98, matrix_accumulator_99), matrix_fragment_76,
                matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
                error_counts);
        }

        condition_1 = int32_t(address_word_37) >= int32_t(uniform_index_word_16);

        if (!condition_0) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_134, matrix_accumulator_135),
                matrix_accumulator_88, matrix_accumulator_89, matrix_accumulator_90,
                matrix_accumulator_91, memory_ranges, error_counts);
        }

        split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x18000u);

        matrix_accumulator_96 = 0;
        matrix_accumulator_97 = 0;

        matrix_accumulator_92 = 0;
        matrix_accumulator_93 = 0;

        matrix_accumulator_98 = 0;
        matrix_accumulator_99 = 0;

        if (!condition_1) {
            split_address_words(matrix_accumulator_138, matrix_accumulator_139,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_94, matrix_accumulator_95));
        }

        matrix_accumulator_134 = parameters[parameter_trained_weights_low_8];
        matrix_accumulator_135 = parameters[parameter_trained_weights_high_9];

        if (!condition_1) {
            split_address_words(matrix_accumulator_136, matrix_accumulator_137,
                                int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_94, matrix_accumulator_95));
        }

        matrix_accumulator_94 = 0;
        matrix_accumulator_95 = 0;

        if (!condition_1) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x200ull,
                matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_98,
                matrix_accumulator_99, memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_136, matrix_accumulator_137),
                matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
                matrix_accumulator_95, memory_ranges, error_counts);
        }

        index_word_0 = threadIdx.y;

        matrix_accumulator_140 = uniform_index_word_5;

        address_word_37 = index_word_0 & 0x1u;

        address_word_37 = matrix_accumulator_140 * 0x2u + address_word_37;

        split_address_words(matrix_accumulator_134, matrix_accumulator_135,
                            int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_134, matrix_accumulator_135));

        packed_half_133 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_134, matrix_accumulator_135), memory_ranges,
            error_counts);

        matrix_fragment_72 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                           select_half_operand<2, false, false>(matrix_accumulator_130));

        matrix_accumulator_84 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                           select_half_operand<2, false, false>(matrix_accumulator_124));

        address_word_86 =
            add_half_pairs(select_half_operand<2, false, false>(address_word_86),
                           select_half_operand<2, false, false>(matrix_accumulator_122));

        matrix_accumulator_85 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                           select_half_operand<2, false, false>(matrix_accumulator_125));

        matrix_accumulator_87 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                           select_half_operand<2, false, false>(matrix_accumulator_123));

        matrix_fragment_74 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                           select_half_operand<2, false, false>(matrix_accumulator_128));

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                select_half_operand<2, false, false>(matrix_accumulator_84));

        matrix_accumulator_124 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_86),
                                select_half_operand<2, false, false>(address_word_86));

        matrix_fragment_73 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                           select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_fragment_75 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                           select_half_operand<2, false, false>(matrix_accumulator_129));

        matrix_accumulator_123 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_125 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                select_half_operand<2, false, false>(matrix_accumulator_87));

        matrix_accumulator_122 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_accumulator_122));

        matrix_accumulator_124 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_accumulator_124));

        matrix_accumulator_123 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73),
            select_half_operand<2, false, false>(matrix_fragment_73),
            select_half_operand<2, false, false>(matrix_accumulator_123));

        matrix_accumulator_125 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75),
            select_half_operand<2, false, false>(matrix_fragment_75),
            select_half_operand<2, false, false>(matrix_accumulator_125));

        matrix_accumulator_122 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                           select_half_operand<2, false, false>(matrix_accumulator_122));

        matrix_accumulator_123 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                           select_half_operand<2, false, false>(matrix_accumulator_123));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_122), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_125 = result_word;
        }

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_123), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_124 = result_word;
        }

        matrix_fragment_80 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                           select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_fragment_82 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                           select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_fragment_68 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                           select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_70 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                           select_half_operand<2, false, false>(matrix_accumulator_112));

        matrix_accumulator_108 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<2, false, false>(matrix_fragment_80));

        matrix_accumulator_110 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                select_half_operand<2, false, false>(matrix_fragment_82));

        matrix_accumulator_108 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_accumulator_110 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_accumulator_110 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                           select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_accumulator_122 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                           select_half_operand<2, false, false>(matrix_accumulator_125));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_110), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_125 = result_word;
        }

        matrix_accumulator_124 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                           select_half_operand<2, false, false>(matrix_accumulator_124));

        matrix_fragment_81 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                           select_half_operand<2, false, false>(matrix_accumulator_111));

        matrix_fragment_83 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                           select_half_operand<2, false, false>(matrix_accumulator_109));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_124), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_111 = result_word;
        }

        matrix_fragment_69 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                           select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_fragment_71 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                           select_half_operand<2, false, false>(matrix_accumulator_113));

        matrix_accumulator_108 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<2, false, false>(matrix_fragment_81));

        matrix_accumulator_109 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_accumulator_108 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_accumulator_109 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71),
            select_half_operand<2, false, false>(matrix_fragment_71),
            select_half_operand<2, false, false>(matrix_accumulator_109));

        matrix_fragment_76 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                           select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_fragment_78 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                           select_half_operand<2, false, false>(address_word_100));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_122), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_123 = result_word;
        }

        matrix_accumulator_108 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                           select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_accumulator_88 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                           select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_accumulator_90 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                           select_half_operand<2, false, false>(matrix_accumulator_104));

        address_word_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_fragment_76));

        matrix_accumulator_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                select_half_operand<2, false, false>(matrix_fragment_78));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_108), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_113 = result_word;
        }

        address_word_100 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_88),
            select_half_operand<2, false, false>(matrix_accumulator_88),
            select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_90),
            select_half_operand<2, false, false>(matrix_accumulator_90),
            select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_accumulator_110 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                           select_half_operand<2, false, false>(matrix_accumulator_125));

        matrix_accumulator_102 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                           select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_91 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                           select_half_operand<2, false, false>(matrix_accumulator_105));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_110), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_109 = result_word;
        }

        matrix_fragment_77 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                           select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_79 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                           select_half_operand<2, false, false>(matrix_accumulator_101));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_102), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_105 = result_word;
        }

        matrix_accumulator_111 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                           select_half_operand<2, false, false>(matrix_accumulator_111));

        matrix_accumulator_89 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                           select_half_operand<2, false, false>(matrix_accumulator_107));

        address_word_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_fragment_77));

        matrix_accumulator_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                select_half_operand<2, false, false>(matrix_fragment_79));

        matrix_accumulator_104 =
            permute_signed(matrix_accumulator_111, matrix_accumulator_111, 0x5432u);

        address_word_100 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_89),
            select_half_operand<2, false, false>(matrix_accumulator_89),
            select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_91),
            select_half_operand<2, false, false>(matrix_accumulator_91),
            select_half_operand<2, false, false>(matrix_accumulator_101));

        matrix_accumulator_122 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                           select_half_operand<2, false, false>(matrix_accumulator_123));

        matrix_accumulator_96 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                           select_half_operand<2, false, false>(matrix_accumulator_120));

        matrix_accumulator_98 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                           select_half_operand<2, false, false>(matrix_accumulator_126));

        matrix_accumulator_111 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                           select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_accumulator_104 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                           select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_115 =
            permute_signed(matrix_accumulator_122, matrix_accumulator_122, 0x5432u);

        matrix_accumulator_92 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                           select_half_operand<2, false, false>(matrix_accumulator_116));

        matrix_accumulator_94 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                           select_half_operand<2, false, false>(matrix_accumulator_118));

        address_word_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_accumulator_96));

        matrix_accumulator_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_accumulator_98));

        matrix_accumulator_108 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                           select_half_operand<2, false, false>(matrix_accumulator_113));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_104), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_107 = result_word;
        }

        matrix_accumulator_99 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                           select_half_operand<2, false, false>(matrix_accumulator_127));

        address_word_100 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94),
            select_half_operand<2, false, false>(matrix_accumulator_94),
            select_half_operand<2, false, false>(matrix_accumulator_101));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_108), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_103 = result_word;
        }

        matrix_accumulator_122 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                           select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_113 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(matrix_accumulator_99));

        matrix_accumulator_95 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                           select_half_operand<2, false, false>(matrix_accumulator_119));

        matrix_accumulator_106 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                           select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_97 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                           select_half_operand<2, false, false>(matrix_accumulator_121));

        matrix_accumulator_122 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x4100410u);

        matrix_accumulator_109 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                           select_half_operand<2, false, false>(matrix_accumulator_109));

        matrix_accumulator_110 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                           select_half_operand<2, false, false>(matrix_accumulator_105));

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95),
            select_half_operand<2, false, false>(matrix_accumulator_95),
            select_half_operand<2, false, false>(matrix_accumulator_113));

        matrix_accumulator_93 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                           select_half_operand<2, false, false>(matrix_accumulator_117));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_106), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_113 = result_word;
        }

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_112 = half_to_float_bits((matrix_accumulator_122 & 65535u) * 65537u);

        matrix_accumulator_105 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x4100410u);

        matrix_accumulator_114 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93),
            select_half_operand<2, false, false>(matrix_accumulator_93),
            select_half_operand<2, false, false>(matrix_accumulator_114));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_110), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_111 = result_word;
        }

        address_word_100 = rsqrt_bits(matrix_accumulator_112);

        matrix_accumulator_104 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                           select_half_operand<2, false, false>(matrix_accumulator_107));

        matrix_accumulator_112 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                           select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_114 =
            permute_signed(matrix_accumulator_109, matrix_accumulator_109, 0x5432u);

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_112), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_115 = result_word;
        }

        matrix_accumulator_109 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                           select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_103 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                           select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_accumulator_108 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_109), 0x4100410u);

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_104), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_109 = result_word;
        }

        matrix_accumulator_114 =
            permute_signed(matrix_accumulator_103, matrix_accumulator_103, 0x5432u);

        matrix_accumulator_113 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                           select_half_operand<2, false, false>(matrix_accumulator_113));

        matrix_accumulator_110 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                           select_half_operand<2, false, false>(matrix_accumulator_111));

        matrix_accumulator_103 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                           select_half_operand<2, false, false>(matrix_accumulator_114));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_113), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_114 = result_word;
        }

        matrix_accumulator_111 =
            permute_signed(matrix_accumulator_110, matrix_accumulator_110, 0x5432u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0x4100410u);

        matrix_accumulator_102 = half_to_float_bits((matrix_accumulator_105 & 65535u) * 65537u);

        matrix_accumulator_107 = half_to_float_bits((matrix_accumulator_105 >> 16) * 65537u);

        matrix_accumulator_112 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                           select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_105 = half_to_float_bits((matrix_accumulator_108 & 65535u) * 65537u);

        matrix_accumulator_110 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                           select_half_operand<2, false, false>(matrix_accumulator_111));

        matrix_accumulator_106 = half_to_float_bits((matrix_accumulator_108 >> 16) * 65537u);

        matrix_accumulator_108 = half_to_float_bits((matrix_accumulator_103 & 65535u) * 65537u);

        matrix_accumulator_111 = half_to_float_bits((matrix_accumulator_103 >> 16) * 65537u);

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_112), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_103 = result_word;
        }

        matrix_accumulator_104 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                           select_half_operand<2, false, false>(matrix_accumulator_109));

        matrix_accumulator_115 =
            permute_signed(matrix_accumulator_104, matrix_accumulator_104, 0x5432u);

        matrix_accumulator_113 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                           select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_104 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                           select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_116 =
            permute_signed(matrix_accumulator_113, matrix_accumulator_113, 0x5432u);

        matrix_accumulator_101 = half_to_float_bits((matrix_accumulator_122 >> 16) * 65537u);

        matrix_accumulator_104 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x4100410u);

        matrix_accumulator_113 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                           select_half_operand<2, false, false>(matrix_accumulator_116));

        matrix_accumulator_110 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x4100410u);

        matrix_accumulator_114 = half_to_float_bits((matrix_accumulator_104 & 65535u) * 65537u);

        matrix_accumulator_116 = half_to_float_bits((matrix_accumulator_104 >> 16) * 65537u);

        matrix_accumulator_103 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                           select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_accumulator_104 = index_word_0 * 0x20u;

        matrix_accumulator_101 = rsqrt_bits(matrix_accumulator_101);

        matrix_accumulator_109 = half_to_float_bits((matrix_accumulator_110 & 65535u) * 65537u);

        matrix_accumulator_118 = matrix_accumulator_104 & 0x20u;

        matrix_accumulator_104 =
            permute_signed(matrix_accumulator_103, matrix_accumulator_103, 0x5432u);

        matrix_accumulator_115 = rsqrt_bits(matrix_accumulator_109);

        matrix_accumulator_104 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                           select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_accumulator_109 = index_word_0 << (0x1u & 31u);

        matrix_accumulator_117 = rsqrt_bits(matrix_accumulator_116);

        matrix_accumulator_101 = pack_half(matrix_accumulator_101, address_word_100);

        matrix_accumulator_102 = rsqrt_bits(matrix_accumulator_102);

        matrix_accumulator_116 = matrix_accumulator_109 & 0x7fcu;

        matrix_accumulator_109 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x4100410u);

        matrix_accumulator_107 = rsqrt_bits(matrix_accumulator_107);

        address_word_100 = half_to_float_bits((matrix_accumulator_109 & 65535u) * 65537u);

        matrix_accumulator_109 = half_to_float_bits((matrix_accumulator_109 >> 16) * 65537u);

        matrix_accumulator_105 = rsqrt_bits(matrix_accumulator_105);

        matrix_accumulator_106 = rsqrt_bits(matrix_accumulator_106);

        matrix_accumulator_113 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_113), 0x4100410u);

        matrix_accumulator_108 = rsqrt_bits(matrix_accumulator_108);

        matrix_accumulator_119 = uniform_index_word_5;

        matrix_accumulator_111 = rsqrt_bits(matrix_accumulator_111);

        matrix_accumulator_112 = half_to_float_bits((matrix_accumulator_113 & 65535u) * 65537u);

        matrix_accumulator_103 = matrix_accumulator_119 * 0x40u + matrix_accumulator_118;

        matrix_accumulator_119 = uniform_parameter_word_7;

        address_word_100 = rsqrt_bits(address_word_100);

        matrix_accumulator_110 = half_to_float_bits((matrix_accumulator_110 >> 16) * 65537u);

        matrix_accumulator_102 = pack_half(matrix_accumulator_107, matrix_accumulator_102);

        matrix_accumulator_113 = half_to_float_bits((matrix_accumulator_113 >> 16) * 65537u);

        matrix_accumulator_109 = rsqrt_bits(matrix_accumulator_109);

        matrix_accumulator_106 = pack_half(matrix_accumulator_106, matrix_accumulator_105);

        matrix_accumulator_104 = matrix_accumulator_119 * 0x8u + matrix_accumulator_116;

        matrix_accumulator_108 = pack_half(matrix_accumulator_111, matrix_accumulator_108);

        matrix_accumulator_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_accumulator_87 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_accumulator_110 = rsqrt_bits(matrix_accumulator_110);

        matrix_accumulator_105 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        packed_half_133 = pack_half(0u, packed_half_133);

        matrix_fragment_80 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                select_half_operand<2, false, false>(matrix_accumulator_108));

        condition_0 = int32_t(matrix_accumulator_104) >= int32_t(uniform_index_word_4);

        matrix_accumulator_114 = rsqrt_bits(matrix_accumulator_114);

        matrix_accumulator_85 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_85), 0x45a845a8u);

        matrix_accumulator_87 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_87), 0x45a845a8u);

        matrix_accumulator_112 = rsqrt_bits(matrix_accumulator_112);

        matrix_fragment_74 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                select_half_operand<2, false, false>(matrix_accumulator_101));

        matrix_fragment_73 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_accumulator_113 = rsqrt_bits(matrix_accumulator_113);

        matrix_fragment_72 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<2, false, false>(matrix_accumulator_101));

        matrix_fragment_75 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        address_word_100 = pack_half(matrix_accumulator_109, address_word_100);

        matrix_accumulator_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                select_half_operand<2, false, false>(matrix_accumulator_101));

        address_word_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_86),
                                select_half_operand<2, false, false>(matrix_accumulator_101));

        matrix_accumulator_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_accumulator_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_87 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_80 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80), 0x45a845a8u);

        matrix_accumulator_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_fragment_81 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_fragment_70 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0x45a845a8u);

        matrix_fragment_69 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0x45a845a8u);

        matrix_fragment_68 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x45a845a8u);

        matrix_fragment_75 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0x45a845a8u);

        matrix_fragment_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_accumulator_85 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(address_word_100));

        matrix_accumulator_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(address_word_100));

        matrix_fragment_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_81 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81), 0x45a845a8u);

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                select_half_operand<0, false, false>(packed_half_133));

        address_word_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                select_half_operand<0, false, false>(packed_half_133));

        if (!condition_0) {
            matrix_accumulator_85 =
                permute_signed(matrix_accumulator_85, matrix_fragment_80, 0x5410u);
        }

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_110 = pack_half(matrix_accumulator_110, matrix_accumulator_115);

        matrix_accumulator_114 = pack_half(matrix_accumulator_117, matrix_accumulator_114);

        matrix_accumulator_112 = pack_half(matrix_accumulator_113, matrix_accumulator_112);

        matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(address_word_100),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                           select_half_operand<2, false, false>(0u));

        address_word_100 = threadIdx.x & 31u;

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<2, false, false>(matrix_accumulator_108));

        if (!condition_0) {
            matrix_fragment_69 = permute_signed(matrix_fragment_80, matrix_accumulator_87, 0x5410u);
        }

        matrix_fragment_83 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<2, false, false>(matrix_accumulator_108));

        if (!condition_0) {
            matrix_fragment_68 = permute_signed(matrix_fragment_68, matrix_fragment_81, 0x5410u);
        }

        matrix_accumulator_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                                select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_fragment_80 = parameters[parameter_output_q_low_2];
        matrix_fragment_81 = parameters[parameter_output_q_high_3];

        matrix_accumulator_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                                select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_fragment_76 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_fragment_78 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                select_half_operand<2, false, false>(matrix_accumulator_110));

        matrix_accumulator_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_77 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_79 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_92 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_accumulator_112));

        matrix_accumulator_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                select_half_operand<2, false, false>(matrix_accumulator_112));

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_accumulator_112));

        matrix_accumulator_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_accumulator_112));

        matrix_accumulator_84 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_84), 0x45a845a8u);

        address_word_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_86), 0x45a845a8u);

        matrix_accumulator_101 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101), 0x45a845a8u);

        matrix_accumulator_102 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x45a845a8u);

        matrix_fragment_71 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71), 0x45a845a8u);

        matrix_accumulator_105 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_105), 0x45a845a8u);

        matrix_fragment_82 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0x45a845a8u);

        matrix_fragment_83 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0x45a845a8u);

        matrix_fragment_73 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_88), 0x45a845a8u);

        matrix_accumulator_89 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_89), 0x45a845a8u);

        matrix_fragment_72 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_90), 0x45a845a8u);

        matrix_accumulator_91 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_91), 0x45a845a8u);

        matrix_fragment_76 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_76), 0x45a845a8u);

        matrix_fragment_77 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_77), 0x45a845a8u);

        matrix_fragment_78 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78), 0x45a845a8u);

        matrix_fragment_79 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_79), 0x45a845a8u);

        matrix_fragment_74 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x45a845a8u);

        matrix_fragment_75 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0x45a845a8u);

        matrix_accumulator_96 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x45a845a8u);

        matrix_accumulator_98 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0x45a845a8u);

        matrix_accumulator_93 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x45a845a8u);

        matrix_accumulator_95 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95), 0x45a845a8u);

        matrix_accumulator_97 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x45a845a8u);

        matrix_accumulator_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                select_half_operand<0, false, false>(packed_half_133));

        address_word_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_86),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_99 = multiply_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99), 0x45a845a8u);

        matrix_accumulator_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_105 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_83 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_73 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_72 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_76 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_77 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_78 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_79 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_74 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_fragment_75 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                          select_half_operand<2, false, false>(0u));

        packed_half_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<0, false, false>(packed_half_133));

        matrix_accumulator_99 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_86),
                                              select_half_operand<2, false, false>(0u));

        matrix_accumulator_112 = matrix_accumulator_104 + 0x1u;

        matrix_accumulator_110 = matrix_accumulator_104 + 0x2u;

        if (!condition_0) {
            matrix_fragment_70 =
                permute_signed(matrix_accumulator_84, matrix_accumulator_99, 0x5410u);
        }

        matrix_accumulator_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                          select_half_operand<2, false, false>(0u));

        condition_1 = int32_t(matrix_accumulator_112) >= int32_t(uniform_index_word_4);

        condition_2 = int32_t(matrix_accumulator_110) >= int32_t(uniform_index_word_4);

        matrix_accumulator_99 = matrix_accumulator_104 + 0x3u;

        matrix_accumulator_102 = (matrix_accumulator_104 << 0x5u) + address_word_37;

        condition_3 = int32_t(matrix_accumulator_99) >= int32_t(uniform_index_word_4);

        matrix_accumulator_102 = matrix_accumulator_102 * 0x80u;

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(0u));

        uniform_index_word_4 = parameters[parameter_LH_18];
        uniform_index_word_5 = parameters[parameter_LW_19];

        address_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                        select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_80, matrix_fragment_81,
                            int64_t(int32_t(matrix_accumulator_102)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_80, matrix_fragment_81));

        address_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(0u));

        matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_84 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_101 = matrix_accumulator_103 * 0x8u;

        matrix_accumulator_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                           select_half_operand<2, false, false>(0u));

        if (!condition_1) {
            matrix_fragment_79 = permute_signed(matrix_accumulator_107, address_word_86, 0x5410u);
        }

        if (!condition_2) {
            matrix_fragment_78 = permute_signed(address_word_37, matrix_fragment_78, 0x5410u);

            address_word_86 = permute_signed(matrix_fragment_77, matrix_accumulator_88, 0x5410u);
        }

        matrix_accumulator_96 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(0u));

        address_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                        select_half_operand<2, false, false>(0u));

        if (!condition_1) {
            matrix_fragment_83 =
                permute_signed(matrix_accumulator_106, matrix_fragment_71, 0x5410u);
        }

        if (!condition_2) {
            matrix_accumulator_88 = permute_signed(matrix_fragment_73, matrix_fragment_72, 0x5410u);
        }

        if (!condition_0) {
            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_80, matrix_fragment_81));
        }

        if (!condition_1) {
            matrix_accumulator_84 =
                permute_signed(matrix_accumulator_84, matrix_accumulator_87, 0x5410u);
        }

        matrix_accumulator_94 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_95 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                          select_half_operand<2, false, false>(0u));

        if (!condition_0) {
            matrix_fragment_71 = matrix_accumulator_85;
        }

        if (!condition_1) {
            matrix_fragment_82 =
                permute_signed(matrix_accumulator_105, matrix_fragment_82, 0x5410u);
        }

        matrix_accumulator_87 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_76 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                           select_half_operand<2, false, false>(0u));

        if (!condition_3) {
            matrix_accumulator_93 = permute_signed(matrix_accumulator_96, address_word_37, 0x5410u);
        }

        matrix_accumulator_108 = 0x4u;

        address_word_37 = (matrix_accumulator_104 << 0xdu) + matrix_accumulator_101;

        if (!condition_0) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73), matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);
        }

        if (!condition_3) {
            matrix_accumulator_94 =
                permute_signed(matrix_accumulator_94, matrix_accumulator_95, 0x5410u);
        }

        if (!condition_2) {
            matrix_accumulator_87 =
                permute_signed(matrix_accumulator_87, matrix_fragment_76, 0x5410u);
        }

        if (!condition_1) {
            split_address_words(matrix_fragment_76, matrix_fragment_77,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_80, matrix_fragment_81));
        }

        if (!condition_3) {
            matrix_accumulator_95 = permute_signed(matrix_fragment_74, matrix_fragment_75, 0x5410u);
        }

        if (!condition_1) {
            matrix_fragment_74 = matrix_fragment_82;

            matrix_fragment_75 = matrix_fragment_79;
        }

        condition_4 = int32_t(matrix_accumulator_104) >= int32_t(uniform_index_word_16);

        split_address_words(matrix_accumulator_108, matrix_accumulator_109,
                            int64_t(int32_t(address_word_37)) *
                                    int64_t(int32_t(matrix_accumulator_108)) +
                                combine_words(uniform_address_word_8, uniform_address_word_9));

        matrix_accumulator_92 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(0u));

        packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                                        select_half_operand<2, false, false>(0u));

        if (!condition_1) {
            matrix_fragment_72 = matrix_accumulator_84;

            matrix_fragment_73 = matrix_fragment_83;
        }

        if (!condition_2) {
            split_address_words(matrix_fragment_82, matrix_fragment_83,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_80, matrix_fragment_81));

            matrix_fragment_79 = address_word_86;
        }

        if (!condition_3) {
            matrix_accumulator_92 = permute_signed(matrix_accumulator_92, packed_half_133, 0x5410u);
        }

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_76, matrix_fragment_77) + 0x4000ull,
                matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                memory_ranges, error_counts);
        }

        split_address_words(matrix_accumulator_90, matrix_accumulator_91,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) + 0x200u);

        matrix_fragment_68 = 0;
        matrix_fragment_69 = 0;

        matrix_fragment_70 = 0;
        matrix_fragment_71 = 0;

        if (!condition_3) {
            matrix_accumulator_84 = matrix_accumulator_95;

            matrix_accumulator_85 = matrix_accumulator_94;

            address_word_86 = matrix_accumulator_93;
        }

        if (!condition_4) {
            split_address_words(matrix_accumulator_90, matrix_accumulator_91,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_90, matrix_accumulator_91));
        }

        if (!condition_2) {
            matrix_fragment_76 = matrix_accumulator_88;

            matrix_fragment_77 = matrix_accumulator_87;
        }

        if (!condition_3) {
            matrix_accumulator_87 = matrix_accumulator_92;
        }

        if (!condition_2) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_82, matrix_fragment_83) + 0x8000ull,
                matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                memory_ranges, error_counts);
        }

        matrix_fragment_72 = 0;
        matrix_fragment_73 = 0;

        matrix_fragment_74 = 0;
        matrix_fragment_75 = 0;

        if (!condition_4) {
            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_108, matrix_accumulator_109));
        }

        if (!condition_3) {
            split_address_words(matrix_fragment_82, matrix_fragment_83,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_80, matrix_fragment_81));

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_82, matrix_fragment_83) + 0xc000ull,
                matrix_accumulator_84, matrix_accumulator_85, address_word_86,
                matrix_accumulator_87, memory_ranges, error_counts);
        }

        if (!condition_4) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_90, matrix_accumulator_91), matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_88, matrix_accumulator_89), matrix_fragment_72,
                matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
                error_counts);
        }

        matrix_accumulator_111 = uint32_t(condition_4) << 4;

        condition_4 = int32_t(matrix_accumulator_112) >= int32_t(uniform_index_word_16);

        split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) +
                                0x8200u);

        split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) +
                                0x8000u);

        matrix_fragment_80 = 0;
        matrix_fragment_81 = 0;

        matrix_fragment_82 = 0;
        matrix_fragment_83 = 0;

        matrix_fragment_76 = 0;
        matrix_fragment_77 = 0;

        matrix_fragment_78 = 0;
        matrix_fragment_79 = 0;

        if (!condition_4) {
            split_address_words(matrix_accumulator_90, matrix_accumulator_91,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_94, matrix_accumulator_95));

            split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_92, matrix_accumulator_93));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_90, matrix_accumulator_91), matrix_fragment_80,
                matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_92, matrix_accumulator_93), matrix_fragment_76,
                matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
                error_counts);
        }

        condition_5 = int32_t(matrix_accumulator_110) >= int32_t(uniform_index_word_16);

        split_address_words(matrix_accumulator_106, matrix_accumulator_107,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) +
                                0x10200u);

        split_address_words(matrix_accumulator_104, matrix_accumulator_105,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) +
                                0x10000u);

        matrix_accumulator_88 = 0;
        matrix_accumulator_89 = 0;

        matrix_accumulator_90 = 0;
        matrix_accumulator_91 = 0;

        matrix_accumulator_84 = 0;
        matrix_accumulator_85 = 0;

        address_word_86 = 0;
        matrix_accumulator_87 = 0;

        if (!condition_5) {
            split_address_words(matrix_accumulator_106, matrix_accumulator_107,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_106, matrix_accumulator_107));

            split_address_words(matrix_accumulator_104, matrix_accumulator_105,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_104, matrix_accumulator_105));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_106, matrix_accumulator_107),
                matrix_accumulator_88, matrix_accumulator_89, matrix_accumulator_90,
                matrix_accumulator_91, memory_ranges, error_counts);
        }

        condition_6 = int32_t(matrix_accumulator_99) >= int32_t(uniform_index_word_16);

        if (!condition_5) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_104, matrix_accumulator_105),
                matrix_accumulator_84, matrix_accumulator_85, address_word_86,
                matrix_accumulator_87, memory_ranges, error_counts);
        }

        split_address_words(matrix_accumulator_108, matrix_accumulator_109,
                            combine_words(matrix_accumulator_108, matrix_accumulator_109) +
                                0x18000u);

        matrix_accumulator_96 = 0;
        matrix_accumulator_97 = 0;

        matrix_accumulator_98 = 0;
        matrix_accumulator_99 = 0;

        matrix_accumulator_92 = 0;
        matrix_accumulator_93 = 0;

        matrix_accumulator_94 = 0;
        matrix_accumulator_95 = 0;

        if (!condition_6) {
            split_address_words(matrix_accumulator_104, matrix_accumulator_105,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_108, matrix_accumulator_109));

            split_address_words(matrix_accumulator_108, matrix_accumulator_109,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_108, matrix_accumulator_109));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_104, matrix_accumulator_105),
                matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
                matrix_accumulator_95, memory_ranges, error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_accumulator_108, matrix_accumulator_109) + 0x200ull,
                matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_98,
                matrix_accumulator_99, memory_ranges, error_counts);
        }

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(packed_half_38));

        matrix_fragment_70 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                           select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_42 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                           select_half_operand<2, false, false>(matrix_accumulator_42));

        matrix_accumulator_40 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                           select_half_operand<2, false, false>(matrix_accumulator_40));

        packed_half_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                             select_half_operand<2, false, false>(packed_half_38));

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        packed_half_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_42),
            select_half_operand<2, false, false>(matrix_accumulator_42),
            select_half_operand<2, false, false>(packed_half_34));

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_40),
            select_half_operand<2, false, false>(matrix_accumulator_40),
            select_half_operand<2, false, false>(matrix_fragment_68));

        packed_half_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(packed_half_39));

        matrix_fragment_71 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                           select_half_operand<2, false, false>(packed_half_35));

        matrix_fragment_68 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                           select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_43 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                           select_half_operand<2, false, false>(matrix_accumulator_43));

        packed_half_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(packed_half_41));

        packed_half_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                                             select_half_operand<2, false, false>(packed_half_39));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        packed_half_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_43),
            select_half_operand<2, false, false>(matrix_accumulator_43),
            select_half_operand<2, false, false>(packed_half_34));

        packed_half_35 =
            fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_41),
                                          select_half_operand<2, false, false>(packed_half_41),
                                          select_half_operand<2, false, false>(packed_half_35));

        packed_half_34 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                        select_half_operand<2, false, false>(packed_half_34));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_69 = result_word;
        }

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_34), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            packed_half_35 = result_word;
        }

        matrix_fragment_69 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                           select_half_operand<2, false, false>(matrix_fragment_69));

        packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(packed_half_34),
                                        select_half_operand<2, false, false>(packed_half_35));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_72 = result_word;
        }

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_35), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_74 = result_word;
        }

        matrix_fragment_72 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                           select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_74 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                           select_half_operand<2, false, false>(matrix_fragment_74));

        matrix_fragment_69 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

        packed_half_35 = permute_signed(matrix_fragment_74, matrix_fragment_74, 0x5432u);

        matrix_fragment_72 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                           select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_74 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                           select_half_operand<2, false, false>(packed_half_35));

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x4100410u);

        packed_half_34 = parameters[parameter_output_k_low_4];
        packed_half_35 = parameters[parameter_output_k_high_5];

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0x4100410u);

        matrix_fragment_68 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

        matrix_fragment_72 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

        matrix_fragment_73 = half_to_float_bits((matrix_fragment_74 & 65535u) * 65537u);

        matrix_fragment_74 = half_to_float_bits((matrix_fragment_74 >> 16) * 65537u);

        matrix_fragment_68 = rsqrt_bits(matrix_fragment_68);

        matrix_fragment_69 = rsqrt_bits(matrix_fragment_72);

        matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

        matrix_fragment_74 = rsqrt_bits(matrix_fragment_74);

        matrix_fragment_69 = pack_half(matrix_fragment_69, matrix_fragment_68);

        matrix_accumulator_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_accumulator_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        packed_half_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_75 = pack_half(matrix_fragment_74, matrix_fragment_73);

        matrix_fragment_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_accumulator_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        packed_half_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_41),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        packed_half_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                            int64_t(int32_t(matrix_accumulator_102)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_34, packed_half_35));

        matrix_accumulator_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_39),
                                              select_half_operand<2, false, false>(0u));

        packed_half_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

        packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_41),
                                       select_half_operand<2, false, false>(0u));

        packed_half_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_38),
                                       select_half_operand<2, false, false>(0u));

        packed_half_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

        packed_half_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                                       select_half_operand<2, false, false>(0u));

        if (!condition_0) {
            matrix_accumulator_43 = permute_signed(matrix_accumulator_43, packed_half_34, 0x5410u);

            matrix_accumulator_42 = permute_signed(matrix_accumulator_42, packed_half_35, 0x5410u);

            split_address_words(packed_half_34, packed_half_35,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_102, matrix_accumulator_103));

            packed_half_41 = permute_signed(packed_half_38, packed_half_41, 0x5410u);

            matrix_accumulator_40 = permute_signed(matrix_fragment_68, packed_half_39, 0x5410u);
        }

        matrix_accumulator_48 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                           select_half_operand<2, false, false>(matrix_accumulator_48));

        matrix_accumulator_50 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                           select_half_operand<2, false, false>(matrix_accumulator_50));

        if (!condition_0) {
            store_four_words_streaming<CheckBounds>(combine_words(packed_half_34, packed_half_35),
                                                    matrix_accumulator_40, packed_half_41,
                                                    matrix_accumulator_42, matrix_accumulator_43,
                                                    memory_ranges, error_counts);
        }

        matrix_accumulator_44 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                           select_half_operand<2, false, false>(matrix_accumulator_44));

        matrix_accumulator_46 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                           select_half_operand<2, false, false>(matrix_accumulator_46));

        packed_half_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                                select_half_operand<2, false, false>(matrix_accumulator_50));

        packed_half_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_46),
            select_half_operand<2, false, false>(matrix_accumulator_46),
            select_half_operand<2, false, false>(packed_half_38));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_accumulator_48));

        packed_half_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_44),
            select_half_operand<2, false, false>(matrix_accumulator_44),
            select_half_operand<2, false, false>(packed_half_35));

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                        select_half_operand<2, false, false>(packed_half_35));

        matrix_accumulator_49 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                           select_half_operand<2, false, false>(matrix_accumulator_49));

        matrix_accumulator_51 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                           select_half_operand<2, false, false>(matrix_accumulator_51));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_38), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            packed_half_41 = result_word;
        }

        matrix_accumulator_45 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                           select_half_operand<2, false, false>(matrix_accumulator_45));

        matrix_accumulator_47 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                           select_half_operand<2, false, false>(matrix_accumulator_47));

        packed_half_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(matrix_accumulator_49));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                                select_half_operand<2, false, false>(matrix_accumulator_51));

        packed_half_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_45),
            select_half_operand<2, false, false>(matrix_accumulator_45),
            select_half_operand<2, false, false>(packed_half_34));

        packed_half_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_47),
            select_half_operand<2, false, false>(matrix_accumulator_47),
            select_half_operand<2, false, false>(packed_half_35));

        matrix_accumulator_56 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                           select_half_operand<2, false, false>(matrix_accumulator_56));

        matrix_accumulator_58 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                           select_half_operand<2, false, false>(matrix_accumulator_58));

        packed_half_39 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                        select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_52 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                           select_half_operand<2, false, false>(matrix_accumulator_52));

        matrix_accumulator_54 =
            add_half_pairs(select_half_operand<2, false, false>(address_word_86),
                           select_half_operand<2, false, false>(matrix_accumulator_54));

        packed_half_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                                select_half_operand<2, false, false>(matrix_accumulator_56));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                select_half_operand<2, false, false>(matrix_accumulator_58));

        packed_half_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_52),
            select_half_operand<2, false, false>(matrix_accumulator_52),
            select_half_operand<2, false, false>(packed_half_34));

        packed_half_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_54),
            select_half_operand<2, false, false>(matrix_accumulator_54),
            select_half_operand<2, false, false>(packed_half_35));

        matrix_accumulator_57 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                           select_half_operand<2, false, false>(matrix_accumulator_57));

        matrix_accumulator_59 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                           select_half_operand<2, false, false>(matrix_accumulator_59));

        matrix_accumulator_42 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                           select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_53 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                           select_half_operand<2, false, false>(matrix_accumulator_53));

        matrix_accumulator_55 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                           select_half_operand<2, false, false>(matrix_accumulator_55));

        packed_half_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                                select_half_operand<2, false, false>(matrix_accumulator_57));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                                select_half_operand<2, false, false>(matrix_accumulator_59));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_39), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_40 = result_word;
        }

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                        select_half_operand<2, false, false>(packed_half_41));

        packed_half_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_53),
            select_half_operand<2, false, false>(matrix_accumulator_53),
            select_half_operand<2, false, false>(packed_half_34));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_42), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_69 = result_word;
        }

        packed_half_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_55),
            select_half_operand<2, false, false>(matrix_accumulator_55),
            select_half_operand<2, false, false>(packed_half_35));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_38), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            packed_half_41 = result_word;
        }

        matrix_accumulator_43 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                           select_half_operand<2, false, false>(packed_half_34));

        matrix_fragment_64 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                           select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_66 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                           select_half_operand<2, false, false>(matrix_fragment_66));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_43), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_68 = result_word;
        }

        matrix_fragment_60 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                           select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_62 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                           select_half_operand<2, false, false>(matrix_fragment_62));

        packed_half_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        packed_half_34 =
            fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                          select_half_operand<2, false, false>(matrix_fragment_60),
                                          select_half_operand<2, false, false>(packed_half_34));

        packed_half_35 =
            fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                          select_half_operand<2, false, false>(matrix_fragment_62),
                                          select_half_operand<2, false, false>(packed_half_35));

        matrix_fragment_67 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                           select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_65 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                           select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                            select_half_operand<2, false, false>(packed_half_34));

        matrix_fragment_63 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                           select_half_operand<2, false, false>(matrix_fragment_63));

        packed_half_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_61 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                           select_half_operand<2, false, false>(matrix_fragment_61));

        packed_half_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        packed_half_39 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                           select_half_operand<2, false, false>(matrix_accumulator_40));

        packed_half_35 =
            fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                          select_half_operand<2, false, false>(matrix_fragment_63),
                                          select_half_operand<2, false, false>(packed_half_35));

        matrix_accumulator_42 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                           select_half_operand<2, false, false>(matrix_fragment_69));

        packed_half_34 =
            fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(packed_half_34));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_39), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_40 = result_word;
        }

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                        select_half_operand<2, false, false>(packed_half_41));

        packed_half_41 = add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                                        select_half_operand<2, false, false>(packed_half_34));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_42), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            packed_half_35 = result_word;
        }

        matrix_accumulator_43 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                           select_half_operand<2, false, false>(matrix_fragment_68));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_41), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_68 = result_word;
        }

        matrix_fragment_71 = permute_signed(packed_half_38, packed_half_38, 0x5432u);

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
            matrix_fragment_69 = result_word;
        }

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

        packed_half_38 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0x4100410u);

        packed_half_39 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                           select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_40 = half_to_float_bits((packed_half_38 & 65535u) * 65537u);

        packed_half_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                                        select_half_operand<2, false, false>(packed_half_35));

        matrix_fragment_71 = rsqrt_bits(matrix_accumulator_40);

        packed_half_41 = add_half_pairs(select_half_operand<2, false, false>(packed_half_41),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_72 = permute_signed(packed_half_39, packed_half_39, 0x5432u);

        matrix_accumulator_40 = permute_signed(packed_half_35, packed_half_35, 0x5432u);

        matrix_fragment_69 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                           select_half_operand<2, false, false>(matrix_fragment_69));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_accumulator_43), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            packed_half_34 = result_word;
        }

        packed_half_35 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_35),
                           select_half_operand<2, false, false>(matrix_accumulator_40));

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(packed_half_41), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            matrix_accumulator_40 = result_word;
        }

        packed_half_39 = add_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_70 = half_to_float_bits((packed_half_38 >> 16) * 65537u);

        {
            uint32_t result_word;
            asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
                : "=r"(result_word)
                : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
            packed_half_38 = result_word;
        }

        packed_half_39 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_39), 0x4100410u);

        matrix_accumulator_42 = half_to_float_bits((packed_half_39 & 65535u) * 65537u);

        packed_half_39 = half_to_float_bits((packed_half_39 >> 16) * 65537u);

        packed_half_35 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_35), 0x4100410u);

        matrix_accumulator_42 = rsqrt_bits(matrix_accumulator_42);

        packed_half_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                                        select_half_operand<2, false, false>(packed_half_34));

        packed_half_39 = rsqrt_bits(packed_half_39);

        matrix_accumulator_40 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_41),
                           select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_43 = half_to_float_bits((packed_half_35 & 65535u) * 65537u);

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(packed_half_38));

        packed_half_35 = half_to_float_bits((packed_half_35 >> 16) * 65537u);

        matrix_fragment_73 = permute_signed(packed_half_34, packed_half_34, 0x5432u);

        packed_half_41 = permute_signed(matrix_accumulator_40, matrix_accumulator_40, 0x5432u);

        matrix_fragment_68 = rsqrt_bits(packed_half_35);

        packed_half_34 = add_half_pairs(select_half_operand<2, false, false>(packed_half_34),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_accumulator_40 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                           select_half_operand<2, false, false>(packed_half_41));

        packed_half_35 = permute_signed(packed_half_38, packed_half_38, 0x5432u);

        packed_half_39 = pack_half(packed_half_39, matrix_accumulator_42);

        packed_half_34 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_34), 0x4100410u);

        packed_half_38 = add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                        select_half_operand<2, false, false>(packed_half_35));

        matrix_accumulator_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_40), 0x4100410u);

        matrix_accumulator_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                                select_half_operand<2, false, false>(packed_half_39));

        matrix_accumulator_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                                select_half_operand<2, false, false>(packed_half_39));

        packed_half_38 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0x4100410u);

        matrix_accumulator_49 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(packed_half_39));

        matrix_fragment_69 = half_to_float_bits((packed_half_34 & 65535u) * 65537u);

        matrix_accumulator_51 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                                select_half_operand<2, false, false>(packed_half_39));

        packed_half_39 = half_to_float_bits((matrix_accumulator_40 & 65535u) * 65537u);

        packed_half_34 = half_to_float_bits((packed_half_34 >> 16) * 65537u);

        matrix_accumulator_40 = half_to_float_bits((matrix_accumulator_40 >> 16) * 65537u);

        matrix_fragment_70 = rsqrt_bits(matrix_fragment_70);

        packed_half_35 = half_to_float_bits((packed_half_38 & 65535u) * 65537u);

        packed_half_38 = half_to_float_bits((packed_half_38 >> 16) * 65537u);

        matrix_accumulator_43 = rsqrt_bits(matrix_accumulator_43);

        matrix_fragment_69 = rsqrt_bits(matrix_fragment_69);

        packed_half_34 = rsqrt_bits(packed_half_34);

        packed_half_39 = rsqrt_bits(packed_half_39);

        matrix_accumulator_40 = rsqrt_bits(matrix_accumulator_40);

        packed_half_35 = rsqrt_bits(packed_half_35);

        packed_half_38 = rsqrt_bits(packed_half_38);

        matrix_fragment_70 = pack_half(matrix_fragment_70, matrix_fragment_71);

        matrix_accumulator_43 = pack_half(matrix_fragment_68, matrix_accumulator_43);

        packed_half_34 = pack_half(packed_half_34, matrix_fragment_69);

        packed_half_39 = pack_half(matrix_accumulator_40, packed_half_39);

        matrix_accumulator_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_accumulator_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        packed_half_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                                select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                select_half_operand<2, false, false>(packed_half_34));

        if (!condition_1) {
            matrix_accumulator_42 = permute_signed(matrix_accumulator_42, packed_half_41, 0x5410u);
        }

        matrix_accumulator_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                select_half_operand<2, false, false>(matrix_accumulator_43));

        packed_half_38 = pack_half(packed_half_38, packed_half_35);

        matrix_accumulator_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                select_half_operand<2, false, false>(matrix_accumulator_43));

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(packed_half_39));

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(packed_half_39));

        matrix_accumulator_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        packed_half_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_accumulator_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                                select_half_operand<2, false, false>(matrix_accumulator_43));

        matrix_accumulator_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                                select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                                select_half_operand<2, false, false>(packed_half_34));

        matrix_accumulator_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                select_half_operand<2, false, false>(matrix_accumulator_43));

        matrix_fragment_72 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                select_half_operand<2, false, false>(packed_half_38));

        matrix_accumulator_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(packed_half_38));

        matrix_accumulator_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(packed_half_38));

        matrix_accumulator_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                select_half_operand<2, false, false>(packed_half_38));

        matrix_accumulator_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(packed_half_39));

        packed_half_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(packed_half_39));

        packed_half_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                          select_half_operand<2, false, false>(0u));

        if (!condition_1) {
            matrix_accumulator_44 = permute_signed(matrix_accumulator_44, packed_half_41, 0x5410u);
        }

        packed_half_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                       select_half_operand<2, false, false>(0u));

        packed_half_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_58 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                          select_half_operand<2, false, false>(0u));

        if (!condition_2) {
            packed_half_35 = permute_signed(matrix_accumulator_53, matrix_accumulator_46, 0x5410u);

            matrix_accumulator_40 = permute_signed(matrix_accumulator_40, packed_half_39, 0x5410u);
        }

        if (!condition_3) {
            matrix_accumulator_51 = permute_signed(matrix_accumulator_51, packed_half_38, 0x5410u);
        }

        if (!condition_1) {
            split_address_words(packed_half_38, packed_half_39,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_102, matrix_accumulator_103));
        }

        condition_0 = int32_t(matrix_accumulator_111) != int32_t(0u);

        if (!condition_1) {
            matrix_accumulator_47 =
                permute_signed(matrix_accumulator_49, matrix_fragment_72, 0x5410u);

            matrix_accumulator_45 =
                permute_signed(matrix_accumulator_45, matrix_accumulator_50, 0x5410u);

            matrix_accumulator_46 = matrix_accumulator_42;
        }

        if (!condition_2) {
            packed_half_34 = permute_signed(packed_half_34, matrix_accumulator_59, 0x5410u);

            packed_half_41 = permute_signed(packed_half_41, matrix_accumulator_58, 0x5410u);
        }

        matrix_accumulator_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_64 = 0x4u;

        matrix_accumulator_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        if (!condition_1) {
            store_four_words_streaming<CheckBounds>(
                combine_words(packed_half_38, packed_half_39) + 0x4000ull, matrix_accumulator_44,
                matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47, memory_ranges,
                error_counts);
        }

        if (!condition_2) {
            split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_102, matrix_accumulator_103));
        }

        if (!condition_3) {
            matrix_accumulator_52 =
                permute_signed(matrix_accumulator_52, matrix_accumulator_43, 0x5410u);

            matrix_accumulator_53 =
                permute_signed(matrix_accumulator_53, matrix_fragment_62, 0x5410u);
        }

        split_address_words(matrix_fragment_64, matrix_fragment_65,
                            int64_t(int32_t(address_word_37)) *
                                    int64_t(int32_t(matrix_fragment_64)) +
                                combine_words(uniform_address_word_10, uniform_address_word_11));

        matrix_accumulator_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_42 = 0;
        matrix_accumulator_43 = 0;

        if (!condition_3) {
            split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_102, matrix_accumulator_103));

            matrix_accumulator_50 =
                permute_signed(matrix_accumulator_50, matrix_fragment_63, 0x5410u);
        }

        if (!condition_2) {
            matrix_accumulator_44 = matrix_accumulator_40;

            matrix_accumulator_45 = packed_half_41;

            matrix_accumulator_46 = packed_half_35;

            matrix_accumulator_47 = packed_half_34;
        }

        matrix_accumulator_40 = 0;
        packed_half_41 = 0;

        if (!condition_0) {
            split_address_words(packed_half_34, packed_half_35,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_64, matrix_fragment_65));
        }

        if (!condition_2) {
            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_48, matrix_accumulator_49) + 0x8000ull,
                matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_46,
                matrix_accumulator_47, memory_ranges, error_counts);
        }

        split_address_words(packed_half_38, packed_half_39,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8000u);

        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x8200u);

        matrix_accumulator_56 = 0;
        matrix_accumulator_57 = 0;

        matrix_accumulator_58 = 0;
        matrix_accumulator_59 = 0;

        matrix_accumulator_54 = 0;
        matrix_accumulator_55 = 0;

        if (!condition_3) {
            matrix_accumulator_48 = matrix_accumulator_53;

            matrix_accumulator_49 = matrix_accumulator_52;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0xc000ull,
                matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
                matrix_accumulator_51, memory_ranges, error_counts);
        }

        if (!condition_0) {
            load_four_words_cached<CheckBounds>(combine_words(packed_half_34, packed_half_35),
                                                matrix_accumulator_40, packed_half_41,
                                                matrix_accumulator_42, matrix_accumulator_43,
                                                memory_ranges, error_counts);
        }

        matrix_accumulator_52 = 0;
        matrix_accumulator_53 = 0;

        matrix_accumulator_44 = 0;
        matrix_accumulator_45 = 0;

        matrix_accumulator_46 = 0;
        matrix_accumulator_47 = 0;

        if (!condition_4) {
            split_address_words(packed_half_38, packed_half_39,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(packed_half_38, packed_half_39));
        }

        split_address_words(matrix_fragment_74, matrix_fragment_75,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x18000u);

        split_address_words(matrix_fragment_70, matrix_fragment_71,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x10000u);

        if (!condition_4) {
            split_address_words(matrix_fragment_68, matrix_fragment_69,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_68, matrix_fragment_69));
        }

        split_address_words(packed_half_34, packed_half_35,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x200u);

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            combine_words(matrix_fragment_64, matrix_fragment_65) + 0x10200u);

        if (!condition_4) {
            load_four_words_cached<CheckBounds>(combine_words(packed_half_38, packed_half_39),
                                                matrix_accumulator_56, matrix_accumulator_57,
                                                matrix_accumulator_58, matrix_accumulator_59,
                                                memory_ranges, error_counts);
        }

        if (!condition_0) {
            split_address_words(packed_half_34, packed_half_35,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(packed_half_34, packed_half_35));
        }

        matrix_accumulator_48 = 0;
        matrix_accumulator_49 = 0;

        if (!condition_4) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_68, matrix_fragment_69), matrix_accumulator_52,
                matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
                error_counts);
        }

        matrix_accumulator_50 = 0;
        matrix_accumulator_51 = 0;

        matrix_fragment_60 = 0;
        matrix_fragment_61 = 0;

        matrix_fragment_62 = 0;
        matrix_fragment_63 = 0;

        if (!condition_0) {
            load_four_words_cached<CheckBounds>(combine_words(packed_half_34, packed_half_35),
                                                matrix_accumulator_44, matrix_accumulator_45,
                                                matrix_accumulator_46, matrix_accumulator_47,
                                                memory_ranges, error_counts);
        }

        matrix_fragment_64 = 0;
        matrix_fragment_65 = 0;

        matrix_fragment_66 = 0;
        matrix_fragment_67 = 0;

        address_word_37 = 0u;

        packed_half_38 = 0;
        packed_half_39 = 0;

        if (!condition_5) {
            split_address_words(matrix_fragment_70, matrix_fragment_71,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_70, matrix_fragment_71));
        }

        if (!condition_6) {
            split_address_words(matrix_fragment_68, matrix_fragment_69,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_74, matrix_fragment_75));
        }

        if (!condition_5) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_70, matrix_fragment_71), matrix_accumulator_48,
                matrix_accumulator_49, matrix_accumulator_50, matrix_accumulator_51, memory_ranges,
                error_counts);

            split_address_words(packed_half_34, packed_half_35,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_72, matrix_fragment_73));
        }

        if (!condition_6) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_68, matrix_fragment_69), matrix_fragment_64,
                matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
                error_counts);

            split_address_words(matrix_fragment_74, matrix_fragment_75,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_74, matrix_fragment_75));
        }

        if (!condition_5) {
            load_four_words_cached<CheckBounds>(combine_words(packed_half_34, packed_half_35),
                                                matrix_fragment_60, matrix_fragment_61,
                                                matrix_fragment_62, matrix_fragment_63,
                                                memory_ranges, error_counts);
        }

        if (!condition_6) {
            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_74, matrix_fragment_75) + 0x200ull, address_word_36,
                address_word_37, packed_half_38, packed_half_39, memory_ranges, error_counts);
        }

        uniform_index_word_4 = uniform_index_word_5 * uniform_index_word_4;

        uniform_index_word_4 = uniform_index_word_4 + 0xffffffffu;

        uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1fu);

        uniform_index_word_5 =
            uint32_t(combine_words(uniform_index_word_5, 0u) << 0x5u >> 32) + uniform_index_word_4;

        uniform_index_word_5 = uniform_index_word_5 & 0xffffffe0u;

        matrix_fragment_71 = uniform_parameter_word_7;

        index_word_0 = index_word_0 & 0x3feu;

        uniform_index_word_4 = uniform_index_word_5 + 0x20u;

        matrix_fragment_70 = matrix_fragment_71 * 0x4u + index_word_0;

        uniform_index_word_4 = uint32_t(int32_t(uniform_index_word_4) >> 0x5u);

        condition_0 = int32_t(matrix_fragment_70) >= int32_t(uniform_index_word_4);

        packed_half_35 = matrix_fragment_70 + 0x1u;

        condition_1 = int32_t(packed_half_35) >= int32_t(uniform_index_word_4);

        matrix_accumulator_40 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_40),
                           select_half_operand<2, false, false>(matrix_accumulator_2));

        packed_half_41 = add_half_pairs(select_half_operand<2, false, false>(packed_half_41),
                                        select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_2 = parameters[parameter_output_v_low_6];
        matrix_accumulator_3 = parameters[parameter_output_v_high_7];

        matrix_fragment_4 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                           select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_5 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                           select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_accumulator_13 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                           select_half_operand<2, false, false>(matrix_accumulator_13));

        matrix_fragment_10 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                           select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_11 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                           select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_accumulator_14 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                           select_half_operand<2, false, false>(matrix_accumulator_14));

        matrix_fragment_6 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                           select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_7 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                           select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_8 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                           select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_9 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                           select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_accumulator_15 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                           select_half_operand<2, false, false>(matrix_accumulator_15));

        matrix_fragment_16 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                           select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_17 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                           select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_18 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                           select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_19 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                           select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_accumulator_20 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                           select_half_operand<2, false, false>(matrix_accumulator_20));

        matrix_accumulator_21 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                           select_half_operand<2, false, false>(matrix_accumulator_21));

        matrix_accumulator_22 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                           select_half_operand<2, false, false>(matrix_accumulator_22));

        matrix_accumulator_23 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                           select_half_operand<2, false, false>(matrix_accumulator_23));

        matrix_fragment_24 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                           select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_25 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                           select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_26 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                           select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_27 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                           select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_28 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                           select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_29 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                           select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_30 =
            add_half_pairs(select_half_operand<2, false, false>(address_word_36),
                           select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_31 =
            add_half_pairs(select_half_operand<2, false, false>(address_word_37),
                           select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_accumulator_32 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                           select_half_operand<2, false, false>(matrix_accumulator_32));

        matrix_accumulator_33 =
            add_half_pairs(select_half_operand<2, false, false>(packed_half_39),
                           select_half_operand<2, false, false>(matrix_accumulator_33));

        matrix_accumulator_12 =
            add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                           select_half_operand<2, false, false>(matrix_accumulator_12));

        packed_half_34 = transpose_half_tile(matrix_accumulator_13);

        matrix_accumulator_40 = transpose_half_tile(matrix_accumulator_40);

        matrix_accumulator_13 = matrix_fragment_70 * 0x2000u + matrix_accumulator_101;

        packed_half_41 = transpose_half_tile(packed_half_41);

        matrix_fragment_4 = transpose_half_tile(matrix_fragment_4);

        matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

        matrix_fragment_6 = transpose_half_tile(matrix_fragment_6);

        matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

        matrix_fragment_8 = transpose_half_tile(matrix_fragment_8);

        matrix_fragment_9 = transpose_half_tile(matrix_fragment_9);

        matrix_fragment_10 = transpose_half_tile(matrix_fragment_10);

        matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

        index_word_0 = transpose_half_tile(matrix_accumulator_12);

        matrix_accumulator_14 = transpose_half_tile(matrix_accumulator_14);

        matrix_accumulator_15 = transpose_half_tile(matrix_accumulator_15);

        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_16 = transpose_half_tile(matrix_fragment_16);

        matrix_fragment_17 = transpose_half_tile(matrix_fragment_17);

        matrix_fragment_18 = transpose_half_tile(matrix_fragment_18);

        matrix_fragment_19 = transpose_half_tile(matrix_fragment_19);

        matrix_accumulator_20 = transpose_half_tile(matrix_accumulator_20);

        matrix_accumulator_21 = transpose_half_tile(matrix_accumulator_21);

        matrix_accumulator_22 = transpose_half_tile(matrix_accumulator_22);

        matrix_accumulator_23 = transpose_half_tile(matrix_accumulator_23);

        matrix_fragment_24 = transpose_half_tile(matrix_fragment_24);

        matrix_fragment_25 = transpose_half_tile(matrix_fragment_25);

        matrix_fragment_26 = transpose_half_tile(matrix_fragment_26);

        matrix_fragment_27 = transpose_half_tile(matrix_fragment_27);

        matrix_fragment_28 = transpose_half_tile(matrix_fragment_28);

        matrix_fragment_29 = transpose_half_tile(matrix_fragment_29);

        matrix_fragment_30 = transpose_half_tile(matrix_fragment_30);

        matrix_fragment_31 = transpose_half_tile(matrix_fragment_31);

        matrix_accumulator_32 = transpose_half_tile(matrix_accumulator_32);

        matrix_accumulator_33 = transpose_half_tile(matrix_accumulator_33);
        if (!condition_0) {
            packed_half_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_34),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_3 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                              select_half_operand<2, false, false>(0u));

            index_word_0 =
                pack_fp8_pair(select_half_operand<2, false, false>(index_word_0),
                              select_half_operand<2, false, false>(matrix_accumulator_15));

            packed_half_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_41),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                              select_half_operand<2, false, false>(packed_half_35));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_16 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                              select_half_operand<2, false, false>(matrix_accumulator_3));

            matrix_accumulator_3 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                              select_half_operand<2, false, false>(packed_half_41));

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                              select_half_operand<2, false, false>(matrix_fragment_5));

            matrix_accumulator_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_14),
                              select_half_operand<2, false, false>(matrix_fragment_17));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                              select_half_operand<2, false, false>(matrix_accumulator_15));

            matrix_fragment_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                              select_half_operand<2, false, false>(matrix_accumulator_3));

            split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));

            matrix_fragment_4 = matrix_fragment_7;

            matrix_fragment_6 = matrix_fragment_9;

            matrix_fragment_5 = matrix_fragment_16;

            matrix_fragment_7 = index_word_0;

            matrix_fragment_8 = matrix_accumulator_15;

            matrix_fragment_9 = matrix_accumulator_14;

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_4,
                matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull,
                matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                memory_ranges, error_counts);
        }
        if (!condition_1) {
            matrix_fragment_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                              select_half_operand<2, false, false>(0u));

            split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                                int64_t(int32_t(address_word_100)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_12, matrix_accumulator_13));

            matrix_accumulator_21 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_19 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_23 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_25 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                              select_half_operand<2, false, false>(matrix_fragment_7));

            matrix_fragment_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                              select_half_operand<2, false, false>(matrix_accumulator_21));

            matrix_fragment_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                              select_half_operand<2, false, false>(matrix_fragment_5));

            matrix_fragment_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                              select_half_operand<2, false, false>(matrix_fragment_19));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                              select_half_operand<2, false, false>(matrix_fragment_11));

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                              select_half_operand<2, false, false>(matrix_fragment_9));

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x8000ull,
                matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                memory_ranges, error_counts);

            matrix_fragment_8 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                              select_half_operand<2, false, false>(matrix_accumulator_23));

            matrix_fragment_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                              select_half_operand<2, false, false>(matrix_fragment_25));

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x8200ull,
                matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                memory_ranges, error_counts);
        }
    }
    index_word_0 = threadIdx.x;

    matrix_accumulator_2 = parameters[parameter_publication_low_16];
    matrix_accumulator_3 = parameters[parameter_publication_high_17];

    matrix_fragment_5 = threadIdx.z;

    matrix_fragment_4 = threadIdx.y;

    __syncthreads();

    condition_0 = combine_words(matrix_accumulator_2, matrix_accumulator_3) != 0ull;

    condition_0 = uint32_t(matrix_accumulator_132) != uint32_t(0x1u) || !condition_0;

    condition_1 = (index_word_0 | matrix_fragment_4 | matrix_fragment_5) != 0;

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        combine_words(uniform_address_word_12, uniform_address_word_13));

    if (!condition_1) {
        __threadfence();

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_accumulator_132, memory_ranges, error_counts);
    }

    if (condition_0) {
        return;
    }

    matrix_fragment_4 = parameters[parameter_LH_18];
    matrix_fragment_5 = parameters[parameter_LW_19];

    __threadfence();

    index_word_0 = matrix_fragment_5 * matrix_fragment_4 + 0x7fu;

    matrix_fragment_5 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    matrix_fragment_5 = uint32_t(combine_words(matrix_fragment_5, 0u) << 0x7u >> 32) + index_word_0;

    __syncthreads();

    if (condition_1) {
        return;
    }

    index_word_0 = uint32_t(int32_t(matrix_fragment_5) >> 0x7u);

    matrix_fragment_11 = blockIdx.x;

    matrix_fragment_7 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    matrix_accumulator_12 = int32_t(index_word_0) < 0 ? 0u - index_word_0 : index_word_0;

    matrix_fragment_6 = __float_as_uint(__int2float_ru(matrix_fragment_7));

    matrix_fragment_6 = reciprocal_bits(matrix_fragment_6);

    matrix_fragment_10 =
        int32_t(matrix_fragment_11) < 0 ? 0u - matrix_fragment_11 : matrix_fragment_11;

    matrix_fragment_4 = matrix_fragment_6 + 0xffffffeu;

    matrix_fragment_6 = 0u - matrix_accumulator_12;

    matrix_fragment_5 = __float2uint_rz(__uint_as_float(matrix_fragment_4));

    matrix_fragment_4 = 0u;

    matrix_fragment_8 = 0u - matrix_fragment_5;

    matrix_fragment_9 = matrix_fragment_8 * matrix_fragment_7;

    matrix_fragment_8 = matrix_fragment_10;

    matrix_fragment_5 = uint32_t(uint64_t(matrix_fragment_5) * matrix_fragment_9 +
                                     combine_words(matrix_fragment_4, matrix_fragment_5) >>
                                 32);

    matrix_fragment_5 = uint32_t(uint64_t(matrix_fragment_5) * matrix_fragment_8 >> 32);

    matrix_fragment_4 = matrix_fragment_5 * matrix_fragment_6 + matrix_fragment_8;

    condition_1 = uint32_t(matrix_fragment_7) > uint32_t(matrix_fragment_4);

    if (!condition_1) {
        matrix_fragment_4 = matrix_fragment_4 + (0u - matrix_fragment_7);

        matrix_fragment_5 = matrix_fragment_5 + 0x1u;
    }

    condition_0 = uint32_t(matrix_fragment_4) >= uint32_t(matrix_fragment_7);

    matrix_fragment_4 = index_word_0 ^ matrix_fragment_11;

    condition_1 = int32_t(matrix_fragment_4) >= int32_t(0u);

    if (condition_0) {
        matrix_fragment_5 = matrix_fragment_5 + 0x1u;
    }

    condition_0 = int32_t(index_word_0) != int32_t(0u);

    if (!condition_1) {
        matrix_fragment_5 = 0u - matrix_fragment_5;
    }

    if (!condition_0) {
        matrix_fragment_5 = ~index_word_0;
    }

    matrix_fragment_7 = 0u - matrix_fragment_5;

    index_word_0 = index_word_0 * matrix_fragment_7 + matrix_fragment_11;

    matrix_fragment_5 = index_word_0 * 0x10u + matrix_fragment_5;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__ __maxnreg__(255) void phoenix_global_attention_transformer_query_key_value(
    KernelArguments parameters) {
    run_global_attention_transformer_query_key_value<false>(parameters.words, nullptr, nullptr);
}
