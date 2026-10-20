// DLSS5 Phoenix optimized kernel: global_attention/transformer/feed_forward_contract.cu
// Coarsest 1024-channel global-transformer stage. Attention spans the full latent token set.
// Projection and feed-forward modules retain their explicit publication and synchronization boundaries.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 17424
#include "../../../common/async_memory.cuh"
#include "../../../common/cached_vector_access.cuh"

template <bool CheckBounds>
__device__ __forceinline__ void
atomic_add_four_half_pairs(uint64_t address, uint32_t word_x, uint32_t word_y, uint32_t word_z,
                           uint32_t word_w, const MemoryRange* memory_ranges,
                           uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, true) ||
            !is_word_address_allowed(address + 12, memory_ranges, true) || address & 15) {
            atomicAdd(error_counts + 1, 1u);
            return;
        }
    asm volatile("red.global.add.noftz.v4.f16x2 [%0],{%1,%2,%3,%4};" ::"l"(address), "r"(word_x),
                 "r"(word_y), "r"(word_z), "r"(word_w)
                 : "memory");
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t read_device_flag(uint64_t address,
                                                     const MemoryRange* memory_ranges,
                                                     uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, false)) {
            atomicAdd(error_counts, 1u);
            return 0;
        }
    uint32_t value;
    asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}
template <bool CheckBounds>
__device__ __forceinline__ void write_device_flag(uint64_t address, uint32_t value,
                                                  const MemoryRange* memory_ranges,
                                                  uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, true)) {
            atomicAdd(error_counts + 1, 1u);
            return;
        }
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;" ::"l"(address), "r"(value) : "memory");
}
#include "../../../common/feed_forward_reduction.cuh"

#include "../../../common/token_dispatch.cuh"

// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_input_skip_optional_low_2 = 2,
    parameter_input_skip_optional_high_3 = 3,
    parameter_output_low_4 = 4,
    parameter_output_high_5 = 5,
    parameter_trained_weights_low_6 = 6,
    parameter_trained_weights_high_7 = 7,
    parameter_reduction_counter_optional_low_8 = 8,
    parameter_reduction_counter_optional_high_9 = 9,
    parameter_reduction_scratch_optional_low_10 = 10,
    parameter_reduction_scratch_optional_high_11 = 11,
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
__device__ __forceinline__ void run_global_attention_transformer_feed_forward_contract(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t address_word_4 = 0;
    uint32_t address_word_5 = 0;
    uint32_t address_word_6 = 0;
    uint32_t address_word_7 = 0;
    uint32_t matrix_accumulator_8 = 0;
    uint32_t matrix_accumulator_9 = 0;
    uint32_t matrix_accumulator_10 = 0;
    uint32_t matrix_accumulator_11 = 0;
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
    uint32_t matrix_fragment_96 = 0;
    uint32_t matrix_fragment_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t matrix_fragment_100 = 0;
    uint32_t matrix_fragment_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
    uint32_t matrix_fragment_106 = 0;
    uint32_t matrix_fragment_107 = 0;
    uint32_t matrix_fragment_108 = 0;
    uint32_t matrix_fragment_109 = 0;
    uint32_t matrix_fragment_110 = 0;
    uint32_t matrix_fragment_111 = 0;
    uint32_t matrix_fragment_112 = 0;
    uint32_t matrix_fragment_113 = 0;
    uint32_t matrix_fragment_114 = 0;
    uint32_t matrix_fragment_115 = 0;
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
    uint32_t address_word_132 = 0;
    uint32_t block_index_133 = 0;
    uint32_t index_word_134 = 0;
    uint32_t index_word_135 = 0;
    uint32_t reused_word_136 = 0;
    uint32_t index_word_137 = 0;
    uint32_t address_word_138 = 0;
    uint32_t address_word_139 = 0;
    uint32_t address_word_140 = 0;
    uint32_t address_word_141 = 0;
    uint32_t address_word_142 = 0;
    uint32_t address_word_143 = 0;
    uint32_t address_word_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t index_word_146 = 0;
    uint32_t index_word_147 = 0;
    uint32_t address_word_148 = 0;
    uint32_t index_word_149 = 0;
    uint32_t reused_word_150 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_address_word_16 = 0;
    uint32_t uniform_index_word_79 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    address_word_4 = parameters[parameter_LH_16];
    address_word_5 = parameters[parameter_LW_17];

    matrix_accumulator_10 = blockIdx.x;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    matrix_fragment_86 = threadIdx.y;

    matrix_accumulator_13 = threadIdx.x;

    block_index_133 = blockIdx.z;

    matrix_accumulator_15 = threadIdx.x & 31u;

    address_word_4 = address_word_5 * address_word_4 + 0xffffffffu;

    address_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

    address_word_6 = truncate_signed_power_of_two<7>(address_word_4);

    condition_1 = (matrix_accumulator_13 | matrix_fragment_86) != 0;

    address_word_7 = address_word_6 + 0x1u;

    // Keep work assignment and copy synchronization out of matrix operands.
    const auto work = token_tile_work(matrix_accumulator_10, address_word_7);
    const auto copy = tile_copy_schedule<shared_bytes, 2>(!condition_1, blockDim.x * blockDim.y);
    uniform_condition_0 = copy.without_block_leader;
    uniform_condition_1 = copy.without_block_leader;
    uniform_address_word_4 = copy.wait_word_low;
    uniform_address_word_5 = copy.wait_word_high;
    uniform_address_word_6 = copy.barrier_byte_offset;
    address_word_132 = work.group;
    matrix_accumulator_2 = parameters[parameter_trained_weights_low_6];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_7];
    const uint32_t row_in_pair = (matrix_fragment_86 * 64u) & 64u;
    packed_half_0 = work.group * 128u + row_in_pair;
    reused_word_136 = packed_half_0 * 8u;

    condition_0 = (matrix_fragment_86 & 0x1u) != 0;

    matrix_accumulator_9 = block_index_133 * 0x40000u + reused_word_136;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    condition_0 = uint32_t(matrix_accumulator_13) != 0u || condition_0;

    address_word_4 = uint32_t(combine_words(address_word_5, 0u) << 0x5u >> 32) + address_word_4;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    address_word_4 = address_word_4 & 0xffffffe0u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_15)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __syncthreads();

    matrix_fragment_84 = address_word_4 + 0x20u;

    index_word_135 = work.tile;

    address_word_4 = 0u;

    matrix_fragment_84 = uint32_t(int32_t(matrix_fragment_84) >> 0x4u);

    address_word_132 = index_word_135 * 0x8u + address_word_132;

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                        matrix_fragment_80, matrix_fragment_81, matrix_fragment_82,
                                        matrix_fragment_83, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x400ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x600ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x8000ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x8200ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x8400ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x8600ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_accumulator_3 = index_word_135 * 0x80u;
    if (!condition_0) {
        matrix_accumulator_2 = matrix_fragment_86 * 0x20u + matrix_accumulator_3;

        matrix_accumulator_3 = matrix_accumulator_2 + 0x20u;

        matrix_accumulator_2 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x7u);

        address_word_6 = uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x19u >> 32) + 0x1u;

        address_word_7 = min(int32_t(address_word_7), int32_t(address_word_6));

        condition_1 = int32_t(matrix_accumulator_2) >= int32_t(address_word_7);
        if (!condition_1) {
            address_word_6 = matrix_accumulator_2;
            do {
                matrix_accumulator_2 = parameters[parameter_wait_optional_low_12];
                matrix_accumulator_3 = parameters[parameter_wait_optional_high_13];

                address_word_5 = address_word_6 * 0x4u + block_index_133;

                address_word_5 = address_word_5 * 0x8u;

                split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                                    uint64_t(uint32_t(address_word_5)) * uint64_t(uint32_t(0x4u)) +
                                        combine_words(matrix_accumulator_2, matrix_accumulator_3));

                address_word_5 = read_device_flag<CheckBounds>(
                    combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                    error_counts);

                address_word_6 = address_word_6 + 0x1u;

                condition_2 = uint32_t(address_word_6) != uint32_t(address_word_7);

                condition_1 = int32_t(address_word_5) > -1;
                if (!condition_1) {
                    do {
                        __nanosleep(0x40u);

                        address_word_5 = read_device_flag<CheckBounds>(
                            combine_words(matrix_accumulator_2, matrix_accumulator_3),
                            memory_ranges, error_counts);

                        condition_1 = int32_t(address_word_5) > -1;
                    } while (!condition_1);
                }
            } while (condition_2);
        }
    }
    __syncwarp(__activemask());

    matrix_accumulator_17 = index_word_135 * 0x8u + matrix_fragment_86;

    uniform_index_word_12 = 0u;

    matrix_accumulator_3 = 0x80u;

    address_word_7 = 0x1u;

    matrix_accumulator_2 = matrix_accumulator_17 * 0x4000u;

    condition_2 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_84);

    condition_1 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_84);

    address_word_5 = matrix_fragment_86 * 0x100u;

    address_word_6 = 0x80u;

    matrix_accumulator_16 = block_index_133 * 0x1000u + matrix_accumulator_2;

    matrix_accumulator_2 = permute_signed(0u, matrix_accumulator_2, 0x7610u);

    uniform_address_word_6 = 0u;

    uniform_index_word_13 = 0x400u;

    __syncthreads();

    matrix_accumulator_9 = !condition_1 ? matrix_accumulator_16 : 0u;

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    split_address_words(matrix_accumulator_10, matrix_accumulator_11, 0ull);

    if (!condition_2) {
        matrix_accumulator_8 = matrix_accumulator_16 + 0x80u;

        matrix_accumulator_3 = matrix_accumulator_9 + 0x80u;

        matrix_accumulator_2 = permute_signed(address_word_7, matrix_accumulator_2, 0x7610u);

        address_word_6 = address_word_5 + 0x80u;

        address_word_7 = address_word_5 + 0x80u;

        condition_3 = uint32_t(matrix_accumulator_8) != uint32_t(matrix_accumulator_3);
    }

    uniform_index_word_14 = (uniform_index_word_12 << 0x18u) + uniform_index_word_13;

    if (!condition_2) {
        condition_1 = uint32_t(address_word_7) != uint32_t(address_word_6);

        matrix_accumulator_3 = !condition_3 ? matrix_accumulator_2 : 0u;

        address_word_6 = parameters[parameter_input_low_0];
        address_word_7 = parameters[parameter_input_high_1];

        matrix_accumulator_3 = !condition_1 ? matrix_accumulator_3 : 0u;
    }

    uniform_address_word_4 = uniform_index_word_14;

    uniform_address_word_5 = uniform_address_word_6;

    if (!condition_2) {
        matrix_accumulator_2 = permute_signed(matrix_accumulator_3, matrix_accumulator_2, 0x7610u);
    }

    matrix_accumulator_8 = matrix_accumulator_2 & 0xffffu;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_address_word_4, uniform_address_word_5));

    condition_1 = uint32_t(matrix_accumulator_8) != 0u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(address_word_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    if (!condition_2) {
        split_address_words(matrix_accumulator_10, matrix_accumulator_11,
                            int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_6, address_word_7));

        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }
    if (!condition_1) {
        if (!condition_2) {
            address_word_6 = parameters[parameter_input_low_0];
            address_word_7 = parameters[parameter_input_high_1];
        }

        split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

        matrix_accumulator_10 = (matrix_fragment_86 << 0xau) + uniform_index_word_14;

        if (!condition_2) {
            split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                                int64_t(int32_t(matrix_accumulator_9)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_6, address_word_7));
        }
        if (condition_2) {
            address_word_7 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 = uniform_address_word_4;

            matrix_accumulator_9 = uniform_address_word_4;

            matrix_accumulator_12 =
                permute_signed(matrix_accumulator_12, matrix_accumulator_9, 0x5410u);

            matrix_accumulator_13 = matrix_accumulator_12;

            matrix_accumulator_14 = matrix_accumulator_12;

            matrix_accumulator_15 = matrix_accumulator_12;

            address_word_6 = address_word_7 * 0x10u + matrix_accumulator_10;

            store_four_shared_words<CheckBounds>(shared_tile, address_word_6, matrix_accumulator_12,
                                                 matrix_accumulator_13, matrix_accumulator_14,
                                                 matrix_accumulator_15, shared_bytes, memory_ranges,
                                                 error_counts);
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_8 = broadcast_active_lane(matrix_accumulator_12);

                uniform_address_word_4 = uniform_index_word_13 + 0x4000u;

                uniform_address_word_9 = broadcast_active_lane(matrix_accumulator_13);

                uniform_address_word_6 = broadcast_active_lane(matrix_accumulator_10);

                uniform_address_word_7 = (uniform_index_word_12 << 0x18u) + uniform_address_word_4;

                address_word_6 = 0x200u;

                uniform_address_word_4 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_4 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_4 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            address_word_7 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_8 = uniform_address_word_4;

            matrix_accumulator_9 = uniform_address_word_4;

            matrix_accumulator_8 =
                permute_signed(matrix_accumulator_8, matrix_accumulator_9, 0x5410u);

            matrix_accumulator_9 = matrix_accumulator_8;

            matrix_accumulator_11 = matrix_accumulator_8;

            address_word_7 = address_word_7 * 0x10u + matrix_accumulator_10;

            matrix_accumulator_10 = matrix_accumulator_8;

            store_four_shared_words<CheckBounds>(shared_tile, address_word_7 + 0x200u,
                                                 matrix_accumulator_8, matrix_accumulator_9,
                                                 matrix_accumulator_10, matrix_accumulator_11,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                matrix_accumulator_12 = parameters[parameter_input_low_0];
                matrix_accumulator_13 = parameters[parameter_input_high_1];

                matrix_accumulator_9 = matrix_accumulator_16 + 0x80u;
            }

            split_address_words(address_word_6, address_word_7, 0ull);

            uniform_address_word_4 = 0xffffffffu;

            if (!condition_2) {
                address_word_6 = matrix_accumulator_9;

                address_word_7 = uint32_t(int32_t(matrix_accumulator_9) >> 0x1fu);
            }

            split_address_words(matrix_accumulator_8, matrix_accumulator_9, 0ull);

            if (!condition_2) {
                const uint64_t byte_address_0 =
                    scaled_word_address(combine_words(matrix_accumulator_12, matrix_accumulator_13),
                                        combine_words(address_word_6, address_word_7));
                matrix_accumulator_11 = uint32_t(byte_address_0);

                matrix_accumulator_9 = uint32_t(byte_address_0 >> 32);

                matrix_accumulator_8 = matrix_accumulator_11;
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = broadcast_active_lane(matrix_accumulator_10);

                uniform_address_word_5 = uniform_index_word_13 + 0x4000u;

                uniform_address_word_6 = broadcast_active_lane(matrix_accumulator_8);

                uniform_address_word_8 = 0x20u;

                uniform_address_word_7 = broadcast_active_lane(matrix_accumulator_9);

                uniform_address_word_5 = (uniform_index_word_12 << 0x18u) + uniform_address_word_5;

                address_word_6 = 0x200u;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_4 = broadcast_active_lane(matrix_accumulator_12);

            uniform_address_word_7 = uniform_index_word_13 + 0x4000u;

            uniform_address_word_5 = broadcast_active_lane(matrix_accumulator_13);

            uniform_address_word_15 = 0x40u;

            uniform_address_word_8 = broadcast_active_lane(matrix_accumulator_10);

            uniform_address_word_7 = (uniform_index_word_12 << 0x18u) + uniform_address_word_7;

            uniform_address_word_9 = broadcast_active_lane(matrix_accumulator_11);

            address_word_6 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_15 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_15 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_accumulator_17 = matrix_accumulator_17 + 0x4u;

    address_word_6 = matrix_accumulator_16 + 0x10000u;

    matrix_accumulator_9 = 0x1u;

    address_word_7 = 0x80u;

    matrix_accumulator_8 = permute_signed(0u, matrix_accumulator_8, 0x7610u);

    condition_2 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_84);

    condition_1 = int32_t(matrix_accumulator_17) >= int32_t(matrix_fragment_84);

    split_address_words(matrix_accumulator_12, matrix_accumulator_13, 0ull);

    matrix_accumulator_11 = !condition_1 ? address_word_6 : 0u;

    address_word_6 = 0x80u;

    if (!condition_2) {
        matrix_accumulator_10 = matrix_accumulator_16 + 0x10080u;

        address_word_7 = matrix_accumulator_11 + 0x80u;

        matrix_accumulator_8 = permute_signed(matrix_accumulator_9, matrix_accumulator_8, 0x7610u);

        address_word_6 = address_word_5 + 0x480u;

        matrix_accumulator_9 = address_word_5 + 0x480u;

        condition_3 = uint32_t(matrix_accumulator_10) != uint32_t(address_word_7);
    }

    address_word_5 = address_word_5 + 0x400u;

    if (!condition_2) {
        condition_1 = uint32_t(matrix_accumulator_9) != uint32_t(address_word_6);

        matrix_accumulator_9 = !condition_3 ? matrix_accumulator_8 : 0u;

        address_word_6 = parameters[parameter_input_low_0];
        address_word_7 = parameters[parameter_input_high_1];

        matrix_accumulator_9 = !condition_1 ? matrix_accumulator_9 : 0u;

        matrix_accumulator_8 = permute_signed(matrix_accumulator_9, matrix_accumulator_8, 0x7610u);
    }

    matrix_accumulator_8 = matrix_accumulator_8 & 0xffffu;

    condition_1 = uint32_t(matrix_accumulator_8) != 0u;

    split_address_words(matrix_accumulator_8, matrix_accumulator_9, 0ull);

    if (!condition_2) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x1000u);

        split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                            int64_t(int32_t(matrix_accumulator_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_6, address_word_7));
    }
    if (!condition_1) {
        if (condition_2) {
            matrix_accumulator_2 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            matrix_accumulator_3 = (address_word_5 << 0x2u) + uniform_index_word_14;

            matrix_accumulator_8 = uniform_address_word_4;

            address_word_7 = uniform_address_word_4;

            matrix_accumulator_8 = permute_signed(matrix_accumulator_8, address_word_7, 0x5410u);

            matrix_accumulator_9 = matrix_accumulator_8;

            matrix_accumulator_10 = matrix_accumulator_8;

            matrix_accumulator_11 = matrix_accumulator_8;

            matrix_accumulator_2 = matrix_accumulator_2 * 0x10u + matrix_accumulator_3;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_2,
                                                 matrix_accumulator_8, matrix_accumulator_9,
                                                 matrix_accumulator_10, matrix_accumulator_11,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                matrix_accumulator_2 = parameters[parameter_input_low_0];
                matrix_accumulator_3 = parameters[parameter_input_high_1];
            }

            uniform_address_word_4 = 0xffffffffu;

            split_address_words(address_word_6, address_word_7, 0ull);

            if (!condition_2) {
                split_address_words(address_word_6, address_word_7,
                                    int64_t(int32_t(matrix_accumulator_11)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_accumulator_2, matrix_accumulator_3));
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = broadcast_active_lane(address_word_5);

                uniform_address_word_5 = uniform_index_word_13 + 0x4000u;

                uniform_address_word_6 = broadcast_active_lane(address_word_6);

                uniform_address_word_8 = 0x20u;

                uniform_address_word_7 = broadcast_active_lane(address_word_7);

                uniform_address_word_5 = (uniform_index_word_12 << 0x18u) + uniform_address_word_5;

                matrix_accumulator_2 = 0x200u;

                uniform_address_word_4 = (uniform_address_word_4 << 0x2u) + uniform_index_word_14;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        if (condition_2) {
            matrix_accumulator_2 = threadIdx.x & 31u;

            uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));

            address_word_5 = (address_word_5 << 0x2u) + uniform_index_word_14;

            matrix_accumulator_8 = uniform_address_word_4;

            matrix_accumulator_3 = uniform_address_word_4;

            matrix_accumulator_8 =
                permute_signed(matrix_accumulator_8, matrix_accumulator_3, 0x5410u);

            matrix_accumulator_9 = matrix_accumulator_8;

            matrix_accumulator_10 = matrix_accumulator_8;

            matrix_accumulator_11 = matrix_accumulator_8;

            matrix_accumulator_2 = matrix_accumulator_2 * 0x10u + address_word_5;

            store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_2 + 0x200u,
                                                 matrix_accumulator_8, matrix_accumulator_9,
                                                 matrix_accumulator_10, matrix_accumulator_11,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            if (!condition_2) {
                matrix_accumulator_10 = parameters[parameter_input_low_0];
                matrix_accumulator_11 = parameters[parameter_input_high_1];

                address_word_7 = matrix_accumulator_16 + 0x10080u;
            }

            split_address_words(matrix_accumulator_2, matrix_accumulator_3, 0ull);

            uniform_address_word_4 = 0xffffffffu;

            if (!condition_2) {
                matrix_accumulator_2 = address_word_7;

                matrix_accumulator_3 = uint32_t(int32_t(address_word_7) >> 0x1fu);
            }

            split_address_words(address_word_6, address_word_7, 0ull);

            if (!condition_2) {
                const uint64_t byte_address_1 =
                    scaled_word_address(combine_words(matrix_accumulator_10, matrix_accumulator_11),
                                        combine_words(matrix_accumulator_2, matrix_accumulator_3));
                matrix_accumulator_9 = uint32_t(byte_address_1);

                address_word_7 = uint32_t(byte_address_1 >> 32);

                address_word_6 = matrix_accumulator_9;
            }

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = broadcast_active_lane(address_word_5);

                uniform_address_word_5 = uniform_index_word_13 + 0x4000u;

                uniform_address_word_6 = broadcast_active_lane(address_word_6);

                uniform_address_word_8 = 0x20u;

                uniform_address_word_7 = broadcast_active_lane(address_word_7);

                uniform_address_word_5 = (uniform_index_word_12 << 0x18u) + uniform_address_word_5;

                matrix_accumulator_2 = 0x200u;

                uniform_address_word_4 = (uniform_address_word_4 << 0x2u) + uniform_index_word_14;

                uniform_address_word_4 = uniform_address_word_4 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, uniform_address_word_5, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_4,
                    combine_words(uniform_address_word_6, uniform_address_word_7),
                    uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
    } else {
        uniform_address_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_address_word_4 = broadcast_active_lane(matrix_accumulator_8);

            uniform_address_word_7 = uniform_index_word_13 + 0x4000u;

            uniform_address_word_5 = broadcast_active_lane(matrix_accumulator_9);

            uniform_address_word_15 = 0x40u;

            uniform_address_word_8 = broadcast_active_lane(matrix_accumulator_12);

            uniform_address_word_7 = (uniform_index_word_12 << 0x18u) + uniform_address_word_7;

            uniform_address_word_9 = broadcast_active_lane(matrix_accumulator_13);

            matrix_accumulator_2 = 0x400u;

            uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_15 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_15 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_5 = 0u;

    uniform_address_word_4 = uniform_index_word_13 + 0x4000u;

    address_word_138 = parameters[parameter_LH_16];
    address_word_139 = parameters[parameter_LW_17];

    condition_2 = uint32_t(block_index_133) != 0u;

    matrix_accumulator_2 = uniform_address_word_4;

    address_word_138 = address_word_139 * address_word_138 + 0xffffffffu;

    address_word_5 = (address_word_5 << 0x18u) + matrix_accumulator_2;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = address_word_5;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_accumulator_2 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 =
        wait_copy_barrier(shared_tile, address_word_5, async_tokens[(((address_word_5) >> 3) & 1)]);
#else
    __syncthreads();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, address_word_5,
                                            async_tokens[(((address_word_5) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    address_word_5 = threadIdx.y;

    matrix_accumulator_14 = 0;
    matrix_accumulator_15 = 0;

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    matrix_accumulator_18 = 0;
    matrix_accumulator_19 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

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

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_48 = 0;
    matrix_accumulator_49 = 0;

    matrix_accumulator_50 = 0;
    matrix_accumulator_51 = 0;

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    address_word_6 = 0;
    address_word_7 = 0;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_accumulator_10 = 0;
    matrix_accumulator_11 = 0;

    matrix_accumulator_12 = 0;
    matrix_accumulator_13 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;
    if (!condition_2) {
        matrix_fragment_86 = matrix_fragment_86 * 0x2u;

        matrix_accumulator_47 = threadIdx.x & 31u;

        matrix_accumulator_2 = parameters[parameter_input_skip_optional_low_2];
        matrix_accumulator_3 = parameters[parameter_input_skip_optional_high_3];

        matrix_fragment_86 = matrix_fragment_86 & 0x7fcu;

        matrix_accumulator_13 = index_word_135 * 0x8u + matrix_fragment_86;

        matrix_accumulator_42 = parameters[parameter_trained_weights_low_6];
        matrix_accumulator_43 = parameters[parameter_trained_weights_high_7];

        address_word_7 = matrix_accumulator_13 * 0x400u + packed_half_0;

        condition_1 = int32_t(matrix_accumulator_13) >= int32_t(matrix_fragment_84);

        matrix_accumulator_9 = matrix_accumulator_13 + 0x1u;

        address_word_7 = address_word_7 * 0x4u;

        condition_2 = int32_t(matrix_accumulator_9) >= int32_t(matrix_fragment_84);

        matrix_accumulator_9 = matrix_accumulator_13 + 0x2u;

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        condition_3 = int32_t(matrix_accumulator_9) >= int32_t(matrix_fragment_84);

        if (condition_1) {
            matrix_accumulator_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));

            matrix_accumulator_39 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));

            matrix_accumulator_35 =
                permute_signed(matrix_accumulator_35, matrix_accumulator_35, 0x5410u);
        }

        if (!condition_1) {
            split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        if (condition_1) {
            matrix_accumulator_39 =
                permute_signed(matrix_accumulator_39, matrix_accumulator_39, 0x5410u);
        }

        matrix_accumulator_19 = matrix_accumulator_13 + 0x3u;

        if (condition_1) {
            matrix_accumulator_32 = matrix_accumulator_35;

            matrix_accumulator_33 = matrix_accumulator_35;

            matrix_accumulator_34 = matrix_accumulator_35;

            matrix_accumulator_36 = matrix_accumulator_39;

            matrix_accumulator_37 = matrix_accumulator_39;

            matrix_accumulator_38 = matrix_accumulator_39;
        }

        if (condition_3) {
            matrix_accumulator_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (!condition_1) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        if (condition_3) {
            matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x200ull,
                matrix_accumulator_32, matrix_accumulator_33, matrix_accumulator_34,
                matrix_accumulator_35, memory_ranges, error_counts);
        }

        if (condition_2) {
            matrix_accumulator_31 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (condition_3) {
            matrix_accumulator_15 =
                permute_signed(matrix_accumulator_15, matrix_accumulator_15, 0x5410u);
        }

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7), matrix_accumulator_36,
                matrix_accumulator_37, matrix_accumulator_38, matrix_accumulator_39, memory_ranges,
                error_counts);
        }

        if (condition_3) {
            matrix_accumulator_23 =
                permute_signed(matrix_accumulator_23, matrix_accumulator_23, 0x5410u);
        }

        if (!condition_2) {
            split_address_words(matrix_accumulator_10, matrix_accumulator_11,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        matrix_accumulator_8 = matrix_accumulator_47;

        condition_1 = int32_t(matrix_accumulator_19) >= int32_t(matrix_fragment_84);

        if (condition_2) {
            matrix_accumulator_31 =
                permute_signed(matrix_accumulator_31, matrix_accumulator_31, 0x5410u);
        }

        matrix_accumulator_8 = matrix_accumulator_8 & 0x7ffffffcu;

        if (!condition_3) {
            split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        if (condition_3) {
            matrix_accumulator_12 = matrix_accumulator_15;

            matrix_accumulator_13 = matrix_accumulator_15;

            matrix_accumulator_14 = matrix_accumulator_15;

            matrix_accumulator_20 = matrix_accumulator_23;

            matrix_accumulator_21 = matrix_accumulator_23;

            matrix_accumulator_22 = matrix_accumulator_23;
        }

        if (!condition_3) {
            split_address_words(matrix_accumulator_18, matrix_accumulator_19,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        matrix_accumulator_51 = 0u - matrix_accumulator_8 + matrix_accumulator_47;

        if (condition_2) {
            matrix_accumulator_28 = matrix_accumulator_31;

            matrix_accumulator_29 = matrix_accumulator_31;

            matrix_accumulator_30 = matrix_accumulator_31;

            matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (!condition_3) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x8000ull,
                matrix_accumulator_12, matrix_accumulator_13, matrix_accumulator_14,
                matrix_accumulator_15, memory_ranges, error_counts);

            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_accumulator_18, matrix_accumulator_19) + 0x8200ull,
                matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_22,
                matrix_accumulator_23, memory_ranges, error_counts);
        }

        if (condition_2) {
            matrix_accumulator_27 =
                permute_signed(matrix_accumulator_27, matrix_accumulator_27, 0x5410u);
        }

        if (!condition_2) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_accumulator_10, matrix_accumulator_11) + 0x4000ull,
                matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_30,
                matrix_accumulator_31, memory_ranges, error_counts);
        }

        matrix_accumulator_17 = matrix_accumulator_51 * 0x2u + packed_half_0;

        matrix_accumulator_18 = packed_half_0 + 0x10u;

        matrix_accumulator_16 = packed_half_0 + 0x8u;

        matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_17) >> 0x1u);

        if (condition_1) {
            matrix_accumulator_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        matrix_accumulator_18 = matrix_accumulator_51 * 0x2u + matrix_accumulator_18;

        matrix_accumulator_44 = packed_half_0 + 0x18u;

        if (condition_2) {
            matrix_accumulator_24 = matrix_accumulator_27;

            matrix_accumulator_25 = matrix_accumulator_27;

            matrix_accumulator_26 = matrix_accumulator_27;
        }

        if (!condition_2) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        if (condition_1) {
            matrix_accumulator_11 =
                permute_signed(matrix_accumulator_11, matrix_accumulator_11, 0x5410u);
        }

        matrix_accumulator_19 = uint32_t(int32_t(matrix_accumulator_18) >> 0x1u);

        matrix_accumulator_16 = matrix_accumulator_51 * 0x2u + matrix_accumulator_16;

        if (!condition_2) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x4200ull, matrix_accumulator_24,
                matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
                error_counts);
        }

        split_address_words(matrix_accumulator_40, matrix_accumulator_41,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        if (condition_1) {
            matrix_accumulator_8 = matrix_accumulator_11;

            matrix_accumulator_9 = matrix_accumulator_11;
        }

        matrix_accumulator_44 = matrix_accumulator_51 * 0x2u + matrix_accumulator_44;

        if (condition_1) {
            matrix_accumulator_10 = matrix_accumulator_11;
        }

        matrix_accumulator_17 = uint32_t(int32_t(matrix_accumulator_16) >> 0x1u);

        split_address_words(matrix_accumulator_18, matrix_accumulator_19,
                            int64_t(int32_t(matrix_accumulator_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        matrix_accumulator_40 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_40, matrix_accumulator_41) + 0x400000ull,
            memory_ranges, error_counts);

        matrix_accumulator_44 = uint32_t(combine_words(matrix_accumulator_44, 0u) << 0x1u >> 32) +
                                matrix_accumulator_44;

        if (!condition_1) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        matrix_accumulator_48 = packed_half_0 + 0x20u;

        matrix_accumulator_45 = uint32_t(int32_t(matrix_accumulator_44) >> 0x1u);

        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0xc000ull, matrix_accumulator_8,
                matrix_accumulator_9, matrix_accumulator_10, matrix_accumulator_11, memory_ranges,
                error_counts);
        }

        if (condition_1) {
            matrix_accumulator_41 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (!condition_1) {
            split_address_words(matrix_accumulator_46, matrix_accumulator_47,
                                int64_t(int32_t(matrix_accumulator_47)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_2, matrix_accumulator_3));
        }

        matrix_accumulator_2 = packed_half_0 + 0x28u;

        matrix_accumulator_50 = packed_half_0 + 0x38u;

        matrix_accumulator_48 = matrix_accumulator_51 * 0x2u + matrix_accumulator_48;

        address_word_6 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_18, matrix_accumulator_19) + 0x400000ull,
            memory_ranges, error_counts);

        split_address_words(matrix_accumulator_44, matrix_accumulator_45,
                            int64_t(int32_t(matrix_accumulator_45)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        address_word_7 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x400000ull,
            memory_ranges, error_counts);

        matrix_accumulator_2 = matrix_accumulator_51 * 0x2u + matrix_accumulator_2;

        matrix_accumulator_50 = matrix_accumulator_51 * 0x2u + matrix_accumulator_50;

        matrix_accumulator_3 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_44, matrix_accumulator_45) + 0x400000ull,
            memory_ranges, error_counts);

        if (condition_1) {
            matrix_accumulator_19 =
                permute_signed(matrix_accumulator_41, matrix_accumulator_41, 0x5410u);
        }

        matrix_accumulator_41 = uint32_t(int32_t(matrix_accumulator_48) >> 0x1u);

        matrix_accumulator_48 = packed_half_0 + 0x30u;

        if (condition_1) {
            matrix_accumulator_16 = matrix_accumulator_19;

            matrix_accumulator_17 = matrix_accumulator_19;

            matrix_accumulator_18 = matrix_accumulator_19;
        }

        split_address_words(matrix_accumulator_44, matrix_accumulator_45,
                            int64_t(int32_t(matrix_accumulator_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        matrix_accumulator_2 =
            uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x1u >> 32) + matrix_accumulator_2;

        matrix_accumulator_41 = matrix_accumulator_51 * 0x2u + matrix_accumulator_48;

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_accumulator_46, matrix_accumulator_47) + 0xc200ull,
                matrix_accumulator_16, matrix_accumulator_17, matrix_accumulator_18,
                matrix_accumulator_19, memory_ranges, error_counts);
        }

        matrix_accumulator_49 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1u);

        matrix_accumulator_41 = uint32_t(int32_t(matrix_accumulator_41) >> 0x1u);

        matrix_accumulator_2 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_44, matrix_accumulator_45) + 0x400000ull,
            memory_ranges, error_counts);

        matrix_accumulator_46 = uint32_t(combine_words(matrix_accumulator_50, 0u) << 0x1u >> 32) +
                                matrix_accumulator_50;

        matrix_fragment_85 = uint32_t(int32_t(matrix_accumulator_46) >> 0x1u);

        split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                            int64_t(int32_t(matrix_accumulator_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        split_address_words(matrix_accumulator_50, matrix_accumulator_51,
                            int64_t(int32_t(matrix_accumulator_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        matrix_accumulator_47 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_48, matrix_accumulator_49) + 0x400000ull,
            memory_ranges, error_counts);

        split_address_words(matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(matrix_fragment_85)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_42, matrix_accumulator_43));

        matrix_accumulator_46 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_50, matrix_accumulator_51) + 0x400000ull,
            memory_ranges, error_counts);

        matrix_fragment_84 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x400000ull, memory_ranges,
            error_counts);

        matrix_accumulator_128 = unpack_fp8_bits(matrix_accumulator_36);

        matrix_accumulator_129 = unpack_fp8_bits(matrix_accumulator_37);

        matrix_accumulator_42 = unpack_fp8_bits(matrix_accumulator_37 >> 16);

        matrix_fragment_87 = unpack_fp8_bits(matrix_accumulator_38);

        matrix_fragment_88 = unpack_fp8_bits(matrix_accumulator_39);

        matrix_accumulator_38 = unpack_fp8_bits(matrix_accumulator_38 >> 16);

        matrix_accumulator_39 = unpack_fp8_bits(matrix_accumulator_39 >> 16);

        matrix_accumulator_41 = unpack_fp8_bits(matrix_accumulator_36 >> 16);

        matrix_accumulator_130 = unpack_fp8_bits(matrix_accumulator_34);

        matrix_accumulator_126 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

        matrix_accumulator_44 = unpack_fp8_bits(matrix_accumulator_12);

        matrix_accumulator_45 = unpack_fp8_bits(matrix_accumulator_13);

        matrix_fragment_89 = unpack_fp8_bits(matrix_accumulator_14 >> 16);

        matrix_accumulator_124 = unpack_fp8_bits(matrix_accumulator_28);

        matrix_accumulator_122 = unpack_fp8_bits(matrix_accumulator_28 >> 16);

        matrix_accumulator_125 = unpack_fp8_bits(matrix_accumulator_29);

        matrix_accumulator_123 = unpack_fp8_bits(matrix_accumulator_29 >> 16);

        matrix_accumulator_120 = unpack_fp8_bits(matrix_accumulator_30);

        matrix_accumulator_118 = unpack_fp8_bits(matrix_accumulator_30 >> 16);

        matrix_accumulator_121 = unpack_fp8_bits(matrix_accumulator_31);

        matrix_accumulator_119 = unpack_fp8_bits(matrix_accumulator_31 >> 16);

        matrix_fragment_90 = unpack_fp8_bits(matrix_accumulator_15);

        matrix_fragment_91 = unpack_fp8_bits(matrix_accumulator_15 >> 16);

        matrix_accumulator_116 = unpack_fp8_bits(matrix_accumulator_24);

        matrix_accumulator_50 = unpack_fp8_bits(matrix_accumulator_24 >> 16);

        matrix_accumulator_117 = unpack_fp8_bits(matrix_accumulator_25);

        matrix_accumulator_51 = unpack_fp8_bits(matrix_accumulator_25 >> 16);

        matrix_accumulator_48 = unpack_fp8_bits(matrix_accumulator_26);

        matrix_accumulator_31 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

        matrix_accumulator_49 = unpack_fp8_bits(matrix_accumulator_27);

        matrix_accumulator_30 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

        matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_12 >> 16);

        matrix_accumulator_28 = unpack_fp8_bits(matrix_accumulator_8);

        matrix_accumulator_29 = unpack_fp8_bits(matrix_accumulator_9);

        matrix_accumulator_25 = unpack_fp8_bits(matrix_accumulator_14);

        matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

        matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

        matrix_accumulator_8 = unpack_fp8_bits(matrix_accumulator_10);

        matrix_accumulator_9 = unpack_fp8_bits(matrix_accumulator_11);

        matrix_fragment_86 = unpack_fp8_bits(matrix_accumulator_22);

        matrix_accumulator_15 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

        matrix_fragment_85 = unpack_fp8_bits(matrix_accumulator_23);

        matrix_accumulator_14 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

        matrix_accumulator_128 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_43 = unpack_fp8_bits(matrix_accumulator_13 >> 16);

        matrix_accumulator_129 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_10 >> 16);

        matrix_accumulator_124 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_11 >> 16);

        matrix_accumulator_125 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                select_half_operand<2, false, false>(matrix_accumulator_40));

        matrix_accumulator_13 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_42),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_37 = unpack_fp8_bits(matrix_accumulator_33);

        matrix_accumulator_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_8),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_131 = unpack_fp8_bits(matrix_accumulator_35);

        matrix_accumulator_25 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_9),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_127 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

        matrix_accumulator_8 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_fragment_92 = unpack_fp8_bits(matrix_accumulator_20);

        matrix_accumulator_9 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_34 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

        matrix_accumulator_118 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_36 = unpack_fp8_bits(matrix_accumulator_32);

        matrix_accumulator_119 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_20 = unpack_fp8_bits(matrix_accumulator_21);

        matrix_accumulator_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_35 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

        matrix_accumulator_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_21 = unpack_fp8_bits(matrix_accumulator_17);

        matrix_accumulator_22 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_23 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                                select_half_operand<2, false, false>(matrix_accumulator_3));

        matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_16);

        matrix_accumulator_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

        matrix_accumulator_12 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_41),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

        matrix_accumulator_123 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

        matrix_accumulator_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_43),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_fragment_87 = unpack_fp8_bits(matrix_accumulator_18);

        matrix_accumulator_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_fragment_89 = unpack_fp8_bits(matrix_accumulator_19);

        matrix_accumulator_27 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_11 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(address_word_6));

        matrix_accumulator_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(address_word_6));

        address_word_7 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_fragment_88 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

        address_word_6 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_fragment_90 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

        matrix_accumulator_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_117 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_20 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_21 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                                select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_accumulator_2 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_3 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_18 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_19 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_51 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                                select_half_operand<2, false, false>(matrix_accumulator_47));

        matrix_accumulator_130 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_131 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_49 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_accumulator_46));

        matrix_accumulator_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_15),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_14),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_127 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_accumulator_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_84));
    }
    uniform_address_word_5 = 0u;

    matrix_fragment_84 = index_word_135 * 0x80u;

    index_word_137 = uint32_t(int32_t(address_word_138) >> 0x1fu);

    uniform_address_word_4 = 0x400u;

    index_word_134 = address_word_5 * 0x2u;

    address_word_144 = 0u;

    address_word_143 = address_word_5 * 0x20u + matrix_fragment_84;

    matrix_fragment_85 =
        uint32_t(combine_words(index_word_137, 0u) << 0x7u >> 32) + address_word_138;

    index_word_147 = address_word_5 * 0x800u;

    index_word_134 = index_word_134 & 0x7fcu;

    index_word_146 = address_word_5 * 0x200u;

    matrix_fragment_84 = address_word_143 + 0x20u;

    matrix_fragment_85 = uint32_t(int32_t(matrix_fragment_85) >> 0x7u);

    address_word_139 = address_word_5 * 0x100u;

    address_word_143 = uint32_t(uint32_t(address_word_143) >> 0x7u);

    matrix_fragment_84 = uint32_t(combine_words(matrix_fragment_84, 0u) << 0x19u >> 32) + 0x1u;

    matrix_fragment_85 = matrix_fragment_85 + 0x1u;

    index_word_147 = index_word_147 & 0x1ff000u;

    index_word_146 = index_word_146 & 0x7fc00u;

    address_word_142 = min(int32_t(matrix_fragment_85), int32_t(matrix_fragment_84));

    uniform_address_word_4 = (uniform_address_word_5 << 0x18u) + uniform_address_word_4;

    address_word_141 = address_word_139 + 0x80u;

    condition_0 = int32_t(address_word_143) >= int32_t(address_word_142) || condition_0;

    address_word_140 = (address_word_5 << 0xau) + uniform_address_word_4;

    address_word_145 = address_word_140 + 0x1000u;

#pragma unroll 2
    do {
        uniform_address_word_5 = 0u;

        index_word_149 = threadIdx.x & 31u;

        matrix_fragment_84 = address_word_144 * 0x80u;

        uniform_address_word_4 = 0x400u;

        address_word_148 = uint32_t(uint32_t(address_word_144) >> 0x6u);

        reused_word_150 = address_word_144 + 0x40u;

        matrix_fragment_84 = matrix_fragment_84 & 0x2000u;

        address_word_148 = ~address_word_148 & 0x1u;

        uniform_address_word_4 = (uniform_address_word_5 << 0x18u) + uniform_address_word_4;

        matrix_fragment_84 = matrix_fragment_84 + uniform_address_word_4;

        __syncwarp(__activemask());

        matrix_fragment_85 = index_word_147 + matrix_fragment_84;

        matrix_fragment_84 = index_word_146 * 0x4u + matrix_fragment_84;

        matrix_fragment_85 = index_word_149 * 0x10u + matrix_fragment_85;

        index_word_149 = index_word_149 * 0x10u + matrix_fragment_84;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_85, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0x200u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0x400u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0x600u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0x800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0xa00u, matrix_fragment_104, matrix_fragment_105,
            matrix_fragment_106, matrix_fragment_107, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0xc00u, matrix_fragment_108, matrix_fragment_109,
            matrix_fragment_110, matrix_fragment_111, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, index_word_149 + 0xe00u, matrix_fragment_112, matrix_fragment_113,
            matrix_fragment_114, matrix_fragment_115, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_128,
                                 matrix_accumulator_129);

        condition_1 = (address_word_144 & 0x40u) != 0;

        condition_2 = uint32_t(address_word_144) >= uint32_t(0x380u);

        address_word_144 = reused_word_150;

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(address_word_6, address_word_7, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, address_word_6,
                                 address_word_7);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_100,
                                 matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_108,
                                 matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_14,
                                 matrix_accumulator_15);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_12,
                                 matrix_accumulator_13);

        multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_10,
                                 matrix_accumulator_11);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(address_word_6, address_word_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_56, matrix_fragment_57, address_word_6,
                                 address_word_7);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_38,
                                 matrix_accumulator_39);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_36,
                                 matrix_accumulator_37);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_34,
                                 matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_32,
                                 matrix_accumulator_33);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_104,
                                 matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_112,
                                 matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_14,
                                 matrix_accumulator_15);
        if (condition_1) {
            if (!condition_0) {
                matrix_fragment_52 = block_index_133 * 0x400u + reused_word_150;

                matrix_fragment_55 = address_word_143;

                matrix_fragment_54 = uint32_t(uint32_t(matrix_fragment_52) >> 0x7u);
                do {
                    matrix_fragment_52 = parameters[parameter_wait_optional_low_12];
                    matrix_fragment_53 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_57 = matrix_fragment_55 * 0x20u + matrix_fragment_54;

                    split_address_words(matrix_fragment_52, matrix_fragment_53,
                                        uint64_t(uint32_t(matrix_fragment_57)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_52, matrix_fragment_53));

                    matrix_fragment_56 = read_device_flag<CheckBounds>(
                        combine_words(matrix_fragment_52, matrix_fragment_53), memory_ranges,
                        error_counts);

                    matrix_fragment_55 = matrix_fragment_55 + 0x1u;

                    condition_3 = uint32_t(matrix_fragment_55) != uint32_t(address_word_142);

                    condition_1 = int32_t(matrix_fragment_56) > -1;
                    if (!condition_1) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_56 = read_device_flag<CheckBounds>(
                                combine_words(matrix_fragment_52, matrix_fragment_53),
                                memory_ranges, error_counts);

                            condition_1 = int32_t(matrix_fragment_56) > -1;
                        } while (!condition_1);
                    }
                } while (condition_3);
            }
            __syncwarp(__activemask());

            __syncthreads();
        }
        matrix_fragment_52 =
            uint32_t(combine_words(index_word_137, 0u) << 0x5u >> 32) + address_word_138;

        matrix_fragment_64 = index_word_135 * 0x8u + address_word_5;

        uniform_address_word_6 = 0u;

        matrix_fragment_61 = block_index_133 * 0x400u + reused_word_150;

        matrix_fragment_52 = matrix_fragment_52 & 0xffffffe0u;

        matrix_fragment_60 = matrix_fragment_64 * 0x4000u;

        matrix_fragment_57 = 0x1u;

        matrix_fragment_54 = permute_signed(0u, matrix_fragment_54, 0x7610u);

        uniform_address_word_4 = 0x400u;

        matrix_fragment_52 = matrix_fragment_52 + 0x20u;

        matrix_fragment_60 = matrix_fragment_61 * 0x4u + matrix_fragment_60;

        matrix_fragment_56 = 0x80u;

        uniform_address_word_7 = 0u;

        uniform_address_word_9 = broadcast_active_lane(address_word_148);

        uniform_address_word_8 = uniform_address_word_4 + 0x4000u;

        index_word_149 = uint32_t(int32_t(matrix_fragment_52) >> 0x4u);

        split_address_words(matrix_fragment_66, matrix_fragment_67, 0ull);

        matrix_fragment_52 = 0x80u;

        condition_3 = int32_t(matrix_fragment_64) >= int32_t(index_word_149);

        condition_4 = int32_t(matrix_fragment_64) >= int32_t(index_word_149);

        split_address_words(matrix_fragment_62, matrix_fragment_63, 0ull);

        matrix_fragment_55 = !condition_4 ? matrix_fragment_60 : 0u;

        uniform_address_word_15 = (uniform_address_word_6 << 0x18u) + uniform_address_word_4;

        if (!condition_3) {
            matrix_fragment_53 = matrix_fragment_60 + 0x80u;

            matrix_fragment_52 = matrix_fragment_55 + 0x80u;

            matrix_fragment_56 = address_word_141;

            matrix_fragment_54 = permute_signed(matrix_fragment_57, matrix_fragment_54, 0x7610u);
        }

        uniform_address_word_4 = uniform_address_word_15;

        uniform_address_word_5 = uniform_address_word_7;

        matrix_fragment_57 = 0u;

        if (!condition_3) {
            condition_5 = uint32_t(matrix_fragment_53) != uint32_t(matrix_fragment_52);

            condition_1 = uint32_t(address_word_141) != uint32_t(matrix_fragment_56);
        }

        matrix_fragment_56 = address_word_148 * 0x2000u;

        uniform_address_word_7 = (uniform_address_word_6 << 0x18u) + uniform_address_word_8;

        if (!condition_3) {
            matrix_fragment_52 = !condition_5 ? matrix_fragment_54 : 0u;
        }

        split_address_words(matrix_fragment_58, matrix_fragment_59,
                            combine_words(matrix_fragment_56, matrix_fragment_57) +
                                combine_words(uniform_address_word_4, uniform_address_word_5));

        uniform_address_word_7 = (uniform_address_word_9 << 0x3u) + uniform_address_word_7;

        if (!condition_3) {
            matrix_fragment_52 = !condition_1 ? matrix_fragment_52 : 0u;
        }

        split_address_words(matrix_fragment_58, matrix_fragment_59,
                            uint64_t(uint32_t(address_word_139)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_fragment_58, matrix_fragment_59));

        if (!condition_3) {
            matrix_fragment_54 = permute_signed(matrix_fragment_52, matrix_fragment_54, 0x7610u);

            matrix_fragment_52 = parameters[parameter_input_low_0];
            matrix_fragment_53 = parameters[parameter_input_high_1];

            split_address_words(matrix_fragment_66, matrix_fragment_67,
                                combine_words(matrix_fragment_58, matrix_fragment_59));
        }

        matrix_fragment_54 = matrix_fragment_54 & 0xffffu;

        condition_1 = uint32_t(matrix_fragment_54) != 0u;

        if (!condition_3) {
            split_address_words(matrix_fragment_62, matrix_fragment_63,
                                int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_52, matrix_fragment_53));
        }
        if (!condition_1) {
            if (!condition_3) {
                matrix_fragment_52 = parameters[parameter_input_low_0];
                matrix_fragment_53 = parameters[parameter_input_high_1];
            }

            split_address_words(matrix_fragment_66, matrix_fragment_67, 0ull);

            matrix_fragment_63 = address_word_140 + matrix_fragment_56;

            if (!condition_3) {
                split_address_words(matrix_fragment_66, matrix_fragment_67,
                                    int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_52, matrix_fragment_53));
            }
            if (condition_3) {
                matrix_fragment_62 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_52 = uniform_address_word_4;

                matrix_fragment_53 = uniform_address_word_4;

                matrix_fragment_52 =
                    permute_signed(matrix_fragment_52, matrix_fragment_53, 0x5410u);

                matrix_fragment_53 = matrix_fragment_52;

                matrix_fragment_54 = matrix_fragment_52;

                matrix_fragment_55 = matrix_fragment_52;

                matrix_fragment_62 = matrix_fragment_62 * 0x10u + matrix_fragment_63;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_62,
                                                     matrix_fragment_52, matrix_fragment_53,
                                                     matrix_fragment_54, matrix_fragment_55,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_4 = broadcast_active_lane(matrix_fragment_66);

                    uniform_address_word_8 = 0x20u;

                    uniform_address_word_5 = broadcast_active_lane(matrix_fragment_67);

                    uniform_address_word_6 = broadcast_active_lane(matrix_fragment_63);

                    matrix_fragment_52 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, uniform_address_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            if (condition_3) {
                matrix_fragment_52 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_53 = uniform_address_word_4;

                matrix_fragment_54 = uniform_address_word_4;

                matrix_fragment_63 = matrix_fragment_52 * 0x10u + matrix_fragment_63;

                matrix_fragment_52 =
                    permute_signed(matrix_fragment_53, matrix_fragment_54, 0x5410u);

                matrix_fragment_53 = matrix_fragment_52;

                matrix_fragment_54 = matrix_fragment_52;

                matrix_fragment_55 = matrix_fragment_52;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_63 + 0x200u,
                                                     matrix_fragment_52, matrix_fragment_53,
                                                     matrix_fragment_54, matrix_fragment_55,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (!condition_3) {
                    matrix_fragment_52 = parameters[parameter_input_low_0];
                    matrix_fragment_53 = parameters[parameter_input_high_1];
                }

                matrix_fragment_55 = matrix_fragment_60 + 0x80u;

                uniform_address_word_4 = 0xffffffffu;

                split_address_words(matrix_fragment_66, matrix_fragment_67, 0ull);

                matrix_fragment_55 = !condition_4 ? matrix_fragment_55 : 0u;

                if (!condition_3) {
                    split_address_words(matrix_fragment_66, matrix_fragment_67,
                                        int64_t(int32_t(matrix_fragment_55)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_52, matrix_fragment_53));
                }

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_6 = broadcast_active_lane(matrix_fragment_63);

                    uniform_address_word_8 = 0x20u;

                    uniform_address_word_4 = broadcast_active_lane(matrix_fragment_66);

                    uniform_address_word_5 = broadcast_active_lane(matrix_fragment_67);

                    matrix_fragment_52 = 0x200u;

                    uniform_address_word_6 = uniform_address_word_6 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, uniform_address_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = broadcast_active_lane(matrix_fragment_66);

                uniform_address_word_16 = 0x40u;

                uniform_address_word_5 = broadcast_active_lane(matrix_fragment_67);

                uniform_address_word_8 = broadcast_active_lane(matrix_fragment_62);

                uniform_address_word_9 = broadcast_active_lane(matrix_fragment_63);

                matrix_fragment_52 = 0x400u;

                uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_16 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_16 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        matrix_fragment_64 = matrix_fragment_64 + 0x4u;

        matrix_fragment_52 = matrix_fragment_60 + 0x10000u;

        matrix_fragment_53 = 0x80u;

        matrix_fragment_62 = 0x1u;

        matrix_fragment_54 = permute_signed(0u, matrix_fragment_54, 0x7610u);

        condition_3 = int32_t(matrix_fragment_64) >= int32_t(index_word_149);

        condition_4 = int32_t(matrix_fragment_64) >= int32_t(index_word_149);

        split_address_words(matrix_fragment_64, matrix_fragment_65, 0ull);

        matrix_fragment_55 = !condition_4 ? matrix_fragment_52 : 0u;

        if (!condition_3) {
            matrix_fragment_52 = matrix_fragment_60 + 0x10080u;

            matrix_fragment_53 = matrix_fragment_55 + 0x80u;

            matrix_fragment_54 = permute_signed(matrix_fragment_62, matrix_fragment_54, 0x7610u);

            split_address_words(matrix_fragment_64, matrix_fragment_65,
                                combine_words(matrix_fragment_58, matrix_fragment_59) + 0x1000u);

            condition_1 = uint32_t(matrix_fragment_52) != uint32_t(matrix_fragment_53);

            matrix_fragment_52 = parameters[parameter_input_low_0];
            matrix_fragment_53 = parameters[parameter_input_high_1];

            matrix_fragment_62 = !condition_1 ? matrix_fragment_54 : 0u;

            matrix_fragment_54 = permute_signed(matrix_fragment_62, matrix_fragment_54, 0x7610u);
        }

        split_address_words(matrix_fragment_62, matrix_fragment_63, 0ull);

        matrix_fragment_54 = matrix_fragment_54 & 0xffffu;

        condition_1 = uint32_t(matrix_fragment_54) != 0u;

        if (!condition_3) {
            split_address_words(matrix_fragment_62, matrix_fragment_63,
                                int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_52, matrix_fragment_53));
        }
        if (!condition_1) {
            if (!condition_3) {
                matrix_fragment_52 = parameters[parameter_input_low_0];
                matrix_fragment_53 = parameters[parameter_input_high_1];
            }

            split_address_words(matrix_fragment_58, matrix_fragment_59, 0ull);

            matrix_fragment_57 = address_word_145 + matrix_fragment_56;

            if (!condition_3) {
                split_address_words(matrix_fragment_58, matrix_fragment_59,
                                    int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_52, matrix_fragment_53));
            }
            if (condition_3) {
                matrix_fragment_56 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_52 = uniform_address_word_4;

                matrix_fragment_53 = uniform_address_word_4;

                matrix_fragment_52 =
                    permute_signed(matrix_fragment_52, matrix_fragment_53, 0x5410u);

                matrix_fragment_53 = matrix_fragment_52;

                matrix_fragment_54 = matrix_fragment_52;

                matrix_fragment_55 = matrix_fragment_52;

                matrix_fragment_56 = matrix_fragment_56 * 0x10u + matrix_fragment_57;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_56,
                                                     matrix_fragment_52, matrix_fragment_53,
                                                     matrix_fragment_54, matrix_fragment_55,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                uniform_address_word_4 = 0xffffffffu;

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_4 = broadcast_active_lane(matrix_fragment_58);

                    uniform_address_word_8 = 0x20u;

                    uniform_address_word_5 = broadcast_active_lane(matrix_fragment_59);

                    uniform_address_word_6 = broadcast_active_lane(matrix_fragment_57);

                    matrix_fragment_52 = 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, uniform_address_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
            if (condition_3) {
                matrix_fragment_52 = threadIdx.x & 31u;

                uniform_address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                       select_half_operand<2, false, false>(0u));

                matrix_fragment_53 = uniform_address_word_4;

                matrix_fragment_54 = uniform_address_word_4;

                matrix_fragment_57 = matrix_fragment_52 * 0x10u + matrix_fragment_57;

                matrix_fragment_52 =
                    permute_signed(matrix_fragment_53, matrix_fragment_54, 0x5410u);

                matrix_fragment_53 = matrix_fragment_52;

                matrix_fragment_54 = matrix_fragment_52;

                matrix_fragment_55 = matrix_fragment_52;

                store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_57 + 0x200u,
                                                     matrix_fragment_52, matrix_fragment_53,
                                                     matrix_fragment_54, matrix_fragment_55,
                                                     shared_bytes, memory_ranges, error_counts);
            } else {
                if (!condition_3) {
                    matrix_fragment_52 = parameters[parameter_input_low_0];
                    matrix_fragment_53 = parameters[parameter_input_high_1];
                }

                matrix_fragment_55 = matrix_fragment_60 + 0x10080u;

                uniform_address_word_4 = 0xffffffffu;

                split_address_words(matrix_fragment_58, matrix_fragment_59, 0ull);

                matrix_fragment_55 = !condition_4 ? matrix_fragment_55 : 0u;

                if (!condition_3) {
                    split_address_words(matrix_fragment_58, matrix_fragment_59,
                                        int64_t(int32_t(matrix_fragment_55)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_52, matrix_fragment_53));
                }

                condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
                if (condition_1) {
                    uniform_address_word_6 = broadcast_active_lane(matrix_fragment_57);

                    uniform_address_word_8 = 0x20u;

                    uniform_address_word_4 = broadcast_active_lane(matrix_fragment_58);

                    uniform_address_word_5 = broadcast_active_lane(matrix_fragment_59);

                    matrix_fragment_52 = 0x200u;

                    uniform_address_word_6 = uniform_address_word_6 + 0x200u;

#if DLSS5_ENABLE_ASYNC_COPIES
                    copy_global_to_shared<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, uniform_address_word_7, shared_bytes,
                        memory_ranges, error_counts);
#else
                    bulk_copy<CheckBounds>(
                        shared_tile, uniform_address_word_6,
                        combine_words(uniform_address_word_4, uniform_address_word_5),
                        uniform_address_word_8 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
                }
            }
        } else {
            uniform_address_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_address_word_4 = broadcast_active_lane(matrix_fragment_64);

                uniform_address_word_16 = 0x40u;

                uniform_address_word_5 = broadcast_active_lane(matrix_fragment_65);

                uniform_address_word_8 = broadcast_active_lane(matrix_fragment_62);

                uniform_address_word_9 = broadcast_active_lane(matrix_fragment_63);

                matrix_fragment_52 = 0x400u;

                uniform_address_word_6 = uniform_address_word_4;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_16 * 16u, uniform_address_word_7, shared_bytes,
                    memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_8, uniform_address_word_9),
                    uniform_address_word_16 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        address_word_148 = threadIdx.x & 31u;

        matrix_fragment_52 = parameters[parameter_trained_weights_low_6];
        matrix_fragment_53 = parameters[parameter_trained_weights_high_7];

        matrix_fragment_61 = matrix_fragment_61 * 0x100u + reused_word_136;

        split_address_words(matrix_fragment_52, matrix_fragment_53,
                            int64_t(int32_t(matrix_fragment_61)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_52, matrix_fragment_53));

        split_address_words(matrix_fragment_84, matrix_fragment_85,
                            int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_52, matrix_fragment_53));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_84, matrix_fragment_85),
                                            matrix_fragment_80, matrix_fragment_81,
                                            matrix_fragment_82, matrix_fragment_83, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x200ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x400ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x600ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x8000ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x8200ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x8400ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x8600ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = uniform_address_word_7;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_86, matrix_fragment_87,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_86 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_1 = wait_copy_barrier(shared_tile, uniform_address_word_7,
                                        async_tokens[(((uniform_address_word_7) >> 3) & 1)]);
#else
        __syncthreads();
        condition_1 = true;
#endif
        if (!condition_1) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_1 =
                    wait_copy_barrier(shared_tile, uniform_address_word_7,
                                      async_tokens[(((uniform_address_word_7) >> 3) & 1)]);
#else
                __syncthreads();
                condition_1 = true;
#endif
            } while (!condition_1);
        }
    } while (!condition_2);
    index_word_147 = index_word_147 + uniform_address_word_15;

    matrix_fragment_85 = (index_word_146 << 0x2u) + uniform_address_word_15;

    address_word_140 = threadIdx.x;

    __syncwarp(__activemask());

    matrix_fragment_84 = address_word_148 * 0x10u + index_word_147;

    condition_0 = uint32_t(block_index_133) != 0u;

    address_word_138 = address_word_148 * 0x10u + matrix_fragment_85;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_84 + 0x2000u, matrix_fragment_84, matrix_fragment_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_138 + 0x2200u, matrix_fragment_88,
                                        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_138 + 0x2400u, matrix_fragment_92,
                                        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_138 + 0x2600u, matrix_fragment_96,
                                        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, address_word_138 + 0x2800u, matrix_fragment_100, matrix_fragment_101,
        matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, address_word_138 + 0x2a00u, matrix_fragment_104, matrix_fragment_105,
        matrix_fragment_106, matrix_fragment_107, shared_bytes, memory_ranges, error_counts);

    index_word_137 = address_word_140 | address_word_5;

    load_four_shared_words<CheckBounds>(
        shared_tile, address_word_138 + 0x2c00u, matrix_fragment_108, matrix_fragment_109,
        matrix_fragment_110, matrix_fragment_111, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, address_word_138 + 0x2e00u, matrix_fragment_112, matrix_fragment_113,
        matrix_fragment_114, matrix_fragment_115, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_12,
                             matrix_accumulator_13);

    multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_10,
                             matrix_accumulator_11);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_50,
                             matrix_accumulator_51);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_8,
                             matrix_accumulator_9);

    multiply_accumulate_half(address_word_6, address_word_7, matrix_fragment_84, matrix_fragment_85,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_72,
                             matrix_fragment_73, address_word_6, address_word_7);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_82, matrix_fragment_83, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_100,
                             matrix_fragment_101, matrix_fragment_102, matrix_fragment_103,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_40,
                             matrix_accumulator_41);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_18,
                             matrix_accumulator_19);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_16,
                             matrix_accumulator_17);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_108,
                             matrix_fragment_109, matrix_fragment_110, matrix_fragment_111,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_14,
                             matrix_accumulator_15);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_12,
                             matrix_accumulator_13);

    multiply_accumulate_half(matrix_accumulator_12, matrix_accumulator_13, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_10,
                             matrix_accumulator_11);

    multiply_accumulate_half(matrix_accumulator_10, matrix_accumulator_11, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = parameters[parameter_reduction_counter_optional_low_8];
    matrix_accumulator_3 = parameters[parameter_reduction_counter_optional_high_9];

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_122,
                             matrix_accumulator_123);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_132)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_50,
                             matrix_accumulator_51);

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_38,
                             matrix_accumulator_39);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_34,
                             matrix_accumulator_35);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_accumulator_14, matrix_accumulator_15, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_8,
                             matrix_accumulator_9);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                             matrix_accumulator_129);

    multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_56, matrix_fragment_57, address_word_6,
                             address_word_7);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_40,
                             matrix_accumulator_41);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_56, matrix_fragment_57, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_112,
                             matrix_fragment_113, matrix_fragment_114, matrix_fragment_115,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_70,
                             matrix_fragment_71);
    if (!condition_0) {
        address_word_6 = parameters[parameter_reduction_scratch_optional_low_10];
        address_word_7 = parameters[parameter_reduction_scratch_optional_high_11];

        address_word_5 = index_word_135 * 0x8u + index_word_134;

        condition_0 = int32_t(address_word_5) >= int32_t(index_word_149);

        packed_half_0 = address_word_5 + 0x1u;

        address_word_4 = address_word_5 + 0x2u;

        matrix_fragment_52 = address_word_5 + 0x3u;

        address_word_5 = address_word_5 * 0x2000u + reused_word_136;

        condition_1 = int32_t(packed_half_0) >= int32_t(index_word_149);

        condition_2 = int32_t(address_word_4) >= int32_t(index_word_149);

        condition_3 = int32_t(matrix_fragment_52) >= int32_t(index_word_149);

        split_address_words(address_word_6, address_word_7,
                            int64_t(int32_t(address_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_6, address_word_7));
        if (!condition_0) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_accumulator_16, matrix_accumulator_17,
                                                    matrix_accumulator_18, matrix_accumulator_19,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x200ull, matrix_accumulator_12,
                matrix_accumulator_13, matrix_accumulator_14, matrix_accumulator_15, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x400ull, matrix_accumulator_8,
                matrix_accumulator_9, matrix_accumulator_10, matrix_accumulator_11, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x600ull, matrix_accumulator_20,
                matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
                error_counts);
        }
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            combine_words(address_word_6, address_word_7) + 0x8000u);
        if (!condition_1) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_8, matrix_accumulator_9));

            split_address_words(matrix_accumulator_10, matrix_accumulator_11,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_accumulator_124, matrix_accumulator_125,
                                                    matrix_accumulator_126, matrix_accumulator_127,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_10, matrix_accumulator_11) + 0x8200ull,
                matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_122,
                matrix_accumulator_123, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_10, matrix_accumulator_11) + 0x8400ull,
                matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_118,
                matrix_accumulator_119, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_10, matrix_accumulator_11) + 0x8600ull,
                matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
                matrix_accumulator_51, memory_ranges, error_counts);
        }
        split_address_words(address_word_6, address_word_7,
                            combine_words(address_word_6, address_word_7) + 0x10000u);
        if (!condition_2) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_8, matrix_accumulator_9));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_accumulator_44, matrix_accumulator_45,
                                                    matrix_accumulator_46, matrix_accumulator_47,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x8200ull,
                matrix_accumulator_40, matrix_accumulator_41, matrix_accumulator_42,
                matrix_accumulator_43, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x8400ull,
                matrix_accumulator_36, matrix_accumulator_37, matrix_accumulator_38,
                matrix_accumulator_39, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x8600ull,
                matrix_accumulator_32, matrix_accumulator_33, matrix_accumulator_34,
                matrix_accumulator_35, memory_ranges, error_counts);
        }
        if (!condition_3) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8000ull, matrix_accumulator_28,
                matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8200ull, matrix_accumulator_24,
                matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8400ull, matrix_fragment_72,
                matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8600ull, matrix_fragment_68,
                matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                error_counts);
        }
    } else {
        address_word_6 = threadIdx.z;

        condition_0 = (address_word_6 | index_word_137) != 0;
        if (!condition_0) {
            address_word_6 = read_device_flag<CheckBounds>(
                combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                error_counts);

            address_word_5 = block_index_133 + 0xffffffffu;

            condition_0 = int32_t(address_word_6) >= int32_t(address_word_5);
            if (!condition_0) {
                do {
                    __nanosleep(0x40u);

                    address_word_6 = read_device_flag<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    condition_0 = int32_t(address_word_6) >= int32_t(address_word_5);
                } while (!condition_0);
            }
        }
        condition_0 = uint32_t(block_index_133) >= uint32_t(0x3u);

        __syncwarp(__activemask());

        __syncthreads();
        if (!condition_0) {
            {
                // The first three split-projection blocks contribute to shared reduction
                // tiles; the last block consumes them through the existing completion path.
                // Bounds are in 16-token tiles, rather than individual image pixels.
                const uint32_t first_token_tile = index_word_135 * 8 + index_word_134;
                const uint32_t token_tile_count = index_word_149;
                const uint32_t feature_word_offset = reused_word_136;
                const uint32_t lane = address_word_148;
                const uint64_t reduction_scratch =
                    combine_words(parameters[parameter_reduction_scratch_optional_low_10],
                                  parameters[parameter_reduction_scratch_optional_high_11]);

                if (int32_t(first_token_tile) < int32_t(token_tile_count)) {
                    accumulate_feed_forward_tile<CheckBounds>(
                        reduction_scratch, first_token_tile, feature_word_offset, lane,
                        {matrix_accumulator_16, matrix_accumulator_17, matrix_accumulator_18,
                         matrix_accumulator_19},
                        {matrix_accumulator_12, matrix_accumulator_13, matrix_accumulator_14,
                         matrix_accumulator_15},
                        {matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_10,
                         matrix_accumulator_11},
                        {matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_22,
                         matrix_accumulator_23},
                        memory_ranges, error_counts);
                }

                if (int32_t(first_token_tile + 1) < int32_t(token_tile_count)) {
                    accumulate_feed_forward_tile<CheckBounds>(
                        reduction_scratch, first_token_tile + 1, feature_word_offset, lane,
                        {matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_126,
                         matrix_accumulator_127},
                        {matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_122,
                         matrix_accumulator_123},
                        {matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_118,
                         matrix_accumulator_119},
                        {matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
                         matrix_accumulator_51},
                        memory_ranges, error_counts);
                }

                if (int32_t(first_token_tile + 2) < int32_t(token_tile_count)) {
                    accumulate_feed_forward_tile<CheckBounds>(
                        reduction_scratch, first_token_tile + 2, feature_word_offset, lane,
                        {matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_46,
                         matrix_accumulator_47},
                        {matrix_accumulator_40, matrix_accumulator_41, matrix_accumulator_42,
                         matrix_accumulator_43},
                        {matrix_accumulator_36, matrix_accumulator_37, matrix_accumulator_38,
                         matrix_accumulator_39},
                        {matrix_accumulator_32, matrix_accumulator_33, matrix_accumulator_34,
                         matrix_accumulator_35},
                        memory_ranges, error_counts);
                }

                if (int32_t(first_token_tile + 3) < int32_t(token_tile_count)) {
                    accumulate_feed_forward_tile<CheckBounds>(
                        reduction_scratch, first_token_tile + 3, feature_word_offset, lane,
                        {matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_30,
                         matrix_accumulator_31},
                        {matrix_accumulator_24, matrix_accumulator_25, matrix_accumulator_26,
                         matrix_accumulator_27},
                        {matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
                         matrix_fragment_75},
                        {matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                         matrix_fragment_71},
                        memory_ranges, error_counts);
                }
            }
        } else {
            address_word_6 = parameters[parameter_reduction_scratch_optional_low_10];
            address_word_7 = parameters[parameter_reduction_scratch_optional_high_11];

            index_word_135 = index_word_135 * 0x8u + index_word_134;

            matrix_fragment_56 = 0;
            matrix_fragment_57 = 0;

            matrix_fragment_58 = 0;
            matrix_fragment_59 = 0;

            matrix_fragment_52 = 0;
            matrix_fragment_53 = 0;

            address_word_5 = index_word_135 * 0x2000u + reused_word_136;

            condition_0 = int32_t(index_word_135) >= int32_t(index_word_149);

            matrix_fragment_60 = index_word_135 + 0x1u;

            matrix_fragment_62 = index_word_135 + 0x2u;

            matrix_fragment_82 = 0;
            matrix_fragment_83 = 0;

            matrix_fragment_84 = 0;
            matrix_fragment_85 = 0;

            matrix_fragment_86 = 0;
            matrix_fragment_87 = 0;

            condition_1 = int32_t(matrix_fragment_60) >= int32_t(index_word_149);

            condition_2 = int32_t(matrix_fragment_62) >= int32_t(index_word_149);

            matrix_fragment_64 = index_word_135 + 0x3u;

            matrix_fragment_54 = 0;
            matrix_fragment_55 = 0;

            condition_3 = int32_t(matrix_fragment_64) >= int32_t(index_word_149);

            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(address_word_5)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_6, address_word_7));

            split_address_words(matrix_fragment_80, matrix_fragment_81,
                                combine_words(address_word_6, address_word_7) + 0x400u);

            split_address_words(matrix_fragment_100, matrix_fragment_101,
                                combine_words(address_word_6, address_word_7) + 0x8400u);

            split_address_words(matrix_fragment_102, matrix_fragment_103,
                                combine_words(address_word_6, address_word_7) + 0x8600u);

            if (!condition_0) {
                split_address_words(matrix_fragment_80, matrix_fragment_81,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_80, matrix_fragment_81));
            }

            split_address_words(address_word_138, address_word_139,
                                combine_words(address_word_6, address_word_7) + 0x10600u);

            split_address_words(matrix_fragment_78, matrix_fragment_79,
                                combine_words(address_word_6, address_word_7) + 0x200u);

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_80, matrix_fragment_81), matrix_fragment_56,
                    matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
                    error_counts);
            }

            if (!condition_1) {
                split_address_words(matrix_fragment_100, matrix_fragment_101,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_100, matrix_fragment_101));

                split_address_words(matrix_fragment_102, matrix_fragment_103,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_102, matrix_fragment_103));
            }

            if (!condition_2) {
                split_address_words(address_word_138, address_word_139,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_138, address_word_139));
            }

            matrix_fragment_80 = 0;
            matrix_fragment_81 = 0;

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_102, matrix_fragment_103), matrix_fragment_84,
                    matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges,
                    error_counts);
            }

            split_address_words(matrix_fragment_94, matrix_fragment_95,
                                combine_words(address_word_6, address_word_7) + 0x600u);

            if (!condition_0) {
                split_address_words(matrix_fragment_78, matrix_fragment_79,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_78, matrix_fragment_79));
            }

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_100, matrix_fragment_101), matrix_fragment_80,
                    matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
                    error_counts);
            }

            matrix_fragment_102 = 0;
            matrix_fragment_103 = 0;

            split_address_words(matrix_fragment_96, matrix_fragment_97,
                                combine_words(address_word_6, address_word_7) + 0x8000u);

            split_address_words(matrix_fragment_98, matrix_fragment_99,
                                combine_words(address_word_6, address_word_7) + 0x8200u);

            matrix_fragment_100 = 0;
            matrix_fragment_101 = 0;

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_78, matrix_fragment_79), matrix_fragment_52,
                    matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
                    error_counts);
            }

            if (!condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_138, address_word_139), matrix_fragment_100,
                    matrix_fragment_101, matrix_fragment_102, matrix_fragment_103, memory_ranges,
                    error_counts);
            }

            matrix_fragment_60 = 0;
            matrix_fragment_61 = 0;

            matrix_fragment_62 = 0;
            matrix_fragment_63 = 0;

            matrix_fragment_64 = 0;
            matrix_fragment_65 = 0;

            matrix_fragment_66 = 0;
            matrix_fragment_67 = 0;

            matrix_fragment_76 = 0;
            matrix_fragment_77 = 0;

            matrix_fragment_78 = 0;
            matrix_fragment_79 = 0;

            if (!condition_0) {
                split_address_words(matrix_fragment_94, matrix_fragment_95,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_94, matrix_fragment_95));
            }

            split_address_words(address_word_140, address_word_141,
                                combine_words(address_word_6, address_word_7) + 0x18000u);

            split_address_words(matrix_fragment_114, matrix_fragment_115,
                                combine_words(address_word_6, address_word_7) + 0x10000u);

            if (!condition_1) {
                split_address_words(matrix_fragment_96, matrix_fragment_97,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_96, matrix_fragment_97));
            }

            split_address_words(matrix_accumulator_128, matrix_accumulator_129,
                                combine_words(address_word_6, address_word_7) + 0x10200u);

            split_address_words(matrix_accumulator_130, matrix_accumulator_131,
                                combine_words(address_word_6, address_word_7) + 0x10400u);

            if (!condition_1) {
                split_address_words(matrix_fragment_98, matrix_fragment_99,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_98, matrix_fragment_99));
            }

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_94, matrix_fragment_95), matrix_fragment_60,
                    matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
                    error_counts);
            }

            matrix_fragment_104 = 0;
            matrix_fragment_105 = 0;

            matrix_fragment_106 = 0;
            matrix_fragment_107 = 0;

            matrix_fragment_88 = 0;
            matrix_fragment_89 = 0;

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_96, matrix_fragment_97), matrix_fragment_64,
                    matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
                    error_counts);
            }

            matrix_fragment_90 = 0;
            matrix_fragment_91 = 0;

            matrix_fragment_92 = 0;
            matrix_fragment_93 = 0;

            if (!condition_3) {
                split_address_words(address_word_140, address_word_141,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_140, address_word_141));
            }

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_98, matrix_fragment_99), matrix_fragment_76,
                    matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
                    error_counts);
            }

            split_address_words(address_word_142, address_word_143,
                                combine_words(address_word_6, address_word_7) + 0x18200u);

            matrix_fragment_94 = 0;
            matrix_fragment_95 = 0;

            if (!condition_2) {
                split_address_words(matrix_fragment_114, matrix_fragment_115,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_114, matrix_fragment_115));
            }

            split_address_words(address_word_144, address_word_145,
                                combine_words(address_word_6, address_word_7) + 0x18400u);

            matrix_fragment_96 = 0;
            matrix_fragment_97 = 0;

            if (!condition_2) {
                split_address_words(
                    matrix_accumulator_128, matrix_accumulator_129,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_128, matrix_accumulator_129));
            }

            if (!condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_140, address_word_141), matrix_fragment_104,
                    matrix_fragment_105, matrix_fragment_106, matrix_fragment_107, memory_ranges,
                    error_counts);
            }

            matrix_fragment_98 = 0;
            matrix_fragment_99 = 0;

            if (!condition_2) {
                split_address_words(
                    matrix_accumulator_130, matrix_accumulator_131,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_130, matrix_accumulator_131));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_114, matrix_fragment_115), matrix_fragment_88,
                    matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges,
                    error_counts);
            }

            matrix_fragment_108 = 0;
            matrix_fragment_109 = 0;

            matrix_fragment_110 = 0;
            matrix_fragment_111 = 0;

            if (!condition_0) {
                split_address_words(address_word_138, address_word_139,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_6, address_word_7));
            }

            if (!condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_128, matrix_accumulator_129),
                    matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                    memory_ranges, error_counts);
            }

            matrix_fragment_112 = 0;
            matrix_fragment_113 = 0;

            address_word_5 = 0u;

            if (!condition_3) {
                split_address_words(address_word_140, address_word_141,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_142, address_word_143));
            }

            if (!condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_130, matrix_accumulator_131),
                    matrix_fragment_96, matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                    memory_ranges, error_counts);
            }

            address_word_6 = 0;
            address_word_7 = 0;

            matrix_fragment_114 = 0;
            matrix_fragment_115 = 0;

            if (!condition_3) {
                split_address_words(address_word_142, address_word_143,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_144, address_word_145));

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_140, address_word_141), matrix_fragment_108,
                    matrix_fragment_109, matrix_fragment_110, matrix_fragment_111, memory_ranges,
                    error_counts);
            }

            matrix_accumulator_128 = 0;
            matrix_accumulator_129 = 0;

            if (!condition_3) {
                split_address_words(address_word_144, address_word_145,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_144, address_word_145));
            }

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_138, address_word_139), matrix_fragment_112,
                    matrix_fragment_113, matrix_fragment_114, matrix_fragment_115, memory_ranges,
                    error_counts);
            }

            matrix_accumulator_130 = 0;
            matrix_accumulator_131 = 0;

            if (!condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_144, address_word_145) + 0x200ull, address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_142, address_word_143), matrix_accumulator_128,
                    matrix_accumulator_129, matrix_accumulator_130, matrix_accumulator_131,
                    memory_ranges, error_counts);
            }

            packed_half_0 = index_word_135 * 0x400u + packed_half_0;

            matrix_accumulator_8 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                               select_half_operand<2, false, false>(matrix_accumulator_8));

            matrix_accumulator_9 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                               select_half_operand<2, false, false>(matrix_accumulator_9));

            matrix_accumulator_10 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                               select_half_operand<2, false, false>(matrix_accumulator_10));

            matrix_accumulator_11 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                               select_half_operand<2, false, false>(matrix_accumulator_11));

            matrix_fragment_100 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_100),
                               select_half_operand<2, false, false>(matrix_accumulator_32));

            matrix_fragment_101 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_101),
                               select_half_operand<2, false, false>(matrix_accumulator_33));

            matrix_accumulator_32 = parameters[parameter_output_low_4];
            matrix_accumulator_33 = parameters[parameter_output_high_5];

            matrix_accumulator_13 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                               select_half_operand<2, false, false>(matrix_accumulator_13));

            matrix_fragment_53 = packed_half_0 * 0x4u;

            matrix_accumulator_12 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                               select_half_operand<2, false, false>(matrix_accumulator_12));

            matrix_accumulator_14 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                               select_half_operand<2, false, false>(matrix_accumulator_14));

            matrix_accumulator_15 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                               select_half_operand<2, false, false>(matrix_accumulator_15));

            matrix_fragment_80 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                               select_half_operand<2, false, false>(matrix_accumulator_116));

            matrix_fragment_81 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                               select_half_operand<2, false, false>(matrix_accumulator_117));

            matrix_fragment_82 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                               select_half_operand<2, false, false>(matrix_accumulator_118));

            matrix_accumulator_20 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                               select_half_operand<2, false, false>(matrix_accumulator_20));

            matrix_accumulator_21 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                               select_half_operand<2, false, false>(matrix_accumulator_21));

            matrix_accumulator_22 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                               select_half_operand<2, false, false>(matrix_accumulator_22));

            matrix_accumulator_23 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                               select_half_operand<2, false, false>(matrix_accumulator_23));

            matrix_fragment_64 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                               select_half_operand<2, false, false>(matrix_accumulator_124));

            split_address_words(matrix_accumulator_32, matrix_accumulator_33,
                                int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_32, matrix_accumulator_33));

            matrix_fragment_65 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                               select_half_operand<2, false, false>(matrix_accumulator_125));

            matrix_fragment_66 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                               select_half_operand<2, false, false>(matrix_accumulator_126));

            matrix_fragment_67 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                               select_half_operand<2, false, false>(matrix_accumulator_127));

            matrix_fragment_76 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                               select_half_operand<2, false, false>(matrix_accumulator_120));

            matrix_fragment_77 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                               select_half_operand<2, false, false>(matrix_accumulator_121));

            matrix_fragment_78 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                               select_half_operand<2, false, false>(matrix_accumulator_122));

            matrix_fragment_79 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                               select_half_operand<2, false, false>(matrix_accumulator_123));

            matrix_fragment_83 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                               select_half_operand<2, false, false>(matrix_accumulator_119));

            matrix_accumulator_48 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                               select_half_operand<2, false, false>(matrix_accumulator_48));

            matrix_accumulator_49 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                               select_half_operand<2, false, false>(matrix_accumulator_49));

            matrix_accumulator_50 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                               select_half_operand<2, false, false>(matrix_accumulator_50));

            matrix_accumulator_51 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                               select_half_operand<2, false, false>(matrix_accumulator_51));

            matrix_accumulator_44 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                               select_half_operand<2, false, false>(matrix_accumulator_44));

            matrix_accumulator_45 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                               select_half_operand<2, false, false>(matrix_accumulator_45));

            matrix_accumulator_46 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                               select_half_operand<2, false, false>(matrix_accumulator_46));

            matrix_accumulator_47 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                               select_half_operand<2, false, false>(matrix_accumulator_47));

            matrix_accumulator_40 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                               select_half_operand<2, false, false>(matrix_accumulator_40));

            matrix_accumulator_41 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                               select_half_operand<2, false, false>(matrix_accumulator_41));

            matrix_accumulator_42 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                               select_half_operand<2, false, false>(matrix_accumulator_42));

            matrix_accumulator_43 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                               select_half_operand<2, false, false>(matrix_accumulator_43));

            matrix_accumulator_36 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                               select_half_operand<2, false, false>(matrix_accumulator_36));

            matrix_accumulator_37 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                               select_half_operand<2, false, false>(matrix_accumulator_37));

            matrix_accumulator_38 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                               select_half_operand<2, false, false>(matrix_accumulator_38));

            matrix_accumulator_39 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                               select_half_operand<2, false, false>(matrix_accumulator_39));

            matrix_accumulator_34 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                               select_half_operand<2, false, false>(matrix_accumulator_34));

            matrix_accumulator_35 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                               select_half_operand<2, false, false>(matrix_accumulator_35));

            matrix_accumulator_28 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                               select_half_operand<2, false, false>(matrix_accumulator_28));

            matrix_accumulator_29 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                               select_half_operand<2, false, false>(matrix_accumulator_29));

            matrix_accumulator_30 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106),
                               select_half_operand<2, false, false>(matrix_accumulator_30));

            matrix_accumulator_31 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_107),
                               select_half_operand<2, false, false>(matrix_accumulator_31));

            matrix_accumulator_24 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108),
                               select_half_operand<2, false, false>(matrix_accumulator_24));

            matrix_accumulator_25 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109),
                               select_half_operand<2, false, false>(matrix_accumulator_25));

            matrix_accumulator_26 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_110),
                               select_half_operand<2, false, false>(matrix_accumulator_26));

            matrix_accumulator_27 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                               select_half_operand<2, false, false>(matrix_accumulator_27));

            matrix_accumulator_16 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                               select_half_operand<2, false, false>(matrix_accumulator_16));

            matrix_accumulator_17 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                               select_half_operand<2, false, false>(matrix_accumulator_17));

            matrix_accumulator_18 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_114),
                               select_half_operand<2, false, false>(matrix_accumulator_18));

            matrix_accumulator_19 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_115),
                               select_half_operand<2, false, false>(matrix_accumulator_19));

            matrix_fragment_72 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                               select_half_operand<2, false, false>(matrix_fragment_72));

            matrix_fragment_73 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                               select_half_operand<2, false, false>(matrix_fragment_73));

            matrix_fragment_74 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                               select_half_operand<2, false, false>(matrix_fragment_74));

            matrix_fragment_75 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                               select_half_operand<2, false, false>(matrix_fragment_75));

            matrix_fragment_68 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_4),
                               select_half_operand<2, false, false>(matrix_fragment_68));

            matrix_fragment_69 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_5),
                               select_half_operand<2, false, false>(matrix_fragment_69));

            matrix_fragment_70 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_6),
                               select_half_operand<2, false, false>(matrix_fragment_70));

            matrix_fragment_71 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_7),
                               select_half_operand<2, false, false>(matrix_fragment_71));
            if (!condition_0) {
                address_word_4 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_15),
                                  select_half_operand<2, false, false>(0u));

                packed_half_0 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                                  select_half_operand<2, false, false>(0u));

                address_word_6 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_11),
                                  select_half_operand<2, false, false>(0u));

                address_word_7 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_13),
                                  select_half_operand<2, false, false>(address_word_4));

                address_word_5 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                  select_half_operand<2, false, false>(packed_half_0));

                matrix_accumulator_13 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                                  select_half_operand<2, false, false>(0u));

                matrix_accumulator_15 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_14),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_52 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                  select_half_operand<2, false, false>(0u));

                matrix_accumulator_17 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_10),
                                  select_half_operand<2, false, false>(0u));

                matrix_accumulator_19 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                  select_half_operand<2, false, false>(0u));

                matrix_accumulator_9 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                                  select_half_operand<2, false, false>(address_word_6));

                address_word_4 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                  select_half_operand<2, false, false>(matrix_accumulator_13));

                address_word_6 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_12),
                                  select_half_operand<2, false, false>(matrix_accumulator_15));

                split_address_words(
                    matrix_accumulator_12, matrix_accumulator_13,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_32, matrix_accumulator_33));

                matrix_accumulator_11 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                  select_half_operand<2, false, false>(matrix_fragment_52));

                matrix_accumulator_8 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                                  select_half_operand<2, false, false>(matrix_accumulator_17));

                matrix_accumulator_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                  select_half_operand<2, false, false>(matrix_accumulator_19));

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_12, matrix_accumulator_13), address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x200ull,
                    matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_10,
                    matrix_accumulator_11, memory_ranges, error_counts);
            }
            split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                                combine_words(matrix_accumulator_32, matrix_accumulator_33) +
                                    0x4000u);

            address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                           select_half_operand<2, false, false>(0u));

            matrix_accumulator_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                              select_half_operand<2, false, false>(0u));

            address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                           select_half_operand<2, false, false>(0u));

            packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                          select_half_operand<2, false, false>(0u));

            address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                           select_half_operand<2, false, false>(0u));

            matrix_accumulator_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                              select_half_operand<2, false, false>(0u));

            address_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                           select_half_operand<2, false, false>(0u));

            matrix_accumulator_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_82 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_83 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_50 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_49 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                              select_half_operand<2, false, false>(0u));
            if (!condition_1) {
                address_word_7 = permute_signed(address_word_7, matrix_accumulator_10, 0x5410u);

                address_word_4 = permute_signed(address_word_4, matrix_accumulator_11, 0x5410u);

                split_address_words(matrix_accumulator_10, matrix_accumulator_11,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_accumulator_8, matrix_accumulator_9));

                address_word_6 = permute_signed(address_word_6, matrix_accumulator_13, 0x5410u);

                address_word_5 = permute_signed(address_word_5, packed_half_0, 0x5410u);

                packed_half_0 =
                    permute_signed(matrix_accumulator_49, matrix_accumulator_12, 0x5410u);

                split_address_words(
                    matrix_accumulator_12, matrix_accumulator_13,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_32, matrix_accumulator_33));

                matrix_accumulator_15 =
                    permute_signed(matrix_accumulator_15, matrix_accumulator_50, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_10, matrix_accumulator_11), address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);

                matrix_accumulator_14 =
                    permute_signed(matrix_accumulator_14, matrix_fragment_83, 0x5410u);

                matrix_accumulator_17 =
                    permute_signed(matrix_accumulator_17, matrix_fragment_82, 0x5410u);

                address_word_4 = matrix_accumulator_17;

                address_word_5 = matrix_accumulator_14;

                address_word_6 = matrix_accumulator_15;

                address_word_7 = packed_half_0;

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x4200ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);
            }
            split_address_words(matrix_accumulator_32, matrix_accumulator_33,
                                combine_words(matrix_accumulator_32, matrix_accumulator_33) +
                                    0x8000u);

            address_word_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_46 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                              select_half_operand<2, false, false>(0u));

            packed_half_0 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_47 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_40),
                              select_half_operand<2, false, false>(0u));

            address_word_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_42),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_41),
                              select_half_operand<2, false, false>(0u));

            address_word_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_43),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_16 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                              select_half_operand<2, false, false>(0u));

            address_word_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_100),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_35 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                              select_half_operand<2, false, false>(0u));
            if (!condition_2) {
                packed_half_0 = permute_signed(packed_half_0, matrix_accumulator_47, 0x5410u);

                address_word_7 = permute_signed(address_word_7, matrix_accumulator_46, 0x5410u);

                matrix_accumulator_15 =
                    permute_signed(matrix_accumulator_15, matrix_accumulator_12, 0x5410u);

                matrix_accumulator_16 =
                    permute_signed(matrix_accumulator_16, matrix_accumulator_13, 0x5410u);

                split_address_words(matrix_accumulator_12, matrix_accumulator_13,
                                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_accumulator_8, matrix_accumulator_9));

                matrix_accumulator_11 =
                    permute_signed(matrix_accumulator_11, address_word_4, 0x5410u);

                matrix_accumulator_10 =
                    permute_signed(matrix_accumulator_10, address_word_5, 0x5410u);

                split_address_words(
                    address_word_4, address_word_5,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_32, matrix_accumulator_33));

                matrix_accumulator_8 = address_word_7;

                matrix_accumulator_9 = packed_half_0;

                matrix_accumulator_14 =
                    permute_signed(matrix_accumulator_14, matrix_accumulator_35, 0x5410u);

                address_word_6 = permute_signed(address_word_6, matrix_accumulator_17, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_accumulator_8,
                    matrix_accumulator_9, matrix_accumulator_10, matrix_accumulator_11,
                    memory_ranges, error_counts);

                address_word_7 = matrix_accumulator_14;

                address_word_4 = matrix_accumulator_16;

                address_word_5 = matrix_accumulator_15;

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_12, matrix_accumulator_13) + 0x4200ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);
            }
            address_word_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                              select_half_operand<2, false, false>(0u));

            address_word_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                              select_half_operand<2, false, false>(0u));

            packed_half_0 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                              select_half_operand<2, false, false>(0u));

            address_word_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                              select_half_operand<2, false, false>(0u));

            address_word_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_8 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_72 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_73 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_12 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_69 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                              select_half_operand<2, false, false>(0u));
            if (!condition_3) {
                address_word_7 = permute_signed(address_word_7, matrix_accumulator_8, 0x5410u);

                split_address_words(
                    matrix_accumulator_32, matrix_accumulator_33,
                    int64_t(int32_t(address_word_148)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_32, matrix_accumulator_33));

                address_word_6 = permute_signed(address_word_6, matrix_accumulator_11, 0x5410u);

                address_word_4 = permute_signed(address_word_4, matrix_accumulator_9, 0x5410u);

                address_word_5 = permute_signed(address_word_5, packed_half_0, 0x5410u);

                matrix_accumulator_11 =
                    permute_signed(matrix_fragment_69, matrix_accumulator_14, 0x5410u);

                matrix_accumulator_10 =
                    permute_signed(matrix_accumulator_10, matrix_accumulator_15, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x4000ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);

                matrix_accumulator_9 =
                    permute_signed(matrix_fragment_73, matrix_accumulator_12, 0x5410u);

                matrix_accumulator_8 =
                    permute_signed(matrix_fragment_72, matrix_accumulator_13, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_accumulator_32, matrix_accumulator_33) + 0x4200ull,
                    matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_10,
                    matrix_accumulator_11, memory_ranges, error_counts);
            }
        }
    }
    packed_half_0 = threadIdx.z;

    __syncthreads();

    condition_0 = (index_word_137 | packed_half_0) != 0;

    if (condition_0) {
        return;
    }

    condition_0 = uint32_t(block_index_133) != uint32_t(0x3u);

    __threadfence();

    write_device_flag<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                   block_index_133, memory_ranges, error_counts);

    if (condition_0) {
        return;
    }

    matrix_accumulator_2 = parameters[parameter_publication_optional_low_14];
    matrix_accumulator_3 = parameters[parameter_publication_optional_high_15];

    split_address_words(address_word_132, block_index_133,
                        int64_t(int32_t(address_word_132)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __threadfence();

    write_device_flag<CheckBounds>(combine_words(address_word_132, block_index_133), 0u,
                                   memory_ranges, error_counts);
    return;
}
extern "C" __global__
__maxnreg__(170) void phoenix_global_attention_transformer_feed_forward_contract(
    KernelArguments parameters) {
    run_global_attention_transformer_feed_forward_contract<false>(parameters.words, nullptr,
                                                                  nullptr);
}
