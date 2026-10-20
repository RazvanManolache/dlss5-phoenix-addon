// DLSS5 Phoenix optimized kernel: global_attention/transformer/projection_wait.cu
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
    parameter_LH_16 = 16,
    parameter_LW_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[18];
};
static_assert(sizeof(KernelArguments) == 72, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_global_attention_transformer_projection_wait(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t address_word_4 = 0;
    uint32_t address_word_5 = 0;
    uint32_t address_word_6 = 0;
    uint32_t address_word_7 = 0;
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
    uint32_t matrix_fragment_36 = 0;
    uint32_t matrix_fragment_37 = 0;
    uint32_t matrix_fragment_38 = 0;
    uint32_t matrix_fragment_39 = 0;
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
    uint32_t matrix_accumulator_68 = 0;
    uint32_t matrix_accumulator_69 = 0;
    uint32_t matrix_accumulator_70 = 0;
    uint32_t matrix_accumulator_71 = 0;
    uint32_t matrix_accumulator_72 = 0;
    uint32_t matrix_accumulator_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
    uint32_t index_word_82 = 0;
    uint32_t address_word_83 = 0;
    uint32_t matrix_accumulator_84 = 0;
    uint32_t matrix_accumulator_85 = 0;
    uint32_t index_word_86 = 0;
    uint32_t index_word_87 = 0;
    uint32_t matrix_accumulator_88 = 0;
    uint32_t matrix_accumulator_89 = 0;
    uint32_t address_word_90 = 0;
    uint32_t index_word_91 = 0;
    uint32_t matrix_accumulator_92 = 0;
    uint32_t matrix_accumulator_93 = 0;
    uint32_t matrix_accumulator_94 = 0;
    uint32_t matrix_accumulator_95 = 0;
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t matrix_accumulator_98 = 0;
    uint32_t matrix_accumulator_99 = 0;
    uint32_t thread_row_100 = 0;
    uint32_t warp_lane_101 = 0;
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
    uint32_t packed_half_116 = 0;
    uint32_t packed_half_117 = 0;
    uint32_t packed_half_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t address_word_120 = 0;
    uint32_t address_word_121 = 0;
    uint32_t address_word_122 = 0;
    uint32_t address_word_123 = 0;
    uint32_t address_word_124 = 0;
    uint32_t address_word_125 = 0;
    uint32_t packed_half_126 = 0;
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
    uint32_t address_word_140 = 0;
    uint32_t address_word_141 = 0;
    uint32_t address_word_142 = 0;
    uint32_t address_word_143 = 0;
    uint32_t address_word_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t address_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_address_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
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

    matrix_fragment_25 = threadIdx.y;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    matrix_fragment_26 = threadIdx.x;

    matrix_fragment_24 = blockIdx.x;

    matrix_fragment_13 = threadIdx.x & 31u;

    address_word_4 = address_word_5 * address_word_4 + 0xffffffffu;

    address_word_7 = uint32_t(int32_t(address_word_4) >> 0x1fu);

    address_word_6 = truncate_signed_power_of_two<7>(address_word_4);

    condition_1 = (matrix_fragment_25 | matrix_fragment_26) != 0;

    matrix_fragment_51 = address_word_6 + 0x1u;

    const auto work = token_tile_work(matrix_fragment_24, matrix_fragment_51);
    const auto copy = tile_copy_schedule<shared_bytes, 2>(!condition_1, blockDim.x * blockDim.y);
    uniform_condition_0 = copy.without_block_leader;
    uniform_condition_1 = copy.without_block_leader;
    uniform_index_word_4 = copy.wait_word_low;
    uniform_address_word_5 = copy.wait_word_high;
    uniform_address_word_6 = copy.barrier_byte_offset;
    address_word_5 = work.group;
    index_word_0 = blockIdx.z;
    address_word_2 = parameters[parameter_trained_weights_low_6];
    address_word_3 = parameters[parameter_trained_weights_high_7];
    index_word_82 = (matrix_fragment_25 * 64u) & 64u;

    index_word_82 = address_word_5 * 0x80u + index_word_82;

    address_word_83 = index_word_82 * 0x8u;

    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }

    matrix_fragment_9 = index_word_0 * 0x10000u + address_word_83;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_2, address_word_3));

    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3),
                                        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22,
                                        matrix_fragment_23, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x200ull,
                                        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18,
                                        matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x400ull,
                                        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14,
                                        matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_2, address_word_3) + 0x600ull,
                                        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10,
                                        matrix_fragment_11, memory_ranges, error_counts);

    condition_0 = (matrix_fragment_25 & 0x1u) != 0;

    address_word_4 = uint32_t(combine_words(address_word_7, 0u) << 0x5u >> 32) + address_word_4;

    condition_0 = uint32_t(matrix_fragment_26) != 0u || condition_0;

    address_word_4 = address_word_4 & 0xffffffe0u;

    index_word_86 = work.tile;

    index_word_87 = address_word_4 + 0x20u;

    address_word_4 = 0u;

    address_word_6 = index_word_86 * 0x80u;

    index_word_87 = uint32_t(int32_t(index_word_87) >> 0x4u);
    if (!condition_0) {
        address_word_2 = matrix_fragment_25 * 0x20u + address_word_6;

        address_word_3 = address_word_2 + 0x20u;

        address_word_2 = uint32_t(uint32_t(address_word_2) >> 0x7u);

        address_word_6 = uint32_t(combine_words(address_word_3, 0u) << 0x19u >> 32) + 0x1u;

        address_word_6 = min(int32_t(matrix_fragment_51), int32_t(address_word_6));

        condition_1 = int32_t(address_word_2) >= int32_t(address_word_6);
        if (!condition_1) {
            address_word_7 = address_word_2;
            do {
                address_word_2 = parameters[parameter_wait_optional_low_12];
                address_word_3 = parameters[parameter_wait_optional_high_13];

                matrix_fragment_27 = address_word_7 * 0x4u + index_word_0;

                matrix_fragment_27 = matrix_fragment_27 * 0x8u;

                split_address_words(address_word_2, address_word_3,
                                    uint64_t(uint32_t(matrix_fragment_27)) *
                                            uint64_t(uint32_t(0x4u)) +
                                        combine_words(address_word_2, address_word_3));

                matrix_fragment_24 = read_device_flag<CheckBounds>(
                    combine_words(address_word_2, address_word_3), memory_ranges, error_counts);

                address_word_7 = address_word_7 + 0x1u;

                condition_2 = uint32_t(address_word_7) != uint32_t(address_word_6);

                condition_1 = int32_t(matrix_fragment_24) > -1;
                if (!condition_1) {
                    do {
                        __nanosleep(0x40u);

                        matrix_fragment_24 = read_device_flag<CheckBounds>(
                            combine_words(address_word_2, address_word_3), memory_ranges,
                            error_counts);

                        condition_1 = int32_t(matrix_fragment_24) > -1;
                    } while (!condition_1);
                }
            } while (condition_2);
        }
    }
    __syncwarp(__activemask());

    uniform_index_word_12 = 0u;

    matrix_fragment_48 = index_word_86 * 0x8u + matrix_fragment_25;

    uniform_index_word_9 = 0x400u;

    __syncthreads();

    matrix_fragment_49 = matrix_fragment_48 * 0x1000u;

    condition_2 = int32_t(matrix_fragment_48) >= int32_t(index_word_87);

    condition_1 = int32_t(matrix_fragment_48) >= int32_t(index_word_87);

    address_word_3 = index_word_0 * 0x400u + matrix_fragment_49;

    address_word_2 = !condition_1 ? address_word_3 : 0u;

    uniform_index_word_4 = (uniform_index_word_12 << 0x18u) + uniform_index_word_9;

    matrix_fragment_50 = (matrix_fragment_25 << 0x9u) + uniform_index_word_4;
    if (condition_2) {
        address_word_7 = threadIdx.x & 31u;

        uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = uniform_index_word_4;

        matrix_fragment_27 = uniform_index_word_4;

        matrix_fragment_28 = permute_signed(matrix_fragment_28, matrix_fragment_27, 0x5410u);

        matrix_fragment_29 = matrix_fragment_28;

        matrix_fragment_30 = matrix_fragment_28;

        matrix_fragment_31 = matrix_fragment_28;

        address_word_2 = address_word_7 * 0x10u + matrix_fragment_50;

        store_four_shared_words<CheckBounds>(
            shared_tile, address_word_2, matrix_fragment_28, matrix_fragment_29, matrix_fragment_30,
            matrix_fragment_31, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_index_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_index_word_4 = parameters[parameter_input_low_0];
            uniform_address_word_5 = parameters[parameter_input_high_1];

            uniform_address_word_7 = broadcast_active_lane(address_word_2);

            uniform_address_word_14 = broadcast_active_lane(matrix_fragment_50);

            uniform_address_word_6 = uniform_index_word_9 + 0x2000u;

            address_word_2 = 0x200u;

            uniform_address_word_15 = (uniform_index_word_12 << 0x18u) + uniform_address_word_6;

            uniform_address_word_6 = 0x20u;

            split_address_words(uniform_index_word_4, uniform_address_word_5,
                                int64_t(int32_t(uniform_address_word_7)) * int64_t(int32_t(0x4u)) +
                                    combine_words(uniform_index_word_4, uniform_address_word_5));

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_index_word_4, uniform_address_word_5),
                uniform_address_word_6 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_index_word_4, uniform_address_word_5),
                                   uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    matrix_fragment_64 = matrix_fragment_48 + 0x4u;

    address_word_2 = address_word_3 + 0x4000u;

    condition_2 = int32_t(matrix_fragment_64) >= int32_t(index_word_87);

    condition_1 = int32_t(matrix_fragment_64) >= int32_t(index_word_87);

    address_word_2 = !condition_1 ? address_word_2 : 0u;
    if (condition_2) {
        address_word_3 = threadIdx.x & 31u;

        uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));

        matrix_fragment_28 = uniform_index_word_4;

        address_word_7 = uniform_index_word_4;

        matrix_fragment_28 = permute_signed(matrix_fragment_28, address_word_7, 0x5410u);

        matrix_fragment_29 = matrix_fragment_28;

        matrix_fragment_30 = matrix_fragment_28;

        matrix_fragment_31 = matrix_fragment_28;

        address_word_3 = address_word_3 * 0x10u + matrix_fragment_50;

        store_four_shared_words<CheckBounds>(
            shared_tile, address_word_3 + 0x800u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);
    } else {
        uniform_index_word_4 = 0xffffffffu;

        condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_1) {
            uniform_index_word_4 = parameters[parameter_input_low_0];
            uniform_address_word_5 = parameters[parameter_input_high_1];

            uniform_address_word_6 = broadcast_active_lane(address_word_2);

            uniform_address_word_8 = broadcast_active_lane(matrix_fragment_50);

            uniform_address_word_7 = uniform_index_word_9 + 0x2000u;

            address_word_2 = 0x200u;

            uniform_address_word_7 = (uniform_index_word_12 << 0x18u) + uniform_address_word_7;

            split_address_words(uniform_index_word_4, uniform_address_word_5,
                                int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                                    combine_words(uniform_index_word_4, uniform_address_word_5));

            uniform_address_word_6 = uniform_address_word_8 + 0x800u;

            uniform_address_word_8 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_6,
                combine_words(uniform_index_word_4, uniform_address_word_5),
                uniform_address_word_8 * 16u, uniform_address_word_7, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                   combine_words(uniform_index_word_4, uniform_address_word_5),
                                   uniform_address_word_8 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    address_word_7 = 0u;

    uniform_index_word_4 = uniform_index_word_9 + 0x2000u;

    matrix_fragment_27 = index_word_86 * 0x80u;

    condition_2 = uint32_t(index_word_0) != 0u;

    thread_row_100 = threadIdx.y;

    address_word_2 = uniform_index_word_4;

    address_word_7 = (address_word_7 << 0x18u) + address_word_2;

#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = address_word_7;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(address_word_2, address_word_3, async_tokens[operand_a >> 3 & 1]);
    }
#else
    address_word_2 = 0u;
#endif

    matrix_fragment_66 = thread_row_100 * 0x20u + matrix_fragment_27;

    address_word_6 = matrix_fragment_66 + 0x20u;

    address_word_6 = uint32_t(combine_words(address_word_6, 0u) << 0x19u >> 32) + 0x1u;

    matrix_fragment_51 = min(int32_t(matrix_fragment_51), int32_t(address_word_6));

#if DLSS5_ENABLE_ASYNC_COPIES
    condition_1 =
        wait_copy_barrier(shared_tile, address_word_7, async_tokens[(((address_word_7) >> 3) & 1)]);
#else
    __syncthreads();
    condition_1 = true;
#endif
    if (!condition_1) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_1 = wait_copy_barrier(shared_tile, address_word_7,
                                            async_tokens[(((address_word_7) >> 3) & 1)]);
#else
            __syncthreads();
            condition_1 = true;
#endif
        } while (!condition_1);
    }
    matrix_fragment_66 = uint32_t(uint32_t(matrix_fragment_66) >> 0x7u);

    address_word_2 = 0;
    address_word_3 = 0;

    address_word_6 = 0;
    address_word_7 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    matrix_accumulator_88 = 0;
    matrix_accumulator_89 = 0;

    matrix_accumulator_92 = 0;
    matrix_accumulator_93 = 0;

    matrix_accumulator_94 = 0;
    matrix_accumulator_95 = 0;

    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_76 = 0;
    matrix_accumulator_77 = 0;

    matrix_accumulator_74 = 0;
    matrix_accumulator_75 = 0;

    matrix_accumulator_72 = 0;
    matrix_accumulator_73 = 0;

    matrix_accumulator_70 = 0;
    matrix_accumulator_71 = 0;

    matrix_accumulator_68 = 0;
    matrix_accumulator_69 = 0;

    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_accumulator_42 = 0;
    matrix_accumulator_43 = 0;

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_40 = 0;
    matrix_accumulator_41 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    condition_0 = int32_t(matrix_fragment_66) >= int32_t(matrix_fragment_51) || condition_0;

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    index_word_91 = thread_row_100 * 0x2u;

    matrix_fragment_65 = thread_row_100 * 0x400u;
    if (!condition_2) {
        matrix_fragment_25 = matrix_fragment_25 * 0x2u;

        matrix_accumulator_69 = threadIdx.x & 31u;

        address_word_2 = parameters[parameter_input_skip_optional_low_2];
        address_word_3 = parameters[parameter_input_skip_optional_high_3];

        matrix_fragment_25 = matrix_fragment_25 & 0x7fcu;

        matrix_fragment_26 = index_word_86 * 0x8u + matrix_fragment_25;

        matrix_accumulator_63 = matrix_fragment_26 * 0x400u + index_word_82;

        condition_3 = int32_t(matrix_fragment_26) >= int32_t(index_word_87);

        address_word_6 = matrix_fragment_26 + 0x1u;

        matrix_accumulator_63 = matrix_accumulator_63 * 0x4u;

        condition_1 = int32_t(address_word_6) >= int32_t(index_word_87);

        address_word_6 = matrix_fragment_26 + 0x2u;

        split_address_words(matrix_accumulator_62, matrix_accumulator_63,
                            int64_t(int32_t(matrix_accumulator_63)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        condition_2 = int32_t(address_word_6) >= int32_t(index_word_87);

        if (condition_3) {
            matrix_accumulator_59 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));

            matrix_accumulator_55 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));

            matrix_accumulator_59 =
                permute_signed(matrix_accumulator_59, matrix_accumulator_59, 0x5410u);
        }

        if (!condition_3) {
            split_address_words(address_word_2, address_word_3,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        if (condition_3) {
            matrix_accumulator_55 =
                permute_signed(matrix_accumulator_55, matrix_accumulator_55, 0x5410u);
        }

        if (condition_1) {
            matrix_accumulator_47 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (condition_3) {
            matrix_accumulator_56 = matrix_accumulator_59;

            matrix_accumulator_57 = matrix_accumulator_59;

            matrix_accumulator_58 = matrix_accumulator_59;

            matrix_accumulator_52 = matrix_accumulator_55;

            matrix_accumulator_53 = matrix_accumulator_55;

            matrix_accumulator_54 = matrix_accumulator_55;
        }

        if (!condition_3) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));

            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_2, address_word_3), matrix_accumulator_56,
                matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
                error_counts);
        }

        if (condition_1) {
            matrix_accumulator_47 =
                permute_signed(matrix_accumulator_47, matrix_accumulator_47, 0x5410u);
        }

        if (!condition_1) {
            split_address_words(matrix_fragment_24, matrix_fragment_25,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        if (!condition_3) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x200ull, matrix_accumulator_52,
                matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
                error_counts);
        }

        if (condition_1) {
            matrix_accumulator_44 = matrix_accumulator_47;

            matrix_accumulator_45 = matrix_accumulator_47;

            matrix_accumulator_46 = matrix_accumulator_47;
        }

        address_word_2 = matrix_fragment_26 + 0x3u;

        if (condition_2) {
            matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
        }

        address_word_6 = matrix_accumulator_69;

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25) + 0x4000ull,
                matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_46,
                matrix_accumulator_47, memory_ranges, error_counts);
        }

        condition_3 = int32_t(address_word_2) >= int32_t(index_word_87);

        address_word_2 = parameters[parameter_trained_weights_low_6];
        address_word_3 = parameters[parameter_trained_weights_high_7];

        if (condition_1) {
            matrix_accumulator_43 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        if (condition_2) {
            matrix_fragment_39 = permute_signed(matrix_fragment_39, matrix_fragment_39, 0x5410u);
        }

        if (condition_1) {
            matrix_accumulator_43 =
                permute_signed(matrix_accumulator_43, matrix_accumulator_43, 0x5410u);
        }

        matrix_fragment_24 = address_word_6 & 0x7ffffffcu;

        matrix_accumulator_60 = index_word_82 + 0x10u;

        if (condition_3) {
            matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
        }

        matrix_fragment_67 = 0u - matrix_fragment_24 + matrix_accumulator_69;

        if (condition_2) {
            matrix_fragment_36 = matrix_fragment_39;

            matrix_fragment_37 = matrix_fragment_39;

            matrix_fragment_38 = matrix_fragment_39;
        }

        if (!condition_2) {
            split_address_words(matrix_fragment_28, matrix_fragment_29,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        if (condition_1) {
            matrix_accumulator_40 = matrix_accumulator_43;

            matrix_accumulator_41 = matrix_accumulator_43;

            matrix_accumulator_42 = matrix_accumulator_43;
        }

        matrix_accumulator_68 = index_word_82 + 0x18u;

        if (!condition_1) {
            split_address_words(matrix_fragment_26, matrix_fragment_27,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        if (condition_3) {
            matrix_fragment_31 = permute_signed(matrix_fragment_31, matrix_fragment_31, 0x5410u);
        }

        if (!condition_2) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_fragment_28, matrix_fragment_29) + 0x8000ull,
                matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                memory_ranges, error_counts);
        }

        if (condition_2) {
            matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
        }

        matrix_accumulator_60 = matrix_fragment_67 * 0x2u + matrix_accumulator_60;

        matrix_accumulator_68 = matrix_fragment_67 * 0x2u + matrix_accumulator_68;

        if (!condition_1) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_fragment_26, matrix_fragment_27) + 0x4200ull,
                matrix_accumulator_40, matrix_accumulator_41, matrix_accumulator_42,
                matrix_accumulator_43, memory_ranges, error_counts);
        }

        if (condition_2) {
            matrix_fragment_35 = permute_signed(matrix_fragment_35, matrix_fragment_35, 0x5410u);
        }

        if (condition_3) {
            matrix_fragment_30 = matrix_fragment_31;

            matrix_fragment_28 = matrix_fragment_31;

            matrix_fragment_29 = matrix_fragment_31;
        }

        matrix_fragment_25 = matrix_fragment_67 * 0x2u + index_word_82;

        matrix_accumulator_61 = uint32_t(int32_t(matrix_accumulator_60) >> 0x1u);

        if (!condition_3) {
            split_address_words(matrix_fragment_26, matrix_fragment_27,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        matrix_accumulator_68 = uint32_t(combine_words(matrix_accumulator_68, 0u) << 0x1u >> 32) +
                                matrix_accumulator_68;

        if (condition_2) {
            matrix_fragment_32 = matrix_fragment_35;

            matrix_fragment_33 = matrix_fragment_35;

            matrix_fragment_34 = matrix_fragment_35;
        }

        if (!condition_2) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        if (!condition_3) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(matrix_fragment_26, matrix_fragment_27) + 0xc000ull,
                matrix_fragment_28, matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                memory_ranges, error_counts);
        }

        matrix_fragment_24 = index_word_82 + 0x8u;

        matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_25) >> 0x1u);

        if (!condition_2) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8200ull, matrix_fragment_32,
                matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
                error_counts);
        }

        split_address_words(matrix_fragment_26, matrix_fragment_27,
                            int64_t(int32_t(matrix_accumulator_61)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_61 = uint32_t(int32_t(matrix_accumulator_68) >> 0x1u);

        matrix_accumulator_68 = index_word_82 + 0x20u;

        matrix_fragment_24 = matrix_fragment_67 * 0x2u + matrix_fragment_24;

        if (condition_3) {
            matrix_accumulator_70 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
        }

        split_address_words(address_word_6, address_word_7,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_93 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_26, matrix_fragment_27) + 0x100000ull, memory_ranges,
            error_counts);

        matrix_accumulator_68 = matrix_fragment_67 * 0x2u + matrix_accumulator_68;

        matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_24) >> 0x1u);

        matrix_accumulator_97 = load_aligned_word<CheckBounds>(
            combine_words(address_word_6, address_word_7) + 0x100000ull, memory_ranges,
            error_counts);

        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        if (condition_3) {
            matrix_fragment_27 =
                permute_signed(matrix_accumulator_70, matrix_accumulator_70, 0x5410u);
        }

        if (!condition_3) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(matrix_accumulator_69)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_accumulator_62, matrix_accumulator_63));
        }

        matrix_accumulator_63 = uint32_t(int32_t(matrix_accumulator_68) >> 0x1u);

        matrix_accumulator_95 = load_aligned_word<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25) + 0x100000ull, memory_ranges,
            error_counts);

        matrix_accumulator_68 = index_word_82 + 0x30u;

        matrix_accumulator_62 = index_word_82 + 0x28u;

        matrix_accumulator_70 = index_word_82 + 0x38u;

        matrix_accumulator_68 = matrix_fragment_67 * 0x2u + matrix_accumulator_68;

        matrix_accumulator_62 = matrix_fragment_67 * 0x2u + matrix_accumulator_62;

        if (condition_3) {
            matrix_fragment_26 = matrix_fragment_27;
        }

        matrix_accumulator_70 = matrix_fragment_67 * 0x2u + matrix_accumulator_70;

        if (condition_3) {
            matrix_fragment_24 = matrix_fragment_27;
        }

        split_address_words(matrix_accumulator_60, matrix_accumulator_61,
                            int64_t(int32_t(matrix_accumulator_61)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        if (condition_3) {
            matrix_fragment_25 = matrix_fragment_27;
        }

        matrix_fragment_67 = uint32_t(int32_t(matrix_accumulator_68) >> 0x1u);

        matrix_accumulator_62 = uint32_t(combine_words(matrix_accumulator_62, 0u) << 0x1u >> 32) +
                                matrix_accumulator_62;

        matrix_accumulator_89 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_60, matrix_accumulator_61) + 0x100000ull,
            memory_ranges, error_counts);

        matrix_accumulator_70 = uint32_t(combine_words(matrix_accumulator_70, 0u) << 0x1u >> 32) +
                                matrix_accumulator_70;

        if (!condition_3) {
            load_four_words_global_cached<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0xc200ull, matrix_fragment_24,
                matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
                error_counts);
        }

        split_address_words(matrix_accumulator_60, matrix_accumulator_61,
                            int64_t(int32_t(matrix_accumulator_63)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_63 = uint32_t(int32_t(matrix_accumulator_62) >> 0x1u);

        split_address_words(address_word_6, address_word_7,
                            int64_t(int32_t(matrix_fragment_67)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_fragment_67 = uint32_t(int32_t(matrix_accumulator_70) >> 0x1u);

        matrix_accumulator_85 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_60, matrix_accumulator_61) + 0x100000ull,
            memory_ranges, error_counts);

        split_address_words(matrix_accumulator_62, matrix_accumulator_63,
                            int64_t(int32_t(matrix_accumulator_63)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        address_word_7 = load_aligned_word<CheckBounds>(
            combine_words(address_word_6, address_word_7) + 0x100000ull, memory_ranges,
            error_counts);

        split_address_words(address_word_2, address_word_3,
                            int64_t(int32_t(matrix_fragment_67)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, address_word_3));

        matrix_accumulator_81 = load_aligned_word<CheckBounds>(
            combine_words(matrix_accumulator_62, matrix_accumulator_63) + 0x100000ull,
            memory_ranges, error_counts);

        address_word_3 = load_aligned_word<CheckBounds>(
            combine_words(address_word_2, address_word_3) + 0x100000ull, memory_ranges,
            error_counts);

        matrix_accumulator_114 = unpack_fp8_bits(matrix_accumulator_56);

        matrix_accumulator_115 = unpack_fp8_bits(matrix_accumulator_57);

        matrix_accumulator_112 = unpack_fp8_bits(matrix_accumulator_56 >> 16);

        matrix_accumulator_113 = unpack_fp8_bits(matrix_accumulator_57 >> 16);

        matrix_accumulator_108 = unpack_fp8_bits(matrix_accumulator_58);

        matrix_accumulator_56 = unpack_fp8_bits(matrix_accumulator_58 >> 16);

        matrix_accumulator_109 = unpack_fp8_bits(matrix_accumulator_59);

        matrix_accumulator_106 = unpack_fp8_bits(matrix_accumulator_44);

        matrix_accumulator_107 = unpack_fp8_bits(matrix_accumulator_45);

        matrix_accumulator_104 = unpack_fp8_bits(matrix_accumulator_44 >> 16);

        matrix_accumulator_105 = unpack_fp8_bits(matrix_accumulator_45 >> 16);

        matrix_accumulator_57 = unpack_fp8_bits(matrix_accumulator_59 >> 16);

        matrix_accumulator_110 = unpack_fp8_bits(matrix_accumulator_52);

        matrix_accumulator_59 = unpack_fp8_bits(matrix_accumulator_52 >> 16);

        matrix_accumulator_111 = unpack_fp8_bits(matrix_accumulator_53);

        matrix_accumulator_58 = unpack_fp8_bits(matrix_accumulator_53 >> 16);

        matrix_accumulator_60 = unpack_fp8_bits(matrix_accumulator_54);

        matrix_accumulator_62 = unpack_fp8_bits(matrix_accumulator_54 >> 16);

        address_word_6 = unpack_fp8_bits(matrix_accumulator_55);

        matrix_accumulator_102 = unpack_fp8_bits(matrix_fragment_36);

        matrix_accumulator_103 = unpack_fp8_bits(matrix_fragment_37);

        matrix_accumulator_98 = unpack_fp8_bits(matrix_fragment_36 >> 16);

        matrix_accumulator_99 = unpack_fp8_bits(matrix_fragment_37 >> 16);

        address_word_2 = unpack_fp8_bits(matrix_accumulator_55 >> 16);

        matrix_accumulator_44 = unpack_fp8_bits(matrix_accumulator_40);

        matrix_accumulator_52 = unpack_fp8_bits(matrix_accumulator_46);

        matrix_accumulator_54 = unpack_fp8_bits(matrix_accumulator_46 >> 16);

        matrix_accumulator_53 = unpack_fp8_bits(matrix_accumulator_47);

        matrix_accumulator_96 = unpack_fp8_bits(matrix_fragment_28);

        matrix_accumulator_94 = unpack_fp8_bits(matrix_fragment_28 >> 16);

        matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_29);

        matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_29 >> 16);

        matrix_accumulator_55 = unpack_fp8_bits(matrix_accumulator_47 >> 16);

        matrix_accumulator_45 = unpack_fp8_bits(matrix_accumulator_40 >> 16);

        matrix_accumulator_47 = unpack_fp8_bits(matrix_accumulator_41);

        matrix_accumulator_46 = unpack_fp8_bits(matrix_accumulator_42);

        matrix_accumulator_74 = unpack_fp8_bits(matrix_accumulator_41 >> 16);

        matrix_accumulator_72 = unpack_fp8_bits(matrix_fragment_32);

        matrix_accumulator_92 = unpack_fp8_bits(matrix_fragment_30);

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_115 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_103 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_88 = unpack_fp8_bits(matrix_fragment_30 >> 16);

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_112 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_113 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_104 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_105 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_fragment_67 = unpack_fp8_bits(matrix_accumulator_42 >> 16);

        matrix_accumulator_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                select_half_operand<2, false, false>(matrix_accumulator_95));

        matrix_accumulator_61 = unpack_fp8_bits(matrix_accumulator_43);

        matrix_accumulator_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_84 = unpack_fp8_bits(matrix_fragment_24);

        matrix_accumulator_80 = unpack_fp8_bits(matrix_fragment_24 >> 16);

        matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_26);

        matrix_accumulator_63 = unpack_fp8_bits(matrix_accumulator_43 >> 16);

        matrix_accumulator_68 = unpack_fp8_bits(matrix_fragment_38);

        matrix_accumulator_70 = unpack_fp8_bits(matrix_fragment_38 >> 16);

        matrix_accumulator_69 = unpack_fp8_bits(matrix_fragment_39);

        matrix_accumulator_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_71 = unpack_fp8_bits(matrix_fragment_39 >> 16);

        matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

        matrix_accumulator_73 = unpack_fp8_bits(matrix_fragment_33);

        matrix_accumulator_75 = unpack_fp8_bits(matrix_fragment_33 >> 16);

        matrix_accumulator_76 = unpack_fp8_bits(matrix_fragment_34);

        matrix_accumulator_78 = unpack_fp8_bits(matrix_fragment_34 >> 16);

        matrix_accumulator_77 = unpack_fp8_bits(matrix_fragment_35);

        matrix_accumulator_79 = unpack_fp8_bits(matrix_fragment_35 >> 16);

        matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_31);

        matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_25);

        matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_26 >> 16);

        matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_27);

        matrix_accumulator_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_31 >> 16);

        matrix_accumulator_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_25 >> 16);

        matrix_accumulator_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_27 >> 16);

        matrix_accumulator_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_6),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(address_word_2),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_108 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_109 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_92 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_110 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_111 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_72 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_73 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_74 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_75 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_80 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_76 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_77 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                select_half_operand<2, false, false>(address_word_7));

        address_word_6 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                select_half_operand<2, false, false>(address_word_7));

        matrix_accumulator_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_78 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_79 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                                select_half_operand<2, false, false>(address_word_3));

        address_word_2 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                select_half_operand<2, false, false>(address_word_3));

        matrix_accumulator_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_accumulator_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                select_half_operand<2, false, false>(matrix_accumulator_89));

        matrix_accumulator_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                select_half_operand<2, false, false>(matrix_accumulator_85));

        matrix_accumulator_81 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        address_word_7 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                select_half_operand<2, false, false>(address_word_7));

        address_word_3 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                select_half_operand<2, false, false>(address_word_3));
    }
    matrix_fragment_67 = 0u;

    index_word_91 = index_word_91 & 0x7fcu;

    matrix_fragment_65 = matrix_fragment_65 & 0xff800u;
    do {
        uniform_address_word_5 = 0u;

        warp_lane_101 = threadIdx.x & 31u;

        matrix_fragment_24 = matrix_fragment_67 * 0x80u;

        uniform_index_word_4 = 0x400u;

        address_word_90 = uint32_t(uint32_t(matrix_fragment_67) >> 0x5u);

        packed_half_116 = matrix_fragment_67;

        matrix_fragment_24 = matrix_fragment_24 & 0x1000u;

        matrix_fragment_67 = matrix_fragment_67 + 0x20u;

        address_word_90 = address_word_90 & 0x1u;

        uniform_index_word_4 = (uniform_address_word_5 << 0x18u) + uniform_index_word_4;

        matrix_fragment_24 = matrix_fragment_24 + uniform_index_word_4 + matrix_fragment_65;

        __syncwarp(__activemask());

        warp_lane_101 = warp_lane_101 * 0x10u + matrix_fragment_24;

        load_four_shared_words<CheckBounds>(
            shared_tile, warp_lane_101, matrix_fragment_24, matrix_fragment_25, matrix_fragment_26,
            matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, warp_lane_101 + 0x200u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, warp_lane_101 + 0x400u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, warp_lane_101 + 0x600u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_114,
                                 matrix_accumulator_115);

        condition_2 = uint32_t(packed_half_116) >= uint32_t(0xc0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_40,
                                 matrix_accumulator_41);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_42,
                                 matrix_accumulator_43);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_68,
                                 matrix_accumulator_69);

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_70,
                                 matrix_accumulator_71);

        multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_72,
                                 matrix_accumulator_73);

        multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_74,
                                 matrix_accumulator_75);

        multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_76,
                                 matrix_accumulator_77);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_88,
                                 matrix_accumulator_89);

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_84,
                                 matrix_accumulator_85);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_80,
                                 matrix_accumulator_81);

        matrix_fragment_14 = index_word_0 * 0x100u + matrix_fragment_67;

        multiply_accumulate_half(address_word_6, address_word_7, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_8, matrix_fragment_9, address_word_6,
                                 address_word_7);

        multiply_accumulate_half(address_word_2, address_word_3, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_10, matrix_fragment_11, address_word_2,
                                 address_word_3);
        if (!condition_0) {
            matrix_fragment_13 = uint32_t(uint32_t(matrix_fragment_14) >> 0x5u);

            matrix_fragment_10 = matrix_fragment_66;
            do {
                matrix_fragment_8 = parameters[parameter_wait_optional_low_12];
                matrix_fragment_9 = parameters[parameter_wait_optional_high_13];

                matrix_fragment_11 = matrix_fragment_10 * 0x20u + matrix_fragment_13;

                split_address_words(matrix_fragment_8, matrix_fragment_9,
                                    uint64_t(uint32_t(matrix_fragment_11)) *
                                            uint64_t(uint32_t(0x4u)) +
                                        combine_words(matrix_fragment_8, matrix_fragment_9));

                matrix_fragment_11 = read_device_flag<CheckBounds>(
                    combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges,
                    error_counts);

                matrix_fragment_10 = matrix_fragment_10 + 0x1u;

                condition_3 = uint32_t(matrix_fragment_10) != uint32_t(matrix_fragment_51);

                condition_1 = int32_t(matrix_fragment_11) > -1;
                if (!condition_1) {
                    do {
                        __nanosleep(0x40u);

                        matrix_fragment_11 = read_device_flag<CheckBounds>(
                            combine_words(matrix_fragment_8, matrix_fragment_9), memory_ranges,
                            error_counts);

                        condition_1 = int32_t(matrix_fragment_11) > -1;
                    } while (!condition_1);
                }
            } while (condition_3);
        }
        __syncwarp(__activemask());

        condition_1 = int32_t(matrix_fragment_48) >= int32_t(index_word_87);

        condition_3 = int32_t(matrix_fragment_48) >= int32_t(index_word_87);

        matrix_fragment_31 = 0u;

        matrix_fragment_28 = 0x400u;

        matrix_fragment_8 = address_word_90 * 0x8u;

        matrix_fragment_13 = matrix_fragment_14 * 0x4u + matrix_fragment_49;

        __syncthreads();

        matrix_fragment_9 = matrix_fragment_28 + 0x2000u;

        matrix_fragment_29 = 0x8u ^ matrix_fragment_8;

        matrix_fragment_8 = address_word_90 * 0x1000u;

        uniform_address_word_8 = broadcast_active_lane(matrix_fragment_9);

        uniform_index_word_4 = broadcast_active_lane(matrix_fragment_29);

        matrix_fragment_9 = 0x1000u ^ matrix_fragment_8;

        matrix_fragment_8 = !condition_3 ? matrix_fragment_13 : 0u;

        matrix_fragment_15 = matrix_fragment_50 + matrix_fragment_9;

        uniform_address_word_5 = broadcast_active_lane(matrix_fragment_31);

        uniform_address_word_8 = (uniform_address_word_5 << 0x18u) + uniform_address_word_8;

        uniform_address_word_15 = uniform_address_word_8 + uniform_index_word_4;
        if (condition_1) {
            matrix_fragment_12 = threadIdx.x & 31u;

            uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                 select_half_operand<2, false, false>(0u));

            matrix_fragment_8 = uniform_index_word_4;

            matrix_fragment_9 = uniform_index_word_4;

            matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

            matrix_fragment_9 = matrix_fragment_8;

            matrix_fragment_10 = matrix_fragment_8;

            matrix_fragment_11 = matrix_fragment_8;

            matrix_fragment_12 = matrix_fragment_12 * 0x10u + matrix_fragment_15;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_12, matrix_fragment_8, matrix_fragment_9,
                matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_index_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_index_word_4 = parameters[parameter_input_low_0];
                uniform_address_word_5 = parameters[parameter_input_high_1];

                uniform_address_word_6 = broadcast_active_lane(matrix_fragment_8);

                uniform_address_word_14 = broadcast_active_lane(matrix_fragment_15);

                matrix_fragment_8 = 0x200u;

                split_address_words(
                    uniform_index_word_4, uniform_address_word_5,
                    int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                        combine_words(uniform_index_word_4, uniform_address_word_5));

                uniform_address_word_6 = 0x20u;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_14,
                    combine_words(uniform_index_word_4, uniform_address_word_5),
                    uniform_address_word_6 * 16u, matrix_fragment_29 + uniform_address_word_8,
                    shared_bytes, memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                       combine_words(uniform_index_word_4, uniform_address_word_5),
                                       uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                                       error_counts);
#endif
            }
        }
        condition_1 = int32_t(matrix_fragment_64) >= int32_t(index_word_87);

        condition_3 = int32_t(matrix_fragment_64) >= int32_t(index_word_87);

        matrix_fragment_8 = matrix_fragment_13 + 0x4000u;

        matrix_fragment_8 = !condition_3 ? matrix_fragment_8 : 0u;
        if (condition_1) {
            matrix_fragment_8 = threadIdx.x & 31u;

            uniform_index_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                 select_half_operand<2, false, false>(0u));

            matrix_fragment_9 = uniform_index_word_4;

            matrix_fragment_10 = uniform_index_word_4;

            matrix_fragment_15 = matrix_fragment_8 * 0x10u + matrix_fragment_15;

            matrix_fragment_8 = permute_signed(matrix_fragment_9, matrix_fragment_10, 0x5410u);

            matrix_fragment_9 = matrix_fragment_8;

            matrix_fragment_10 = matrix_fragment_8;

            matrix_fragment_11 = matrix_fragment_8;

            store_four_shared_words<CheckBounds>(
                shared_tile, matrix_fragment_15 + 0x800u, matrix_fragment_8, matrix_fragment_9,
                matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);
        } else {
            uniform_index_word_4 = 0xffffffffu;

            condition_1 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_1) {
                uniform_index_word_4 = parameters[parameter_input_low_0];
                uniform_address_word_5 = parameters[parameter_input_high_1];

                uniform_address_word_6 = broadcast_active_lane(matrix_fragment_8);

                uniform_address_word_7 = broadcast_active_lane(matrix_fragment_15);

                uniform_address_word_13 = 0x20u;

                matrix_fragment_8 = 0x200u;

                split_address_words(
                    uniform_index_word_4, uniform_address_word_5,
                    int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                        combine_words(uniform_index_word_4, uniform_address_word_5));

                uniform_address_word_6 = uniform_address_word_7 + 0x800u;

                uniform_address_word_7 = uniform_address_word_15;

#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_index_word_4, uniform_address_word_5),
                    uniform_address_word_13 * 16u, matrix_fragment_29 + uniform_address_word_8,
                    shared_bytes, memory_ranges, error_counts);
#else
                bulk_copy<CheckBounds>(shared_tile, uniform_address_word_6,
                                       combine_words(uniform_index_word_4, uniform_address_word_5),
                                       uniform_address_word_13 * 16u, shared_bytes, memory_ranges,
                                       error_counts);
#endif
            }
        }
        address_word_90 = threadIdx.x & 31u;

        matrix_fragment_8 = parameters[parameter_trained_weights_low_6];
        matrix_fragment_9 = parameters[parameter_trained_weights_high_7];

        matrix_fragment_11 = matrix_fragment_14 * 0x100u + address_word_83;

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_24, matrix_fragment_25),
                                            matrix_fragment_20, matrix_fragment_21,
                                            matrix_fragment_22, matrix_fragment_23, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25) + 0x200ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25) + 0x400ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25) + 0x600ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = matrix_fragment_29 + uniform_address_word_8;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_26, matrix_fragment_27,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_26 = 0u;
#endif

#if DLSS5_ENABLE_ASYNC_COPIES
        condition_1 = wait_copy_barrier(
            shared_tile, matrix_fragment_29 + uniform_address_word_8,
            async_tokens[(((matrix_fragment_29 + uniform_address_word_8) >> 3) & 1)]);
#else
        __syncthreads();
        condition_1 = true;
#endif
        if (!condition_1) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_1 = wait_copy_barrier(
                    shared_tile, matrix_fragment_29 + uniform_address_word_8,
                    async_tokens[(((matrix_fragment_29 + uniform_address_word_8) >> 3) & 1)]);
#else
                __syncthreads();
                condition_1 = true;
#endif
            } while (!condition_1);
        }
    } while (!condition_2);
    matrix_fragment_28 = (matrix_fragment_31 << 0x18u) + matrix_fragment_28;

    matrix_fragment_33 = threadIdx.x;

    __syncwarp(__activemask());

    matrix_fragment_65 = matrix_fragment_28 + matrix_fragment_65;

    condition_0 = uint32_t(index_word_0) != 0u;

    matrix_fragment_32 = address_word_90 * 0x10u + matrix_fragment_65;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_32 + 0x1000u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_32 + 0x1200u, matrix_fragment_64, matrix_fragment_65,
        matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_32 + 0x1400u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_32 + 0x1600u, matrix_fragment_28, matrix_fragment_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    thread_row_100 = matrix_fragment_33 | thread_row_100;

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_44,
                             matrix_accumulator_45);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_8, matrix_fragment_9, address_word_6, address_word_7);

    address_word_6 = parameters[parameter_reduction_counter_optional_low_8];
    address_word_7 = parameters[parameter_reduction_counter_optional_high_9];

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_40,
                             matrix_accumulator_41);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_40, matrix_accumulator_41, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_42, matrix_accumulator_43, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_42,
                             matrix_accumulator_43);

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_46,
                             matrix_accumulator_47);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_104,
                             matrix_accumulator_105);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_10, matrix_fragment_11, address_word_2,
                             address_word_3);

    address_word_3 = index_word_86 * 0x8u + address_word_5;

    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_6, address_word_7));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_70,
                             matrix_accumulator_71);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_80,
                             matrix_accumulator_81);
    if (!condition_0) {
        address_word_6 = parameters[parameter_reduction_scratch_optional_low_10];
        address_word_7 = parameters[parameter_reduction_scratch_optional_high_11];

        index_word_86 = index_word_86 * 0x8u + index_word_91;

        address_word_83 = index_word_86 * 0x2000u + address_word_83;

        condition_0 = int32_t(index_word_86) >= int32_t(index_word_87);

        address_word_4 = index_word_86 + 0x1u;

        matrix_fragment_24 = index_word_86 + 0x2u;

        matrix_fragment_26 = index_word_86 + 0x3u;

        condition_1 = int32_t(address_word_4) >= int32_t(index_word_87);

        condition_2 = int32_t(matrix_fragment_24) >= int32_t(index_word_87);

        condition_3 = int32_t(matrix_fragment_26) >= int32_t(index_word_87);

        split_address_words(address_word_6, address_word_7,
                            int64_t(int32_t(address_word_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_6, address_word_7));
        if (!condition_0) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_fragment_32, matrix_fragment_33,
                                                    matrix_fragment_34, matrix_fragment_35,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x200ull, matrix_fragment_36,
                matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x400ull, matrix_accumulator_40,
                matrix_accumulator_41, matrix_accumulator_42, matrix_accumulator_43, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_4, address_word_5) + 0x600ull, matrix_accumulator_44,
                matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47, memory_ranges,
                error_counts);
        }
        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            combine_words(address_word_6, address_word_7) + 0x8000u);
        if (!condition_1) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_24, matrix_fragment_25));

            split_address_words(matrix_fragment_26, matrix_fragment_27,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_fragment_48, matrix_fragment_49,
                                                    matrix_fragment_50, matrix_fragment_51,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_26, matrix_fragment_27) + 0x8200ull,
                matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_54,
                matrix_accumulator_55, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_26, matrix_fragment_27) + 0x8400ull,
                matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58,
                matrix_accumulator_59, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_26, matrix_fragment_27) + 0x8600ull,
                matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_62,
                matrix_accumulator_63, memory_ranges, error_counts);
        }
        split_address_words(address_word_6, address_word_7,
                            combine_words(address_word_6, address_word_7) + 0x10000u);
        if (!condition_2) {
            split_address_words(address_word_4, address_word_5,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            split_address_words(matrix_fragment_24, matrix_fragment_25,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_24, matrix_fragment_25));

            store_four_words_streaming<CheckBounds>(combine_words(address_word_4, address_word_5),
                                                    matrix_fragment_64, matrix_fragment_65,
                                                    matrix_fragment_66, matrix_fragment_67,
                                                    memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25) + 0x8200ull,
                matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_70,
                matrix_accumulator_71, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25) + 0x8400ull,
                matrix_accumulator_72, matrix_accumulator_73, matrix_accumulator_74,
                matrix_accumulator_75, memory_ranges, error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(matrix_fragment_24, matrix_fragment_25) + 0x8600ull,
                matrix_accumulator_76, matrix_accumulator_77, matrix_accumulator_78,
                matrix_accumulator_79, memory_ranges, error_counts);
        }
        if (!condition_3) {
            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_6, address_word_7));

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8000ull, matrix_fragment_20,
                matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8200ull, matrix_fragment_16,
                matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8400ull, matrix_fragment_12,
                matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
                error_counts);

            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_6, address_word_7) + 0x8600ull, matrix_fragment_8,
                matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
                error_counts);
        }
    } else {
        address_word_5 = threadIdx.z;

        condition_0 = (address_word_5 | thread_row_100) != 0;
        if (!condition_0) {
            address_word_6 = read_device_flag<CheckBounds>(
                combine_words(address_word_2, address_word_3), memory_ranges, error_counts);

            address_word_5 = index_word_0 + 0xffffffffu;

            condition_0 = int32_t(address_word_6) >= int32_t(address_word_5);
            if (!condition_0) {
                do {
                    __nanosleep(0x40u);

                    address_word_6 = read_device_flag<CheckBounds>(
                        combine_words(address_word_2, address_word_3), memory_ranges, error_counts);

                    condition_0 = int32_t(address_word_6) >= int32_t(address_word_5);
                } while (!condition_0);
            }
        }
        condition_0 = uint32_t(index_word_0) >= uint32_t(0x3u);

        __syncwarp(__activemask());

        __syncthreads();
        if (!condition_0) {
            address_word_4 = index_word_86 * 0x8u + index_word_91;

            condition_3 = int32_t(address_word_4) >= int32_t(index_word_87);

            index_word_82 = address_word_4 + 0x1u;

            matrix_accumulator_84 = address_word_4 + 0x2u;

            index_word_86 = address_word_4 + 0x3u;

            condition_0 = int32_t(index_word_82) >= int32_t(index_word_87);

            condition_1 = int32_t(matrix_accumulator_84) >= int32_t(index_word_87);

            condition_2 = int32_t(index_word_86) >= int32_t(index_word_87);
            if (!condition_3) {
                uniform_index_word_4 = parameters[parameter_reduction_scratch_optional_low_10];
                uniform_address_word_5 = parameters[parameter_reduction_scratch_optional_high_11];

                address_word_4 = address_word_4 * 0x2000u + address_word_83;

                address_word_6 = address_word_4 + 0x80u;

                address_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

                matrix_fragment_24 = address_word_4 + 0x100u;

                matrix_fragment_26 = address_word_4 + 0x180u;

                address_word_7 = uint32_t(int32_t(address_word_6) >> 0x1fu);

                split_address_words(address_word_4, address_word_5,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_4, address_word_5));

                matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_24) >> 0x1fu);

                matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_26) >> 0x1fu);

                split_address_words(address_word_6, address_word_7,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_6, address_word_7));

                const uint64_t byte_address_0 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_4, address_word_5));
                matrix_fragment_28 = uint32_t(byte_address_0);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_24, matrix_fragment_25));

                matrix_fragment_29 = uint32_t(byte_address_0 >> 32);

                const uint64_t byte_address_1 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_6, address_word_7));
                address_word_4 = uint32_t(byte_address_1);

                split_address_words(matrix_fragment_26, matrix_fragment_27,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_26, matrix_fragment_27));

                const uint64_t byte_address_2 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_24, matrix_fragment_25));
                matrix_fragment_30 = uint32_t(byte_address_2);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_28, matrix_fragment_29), matrix_fragment_32,
                    matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
                    error_counts);

                address_word_5 = uint32_t(byte_address_1 >> 32);

                const uint64_t byte_address_3 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_26, matrix_fragment_27));
                matrix_accumulator_80 = uint32_t(byte_address_3);

                matrix_fragment_31 = uint32_t(byte_address_2 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_fragment_36,
                    matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
                    error_counts);

                matrix_accumulator_81 = uint32_t(byte_address_3 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_30, matrix_fragment_31), matrix_accumulator_40,
                    matrix_accumulator_41, matrix_accumulator_42, matrix_accumulator_43,
                    memory_ranges, error_counts);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_accumulator_80, matrix_accumulator_81),
                    matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_46,
                    matrix_accumulator_47, memory_ranges, error_counts);
            }
            if (!condition_0) {
                uniform_index_word_4 = parameters[parameter_reduction_scratch_optional_low_10];
                uniform_address_word_5 = parameters[parameter_reduction_scratch_optional_high_11];

                address_word_4 = index_word_82 * 0x2000u + address_word_83;

                address_word_6 = address_word_4 + 0x80u;

                address_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

                matrix_fragment_24 = address_word_4 + 0x100u;

                matrix_fragment_26 = address_word_4 + 0x180u;

                address_word_7 = uint32_t(int32_t(address_word_6) >> 0x1fu);

                split_address_words(address_word_4, address_word_5,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_4, address_word_5));

                matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_24) >> 0x1fu);

                matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_26) >> 0x1fu);

                split_address_words(address_word_6, address_word_7,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_6, address_word_7));

                const uint64_t byte_address_4 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_4, address_word_5));
                matrix_fragment_28 = uint32_t(byte_address_4);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_24, matrix_fragment_25));

                matrix_fragment_29 = uint32_t(byte_address_4 >> 32);

                const uint64_t byte_address_5 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_6, address_word_7));
                address_word_4 = uint32_t(byte_address_5);

                split_address_words(matrix_fragment_26, matrix_fragment_27,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_26, matrix_fragment_27));

                const uint64_t byte_address_6 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_24, matrix_fragment_25));
                matrix_fragment_30 = uint32_t(byte_address_6);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_28, matrix_fragment_29), matrix_fragment_48,
                    matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
                    error_counts);

                address_word_5 = uint32_t(byte_address_5 >> 32);

                const uint64_t byte_address_7 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_26, matrix_fragment_27));
                matrix_fragment_32 = uint32_t(byte_address_7);

                matrix_fragment_31 = uint32_t(byte_address_6 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_accumulator_52,
                    matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55,
                    memory_ranges, error_counts);

                matrix_fragment_33 = uint32_t(byte_address_7 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_30, matrix_fragment_31), matrix_accumulator_56,
                    matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                    memory_ranges, error_counts);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_32, matrix_fragment_33), matrix_accumulator_60,
                    matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                    memory_ranges, error_counts);
            }
            if (!condition_1) {
                uniform_index_word_4 = parameters[parameter_reduction_scratch_optional_low_10];
                uniform_address_word_5 = parameters[parameter_reduction_scratch_optional_high_11];

                address_word_4 = matrix_accumulator_84 * 0x2000u + address_word_83;

                address_word_6 = address_word_4 + 0x80u;

                address_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

                matrix_fragment_24 = address_word_4 + 0x100u;

                matrix_fragment_26 = address_word_4 + 0x180u;

                address_word_7 = uint32_t(int32_t(address_word_6) >> 0x1fu);

                split_address_words(address_word_4, address_word_5,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_4, address_word_5));

                matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_24) >> 0x1fu);

                matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_26) >> 0x1fu);

                split_address_words(address_word_6, address_word_7,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_6, address_word_7));

                const uint64_t byte_address_8 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_4, address_word_5));
                matrix_fragment_28 = uint32_t(byte_address_8);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_24, matrix_fragment_25));

                matrix_fragment_29 = uint32_t(byte_address_8 >> 32);

                const uint64_t byte_address_9 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_6, address_word_7));
                address_word_4 = uint32_t(byte_address_9);

                split_address_words(matrix_fragment_26, matrix_fragment_27,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_26, matrix_fragment_27));

                const uint64_t byte_address_10 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_24, matrix_fragment_25));
                matrix_fragment_30 = uint32_t(byte_address_10);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_28, matrix_fragment_29), matrix_fragment_64,
                    matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
                    error_counts);

                address_word_5 = uint32_t(byte_address_9 >> 32);

                const uint64_t byte_address_11 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_26, matrix_fragment_27));
                matrix_fragment_32 = uint32_t(byte_address_11);

                matrix_fragment_31 = uint32_t(byte_address_10 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_accumulator_68,
                    matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71,
                    memory_ranges, error_counts);

                matrix_fragment_33 = uint32_t(byte_address_11 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_30, matrix_fragment_31), matrix_accumulator_72,
                    matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75,
                    memory_ranges, error_counts);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_32, matrix_fragment_33), matrix_accumulator_76,
                    matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                    memory_ranges, error_counts);
            }
            if (!condition_2) {
                uniform_index_word_4 = parameters[parameter_reduction_scratch_optional_low_10];
                uniform_address_word_5 = parameters[parameter_reduction_scratch_optional_high_11];

                address_word_4 = index_word_86 * 0x2000u + address_word_83;

                address_word_6 = address_word_4 + 0x80u;

                address_word_5 = uint32_t(int32_t(address_word_4) >> 0x1fu);

                matrix_fragment_24 = address_word_4 + 0x100u;

                matrix_fragment_26 = address_word_4 + 0x180u;

                address_word_7 = uint32_t(int32_t(address_word_6) >> 0x1fu);

                split_address_words(address_word_4, address_word_5,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_4, address_word_5));

                matrix_fragment_25 = uint32_t(int32_t(matrix_fragment_24) >> 0x1fu);

                matrix_fragment_27 = uint32_t(int32_t(matrix_fragment_26) >> 0x1fu);

                split_address_words(address_word_6, address_word_7,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_6, address_word_7));

                const uint64_t byte_address_12 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_4, address_word_5));
                matrix_fragment_28 = uint32_t(byte_address_12);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_24, matrix_fragment_25));

                matrix_fragment_29 = uint32_t(byte_address_12 >> 32);

                const uint64_t byte_address_13 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_6, address_word_7));
                address_word_4 = uint32_t(byte_address_13);

                split_address_words(matrix_fragment_26, matrix_fragment_27,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_26, matrix_fragment_27));

                const uint64_t byte_address_14 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_24, matrix_fragment_25));
                matrix_fragment_30 = uint32_t(byte_address_14);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_28, matrix_fragment_29), matrix_fragment_20,
                    matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
                    error_counts);

                address_word_5 = uint32_t(byte_address_13 >> 32);

                const uint64_t byte_address_15 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_26, matrix_fragment_27));
                matrix_fragment_32 = uint32_t(byte_address_15);

                matrix_fragment_31 = uint32_t(byte_address_14 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_fragment_16,
                    matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
                    error_counts);

                matrix_fragment_33 = uint32_t(byte_address_15 >> 32);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_30, matrix_fragment_31), matrix_fragment_12,
                    matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
                    error_counts);

                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_32, matrix_fragment_33), matrix_fragment_8,
                    matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
                    error_counts);
            }
        } else {
            address_word_6 = parameters[parameter_reduction_scratch_optional_low_10];
            address_word_7 = parameters[parameter_reduction_scratch_optional_high_11];

            matrix_accumulator_80 = index_word_86 * 0x8u + index_word_91;

            matrix_fragment_24 = 0;
            matrix_fragment_25 = 0;

            matrix_fragment_26 = 0;
            matrix_fragment_27 = 0;

            matrix_fragment_28 = 0;
            matrix_fragment_29 = 0;

            address_word_83 = matrix_accumulator_80 * 0x2000u + address_word_83;

            condition_0 = int32_t(matrix_accumulator_80) >= int32_t(index_word_87);

            matrix_accumulator_84 = matrix_accumulator_80 + 0x1u;

            matrix_accumulator_92 = matrix_accumulator_80 + 0x3u;

            index_word_86 = matrix_accumulator_80 + 0x2u;

            matrix_fragment_30 = 0;
            matrix_fragment_31 = 0;

            condition_1 = int32_t(matrix_accumulator_84) >= int32_t(index_word_87);

            condition_3 = int32_t(matrix_accumulator_92) >= int32_t(index_word_87);

            condition_2 = int32_t(index_word_86) >= int32_t(index_word_87);

            split_address_words(address_word_6, address_word_7,
                                int64_t(int32_t(address_word_83)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_6, address_word_7));

            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                combine_words(address_word_6, address_word_7) + 0x200u);

            split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                                combine_words(address_word_6, address_word_7) + 0x8000u);

            split_address_words(matrix_accumulator_98, matrix_accumulator_99,
                                combine_words(address_word_6, address_word_7) + 0x400u);

            if (!condition_0) {
                split_address_words(
                    matrix_accumulator_88, matrix_accumulator_89,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_88, matrix_accumulator_89));
            }

            matrix_accumulator_92 = 0;
            matrix_accumulator_93 = 0;

            matrix_accumulator_94 = 0;
            matrix_accumulator_95 = 0;

            if (!condition_1) {
                split_address_words(
                    matrix_accumulator_102, matrix_accumulator_103,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_102, matrix_accumulator_103));
            }

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_88, matrix_accumulator_89), matrix_fragment_24,
                    matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
                    error_counts);
            }

            matrix_accumulator_84 = 0;
            matrix_accumulator_85 = 0;

            index_word_86 = 0;
            index_word_87 = 0;

            if (!condition_0) {
                split_address_words(
                    matrix_accumulator_98, matrix_accumulator_99,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_98, matrix_accumulator_99));
            }

            split_address_words(address_word_120, address_word_121,
                                combine_words(address_word_6, address_word_7) + 0x8200u);

            split_address_words(address_word_122, address_word_123,
                                combine_words(address_word_6, address_word_7) + 0x8400u);

            split_address_words(address_word_124, address_word_125,
                                combine_words(address_word_6, address_word_7) + 0x8600u);

            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                combine_words(address_word_6, address_word_7) + 0x600u);

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_102, matrix_accumulator_103),
                    matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
                    matrix_accumulator_95, memory_ranges, error_counts);
            }

            if (!condition_0) {
                split_address_words(
                    matrix_accumulator_88, matrix_accumulator_89,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_88, matrix_accumulator_89));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_98, matrix_accumulator_99), matrix_fragment_28,
                    matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
                    error_counts);
            }

            matrix_accumulator_96 = 0;
            matrix_accumulator_97 = 0;

            matrix_accumulator_104 = 0;
            matrix_accumulator_105 = 0;

            matrix_accumulator_106 = 0;
            matrix_accumulator_107 = 0;

            matrix_accumulator_108 = 0;
            matrix_accumulator_109 = 0;

            split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                                combine_words(address_word_6, address_word_7) + 0x10200u);

            matrix_accumulator_110 = 0;
            matrix_accumulator_111 = 0;

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_88, matrix_accumulator_89),
                    matrix_accumulator_84, matrix_accumulator_85, index_word_86, index_word_87,
                    memory_ranges, error_counts);
            }

            if (!condition_1) {
                split_address_words(address_word_120, address_word_121,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_120, address_word_121));
            }

            matrix_accumulator_98 = 0;
            matrix_accumulator_99 = 0;

            packed_half_116 = 0;
            packed_half_117 = 0;

            packed_half_118 = 0;
            packed_half_119 = 0;

            if (!condition_1) {
                split_address_words(address_word_122, address_word_123,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_122, address_word_123));
            }

            split_address_words(address_word_140, address_word_141,
                                combine_words(address_word_6, address_word_7) + 0x10400u);

            split_address_words(address_word_142, address_word_143,
                                combine_words(address_word_6, address_word_7) + 0x10600u);

            if (!condition_1) {
                split_address_words(address_word_124, address_word_125,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_124, address_word_125));
            }

            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                combine_words(address_word_6, address_word_7) + 0x10000u);

            split_address_words(address_word_144, address_word_145,
                                combine_words(address_word_6, address_word_7) + 0x18000u);

            if (!condition_2) {
                split_address_words(
                    matrix_accumulator_102, matrix_accumulator_103,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_102, matrix_accumulator_103));
            }

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_120, address_word_121), matrix_accumulator_96,
                    matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99,
                    memory_ranges, error_counts);
            }

            matrix_accumulator_112 = 0;
            matrix_accumulator_113 = 0;

            matrix_accumulator_114 = 0;
            matrix_accumulator_115 = 0;

            packed_half_126 = 0;
            packed_half_127 = 0;

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_122, address_word_123), matrix_accumulator_104,
                    matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
                    memory_ranges, error_counts);
            }

            packed_half_128 = 0;
            packed_half_129 = 0;

            packed_half_130 = 0;
            packed_half_131 = 0;

            if (!condition_2) {
                split_address_words(
                    matrix_accumulator_88, matrix_accumulator_89,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_88, matrix_accumulator_89));
            }

            if (!condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_124, address_word_125), matrix_accumulator_108,
                    matrix_accumulator_109, matrix_accumulator_110, matrix_accumulator_111,
                    memory_ranges, error_counts);
            }

            split_address_words(address_word_146, address_word_147,
                                combine_words(address_word_6, address_word_7) + 0x18400u);

            address_word_120 = 0;
            address_word_121 = 0;

            if (!condition_2) {
                split_address_words(address_word_140, address_word_141,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_140, address_word_141));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_102, matrix_accumulator_103), packed_half_116,
                    packed_half_117, packed_half_118, packed_half_119, memory_ranges, error_counts);
            }

            address_word_122 = 0;
            address_word_123 = 0;

            if (!condition_2) {
                split_address_words(address_word_142, address_word_143,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_142, address_word_143));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_88, matrix_accumulator_89),
                    matrix_accumulator_112, matrix_accumulator_113, matrix_accumulator_114,
                    matrix_accumulator_115, memory_ranges, error_counts);
            }

            address_word_124 = 0;
            address_word_125 = 0;

            if (!condition_3) {
                split_address_words(address_word_144, address_word_145,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_144, address_word_145));
            }

            if (!condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_140, address_word_141), address_word_120,
                    address_word_121, address_word_122, address_word_123, memory_ranges,
                    error_counts);
            }

            split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                                combine_words(address_word_6, address_word_7) + 0x18200u);

            if (!condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_142, address_word_143), address_word_124,
                    address_word_125, packed_half_126, packed_half_127, memory_ranges,
                    error_counts);
            }

            packed_half_136 = 0;
            packed_half_137 = 0;

            packed_half_138 = 0;
            packed_half_139 = 0;

            if (!condition_0) {
                split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_6, address_word_7));
            }

            if (!condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_144, address_word_145), packed_half_128,
                    packed_half_129, packed_half_130, packed_half_131, memory_ranges, error_counts);
            }

            packed_half_132 = 0;
            packed_half_133 = 0;

            packed_half_134 = 0;
            packed_half_135 = 0;

            address_word_140 = 0;
            address_word_141 = 0;

            address_word_5 = 0u;

            address_word_6 = 0;
            address_word_7 = 0;

            address_word_142 = 0;
            address_word_143 = 0;

            if (!condition_3) {
                split_address_words(
                    matrix_accumulator_102, matrix_accumulator_103,
                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_102, matrix_accumulator_103));
            }

            if (!condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_88, matrix_accumulator_89), packed_half_136,
                    packed_half_137, packed_half_138, packed_half_139, memory_ranges, error_counts);
            }

            if (!condition_3) {
                split_address_words(address_word_144, address_word_145,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_146, address_word_147));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_102, matrix_accumulator_103), packed_half_132,
                    packed_half_133, packed_half_134, packed_half_135, memory_ranges, error_counts);

                split_address_words(address_word_146, address_word_147,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_146, address_word_147));

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_144, address_word_145), address_word_140,
                    address_word_141, address_word_142, address_word_143, memory_ranges,
                    error_counts);

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_146, address_word_147) + 0x200ull, address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);
            }

            index_word_82 = matrix_accumulator_80 * 0x400u + index_word_82;

            matrix_fragment_24 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                               select_half_operand<2, false, false>(matrix_fragment_36));

            matrix_fragment_25 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                               select_half_operand<2, false, false>(matrix_fragment_37));

            matrix_fragment_36 = parameters[parameter_output_low_4];
            matrix_fragment_37 = parameters[parameter_output_high_5];

            matrix_fragment_27 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                               select_half_operand<2, false, false>(matrix_fragment_39));

            matrix_fragment_39 = index_word_82 * 0x4u;

            matrix_fragment_26 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                               select_half_operand<2, false, false>(matrix_fragment_38));

            matrix_fragment_48 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                               select_half_operand<2, false, false>(matrix_fragment_48));

            matrix_fragment_49 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                               select_half_operand<2, false, false>(matrix_fragment_49));

            matrix_fragment_50 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                               select_half_operand<2, false, false>(matrix_fragment_50));

            matrix_fragment_28 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                               select_half_operand<2, false, false>(matrix_accumulator_40));

            matrix_fragment_29 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                               select_half_operand<2, false, false>(matrix_accumulator_41));

            matrix_fragment_30 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                               select_half_operand<2, false, false>(matrix_accumulator_42));

            matrix_fragment_31 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                               select_half_operand<2, false, false>(matrix_accumulator_43));

            matrix_accumulator_44 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                               select_half_operand<2, false, false>(matrix_accumulator_44));

            matrix_accumulator_45 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                               select_half_operand<2, false, false>(matrix_accumulator_45));

            split_address_words(matrix_fragment_36, matrix_fragment_37,
                                int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_36, matrix_fragment_37));

            matrix_accumulator_46 =
                add_half_pairs(select_half_operand<2, false, false>(index_word_86),
                               select_half_operand<2, false, false>(matrix_accumulator_46));

            matrix_accumulator_47 =
                add_half_pairs(select_half_operand<2, false, false>(index_word_87),
                               select_half_operand<2, false, false>(matrix_accumulator_47));

            matrix_fragment_51 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                               select_half_operand<2, false, false>(matrix_fragment_51));

            matrix_accumulator_52 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                               select_half_operand<2, false, false>(matrix_accumulator_52));

            matrix_accumulator_53 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                               select_half_operand<2, false, false>(matrix_accumulator_53));

            matrix_accumulator_54 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                               select_half_operand<2, false, false>(matrix_accumulator_54));

            matrix_accumulator_55 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                               select_half_operand<2, false, false>(matrix_accumulator_55));

            matrix_accumulator_56 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                               select_half_operand<2, false, false>(matrix_accumulator_56));

            matrix_accumulator_57 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                               select_half_operand<2, false, false>(matrix_accumulator_57));

            matrix_accumulator_58 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                               select_half_operand<2, false, false>(matrix_accumulator_58));

            matrix_accumulator_59 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                               select_half_operand<2, false, false>(matrix_accumulator_59));

            matrix_accumulator_60 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                               select_half_operand<2, false, false>(matrix_accumulator_60));

            matrix_accumulator_61 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                               select_half_operand<2, false, false>(matrix_accumulator_61));

            matrix_accumulator_62 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                               select_half_operand<2, false, false>(matrix_accumulator_62));

            matrix_accumulator_63 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                               select_half_operand<2, false, false>(matrix_accumulator_63));

            matrix_fragment_64 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                               select_half_operand<2, false, false>(matrix_fragment_64));

            matrix_fragment_65 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                               select_half_operand<2, false, false>(matrix_fragment_65));

            matrix_fragment_66 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                               select_half_operand<2, false, false>(matrix_fragment_66));

            matrix_fragment_67 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                               select_half_operand<2, false, false>(matrix_fragment_67));

            matrix_accumulator_68 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                               select_half_operand<2, false, false>(matrix_accumulator_68));

            matrix_accumulator_69 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                               select_half_operand<2, false, false>(matrix_accumulator_69));

            matrix_accumulator_70 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                               select_half_operand<2, false, false>(matrix_accumulator_70));

            packed_half_136 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                               select_half_operand<2, false, false>(matrix_fragment_32));

            packed_half_137 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                               select_half_operand<2, false, false>(matrix_fragment_33));

            packed_half_138 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                               select_half_operand<2, false, false>(matrix_fragment_34));

            packed_half_139 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                               select_half_operand<2, false, false>(matrix_fragment_35));

            matrix_accumulator_71 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                               select_half_operand<2, false, false>(matrix_accumulator_71));

            matrix_accumulator_72 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_120),
                               select_half_operand<2, false, false>(matrix_accumulator_72));

            matrix_accumulator_73 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_121),
                               select_half_operand<2, false, false>(matrix_accumulator_73));

            matrix_accumulator_74 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_122),
                               select_half_operand<2, false, false>(matrix_accumulator_74));

            matrix_accumulator_75 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_123),
                               select_half_operand<2, false, false>(matrix_accumulator_75));

            matrix_accumulator_76 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_124),
                               select_half_operand<2, false, false>(matrix_accumulator_76));

            matrix_accumulator_77 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_125),
                               select_half_operand<2, false, false>(matrix_accumulator_77));

            matrix_accumulator_78 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_126),
                               select_half_operand<2, false, false>(matrix_accumulator_78));

            matrix_accumulator_79 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_127),
                               select_half_operand<2, false, false>(matrix_accumulator_79));

            matrix_fragment_20 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                               select_half_operand<2, false, false>(matrix_fragment_20));

            matrix_fragment_21 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                               select_half_operand<2, false, false>(matrix_fragment_21));

            matrix_fragment_22 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                               select_half_operand<2, false, false>(matrix_fragment_22));

            matrix_fragment_23 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                               select_half_operand<2, false, false>(matrix_fragment_23));

            matrix_fragment_16 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_132),
                               select_half_operand<2, false, false>(matrix_fragment_16));

            matrix_fragment_17 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                               select_half_operand<2, false, false>(matrix_fragment_17));

            matrix_fragment_18 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                               select_half_operand<2, false, false>(matrix_fragment_18));

            matrix_fragment_19 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                               select_half_operand<2, false, false>(matrix_fragment_19));

            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_140),
                               select_half_operand<2, false, false>(matrix_fragment_12));

            matrix_fragment_13 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_141),
                               select_half_operand<2, false, false>(matrix_fragment_13));

            matrix_fragment_14 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_142),
                               select_half_operand<2, false, false>(matrix_fragment_14));

            matrix_fragment_15 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_143),
                               select_half_operand<2, false, false>(matrix_fragment_15));

            matrix_fragment_32 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_4),
                               select_half_operand<2, false, false>(matrix_fragment_8));

            matrix_fragment_33 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_5),
                               select_half_operand<2, false, false>(matrix_fragment_9));

            matrix_fragment_34 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_6),
                               select_half_operand<2, false, false>(matrix_fragment_10));

            matrix_fragment_35 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_7),
                               select_half_operand<2, false, false>(matrix_fragment_11));
            if (!condition_0) {
                address_word_6 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_8 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                  select_half_operand<2, false, false>(0u));

                address_word_4 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                                  select_half_operand<2, false, false>(0u));

                address_word_7 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                  select_half_operand<2, false, false>(address_word_6));

                matrix_fragment_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_9 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                  select_half_operand<2, false, false>(matrix_fragment_8));

                matrix_fragment_25 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_27 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_29 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                  select_half_operand<2, false, false>(0u));

                matrix_fragment_31 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                  select_half_operand<2, false, false>(0u));

                address_word_5 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                                  select_half_operand<2, false, false>(address_word_4));

                matrix_fragment_11 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                  select_half_operand<2, false, false>(matrix_fragment_10));

                address_word_4 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                  select_half_operand<2, false, false>(matrix_fragment_25));

                address_word_6 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                  select_half_operand<2, false, false>(matrix_fragment_27));

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_36, matrix_fragment_37));

                matrix_fragment_8 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                  select_half_operand<2, false, false>(matrix_fragment_29));

                matrix_fragment_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                  select_half_operand<2, false, false>(matrix_fragment_31));

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_24, matrix_fragment_25), address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_24, matrix_fragment_25) + 0x200ull,
                    matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                    memory_ranges, error_counts);
            }
            split_address_words(matrix_fragment_8, matrix_fragment_9,
                                combine_words(matrix_fragment_36, matrix_fragment_37) + 0x4000u);

            address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                              select_half_operand<2, false, false>(0u));

            address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                              select_half_operand<2, false, false>(0u));

            address_word_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_25 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                              select_half_operand<2, false, false>(0u));

            address_word_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_24 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_29 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_58 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_28 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_59 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_27 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_62 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_26 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_63 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                              select_half_operand<2, false, false>(0u));
            if (!condition_1) {
                address_word_5 = permute_signed(address_word_5, matrix_fragment_10, 0x5410u);

                address_word_4 = permute_signed(address_word_4, matrix_fragment_11, 0x5410u);

                split_address_words(matrix_fragment_10, matrix_fragment_11,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_8, matrix_fragment_9));

                address_word_7 = permute_signed(address_word_7, matrix_fragment_24, 0x5410u);

                address_word_6 = permute_signed(address_word_6, matrix_fragment_25, 0x5410u);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_36, matrix_fragment_37));

                matrix_fragment_26 =
                    permute_signed(matrix_fragment_26, matrix_accumulator_63, 0x5410u);

                matrix_fragment_27 =
                    permute_signed(matrix_fragment_27, matrix_accumulator_62, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_10, matrix_fragment_11), address_word_4,
                    address_word_5, address_word_6, address_word_7, memory_ranges, error_counts);

                matrix_fragment_28 =
                    permute_signed(matrix_fragment_28, matrix_accumulator_59, 0x5410u);

                matrix_fragment_29 =
                    permute_signed(matrix_fragment_29, matrix_accumulator_58, 0x5410u);

                address_word_4 = matrix_fragment_29;

                address_word_5 = matrix_fragment_28;

                address_word_6 = matrix_fragment_27;

                address_word_7 = matrix_fragment_26;

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_24, matrix_fragment_25) + 0x4200ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);
            }
            split_address_words(matrix_fragment_36, matrix_fragment_37,
                                combine_words(matrix_fragment_36, matrix_fragment_37) + 0x8000u);

            address_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_66 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                              select_half_operand<2, false, false>(0u));

            address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_67 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                              select_half_operand<2, false, false>(0u));

            address_word_5 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                              select_half_operand<2, false, false>(0u));

            address_word_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_29 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_74 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_28 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_75 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_27 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_78 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_26 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                              select_half_operand<2, false, false>(0u));

            matrix_accumulator_79 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                              select_half_operand<2, false, false>(0u));
            if (!condition_2) {
                address_word_6 = permute_signed(address_word_6, matrix_fragment_67, 0x5410u);

                split_address_words(matrix_fragment_24, matrix_fragment_25,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_8, matrix_fragment_9));

                address_word_7 = permute_signed(address_word_7, matrix_fragment_66, 0x5410u);

                matrix_fragment_11 = permute_signed(matrix_fragment_11, address_word_4, 0x5410u);

                matrix_fragment_10 = permute_signed(matrix_fragment_10, address_word_5, 0x5410u);

                split_address_words(address_word_4, address_word_5,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_36, matrix_fragment_37));

                matrix_fragment_8 = address_word_7;

                matrix_fragment_9 = address_word_6;

                matrix_fragment_26 =
                    permute_signed(matrix_fragment_26, matrix_accumulator_79, 0x5410u);

                matrix_fragment_27 =
                    permute_signed(matrix_fragment_27, matrix_accumulator_78, 0x5410u);

                matrix_fragment_28 =
                    permute_signed(matrix_fragment_28, matrix_accumulator_75, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_4, address_word_5), matrix_fragment_8,
                    matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
                    error_counts);

                matrix_fragment_29 =
                    permute_signed(matrix_fragment_29, matrix_accumulator_74, 0x5410u);

                address_word_6 = matrix_fragment_27;

                address_word_7 = matrix_fragment_26;

                address_word_4 = matrix_fragment_29;

                address_word_5 = matrix_fragment_28;

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_24, matrix_fragment_25) + 0x4200ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);
            }
            address_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                              select_half_operand<2, false, false>(0u));

            address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                              select_half_operand<2, false, false>(0u));

            address_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                              select_half_operand<2, false, false>(0u));

            address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                           select_half_operand<2, false, false>(0u));

            matrix_fragment_8 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_12 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_32 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_33 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_16 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                              select_half_operand<2, false, false>(0u));
            if (!condition_3) {
                address_word_7 = permute_signed(address_word_7, matrix_fragment_10, 0x5410u);

                split_address_words(matrix_fragment_36, matrix_fragment_37,
                                    int64_t(int32_t(address_word_90)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_36, matrix_fragment_37));

                address_word_6 = permute_signed(address_word_6, matrix_fragment_11, 0x5410u);

                address_word_5 = permute_signed(address_word_5, matrix_fragment_8, 0x5410u);

                address_word_4 = permute_signed(address_word_4, matrix_fragment_9, 0x5410u);

                matrix_fragment_11 =
                    permute_signed(matrix_fragment_33, matrix_fragment_16, 0x5410u);

                matrix_fragment_10 =
                    permute_signed(matrix_fragment_32, matrix_fragment_15, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_36, matrix_fragment_37) + 0x4000ull,
                    address_word_4, address_word_5, address_word_6, address_word_7, memory_ranges,
                    error_counts);

                matrix_fragment_9 = permute_signed(matrix_fragment_13, matrix_fragment_14, 0x5410u);

                matrix_fragment_8 = permute_signed(matrix_fragment_12, matrix_fragment_17, 0x5410u);

                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_36, matrix_fragment_37) + 0x4200ull,
                    matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                    memory_ranges, error_counts);
            }
        }
    }
    address_word_5 = threadIdx.z;

    __syncthreads();

    condition_0 = (address_word_5 | thread_row_100) != 0;

    if (condition_0) {
        return;
    }

    __threadfence();

    write_device_flag<CheckBounds>(combine_words(address_word_2, address_word_3), index_word_0,
                                   memory_ranges, error_counts);
    return;
}
extern "C" __global__ __maxnreg__(144) void phoenix_global_attention_transformer_projection_wait(
    KernelArguments parameters) {
    run_global_attention_transformer_projection_wait<false>(parameters.words, nullptr, nullptr);
}
