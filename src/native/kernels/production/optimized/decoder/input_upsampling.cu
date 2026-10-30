#include "../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: decoder/input_upsampling.cu
// Decoder input projection/upsampling stage, combining the coarse result with its matching encoder skip.
// The launch ABI and skip/publication buffers are described by the native model launch contract.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 3088
#include "../../common/async_memory.cuh"
#include "../../common/cached_vector_access.cuh"

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
#if PHOENIX_STYLE_ARCH >= 900
    asm volatile("red.global.add.noftz.v4.f16x2 [%0],{%1,%2,%3,%4};" ::"l"(address), "r"(word_x),
                 "r"(word_y), "r"(word_z), "r"(word_w)
                 : "memory");
#else
    asm volatile("red.global.add.noftz.f16x2 [%0],%1;\n"
                 "red.global.add.noftz.f16x2 [%0+4],%2;\n"
                 "red.global.add.noftz.f16x2 [%0+8],%3;\n"
                 "red.global.add.noftz.f16x2 [%0+12],%4;"
                 :: "l"(address), "r"(word_x), "r"(word_y), "r"(word_z), "r"(word_w) : "memory");
#endif
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
#include "../../common/token_dispatch.cuh"

// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_low_0 = 0,
    parameter_input_low_high_1 = 1,
    parameter_input_encoder_skip_low_2 = 2,
    parameter_input_encoder_skip_high_3 = 3,
    parameter_output_low_4 = 4,
    parameter_output_high_5 = 5,
    parameter_counter_low_8 = 8,
    parameter_counter_high_9 = 9,
    parameter_publication_low_10 = 10,
    parameter_publication_high_11 = 11,
    parameter_scratch_secondary_low_12 = 12,
    parameter_scratch_secondary_high_13 = 13,
    parameter_trained_weights_low_14 = 14,
    parameter_trained_weights_high_15 = 15,
    parameter_LH_16 = 16,
    parameter_LW_17 = 17,
    parameter_tensor_height_18 = 18,
    parameter_tensor_width_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[20];
};
static_assert(sizeof(KernelArguments) == 80, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_decoder_input_upsampling(const uint32_t* parameters,
                                                             const MemoryRange* memory_ranges,
                                                             uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t address_word_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t packed_half_8 = 0;
    uint32_t packed_half_9 = 0;
    uint32_t packed_half_10 = 0;
    uint32_t address_word_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t packed_half_16 = 0;
    uint32_t matrix_fragment_17 = 0;
    uint32_t packed_half_18 = 0;
    uint32_t matrix_fragment_19 = 0;
    uint32_t matrix_fragment_20 = 0;
    uint32_t packed_half_21 = 0;
    uint32_t packed_half_22 = 0;
    uint32_t packed_half_23 = 0;
    uint32_t address_word_24 = 0;
    uint32_t address_word_25 = 0;
    uint32_t address_word_26 = 0;
    uint32_t address_word_27 = 0;
    uint32_t packed_half_28 = 0;
    uint32_t packed_half_29 = 0;
    uint32_t packed_half_30 = 0;
    uint32_t packed_half_31 = 0;
    uint32_t address_word_32 = 0;
    uint32_t address_word_33 = 0;
    uint32_t packed_half_34 = 0;
    uint32_t address_word_35 = 0;
    uint32_t packed_half_36 = 0;
    uint32_t address_word_37 = 0;
    uint32_t address_word_38 = 0;
    uint32_t address_word_39 = 0;
    uint32_t matrix_fragment_40 = 0;
    uint32_t matrix_fragment_41 = 0;
    uint32_t packed_half_42 = 0;
    uint32_t matrix_fragment_43 = 0;
    uint32_t matrix_fragment_44 = 0;
    uint32_t matrix_fragment_45 = 0;
    uint32_t matrix_fragment_46 = 0;
    uint32_t matrix_fragment_47 = 0;
    uint32_t packed_half_48 = 0;
    uint32_t packed_half_49 = 0;
    uint32_t packed_half_50 = 0;
    uint32_t packed_half_51 = 0;
    uint32_t matrix_fragment_52 = 0;
    uint32_t matrix_fragment_53 = 0;
    uint32_t matrix_fragment_54 = 0;
    uint32_t matrix_fragment_55 = 0;
    uint32_t address_word_56 = 0;
    uint32_t packed_half_57 = 0;
    uint32_t address_word_58 = 0;
    uint32_t packed_half_59 = 0;
    uint32_t matrix_fragment_60 = 0;
    uint32_t matrix_fragment_61 = 0;
    uint32_t matrix_fragment_62 = 0;
    uint32_t matrix_fragment_63 = 0;
    uint32_t matrix_fragment_64 = 0;
    uint32_t matrix_fragment_65 = 0;
    uint32_t matrix_fragment_66 = 0;
    uint32_t packed_half_67 = 0;
    uint32_t matrix_fragment_68 = 0;
    uint32_t matrix_fragment_69 = 0;
    uint32_t matrix_fragment_70 = 0;
    uint32_t matrix_fragment_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t matrix_fragment_74 = 0;
    uint32_t matrix_fragment_75 = 0;
    uint32_t packed_half_76 = 0;
    uint32_t packed_half_77 = 0;
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
    uint32_t packed_half_100 = 0;
    uint32_t matrix_fragment_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
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
    uint32_t packed_half_114 = 0;
    uint32_t packed_half_115 = 0;
    uint32_t packed_half_116 = 0;
    uint32_t packed_half_117 = 0;
    uint32_t packed_half_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t matrix_accumulator_120 = 0;
    uint32_t matrix_accumulator_121 = 0;
    uint32_t matrix_accumulator_122 = 0;
    uint32_t matrix_accumulator_123 = 0;
    uint32_t packed_half_124 = 0;
    uint32_t address_word_125 = 0;
    uint32_t packed_half_126 = 0;
    uint32_t packed_half_127 = 0;
    uint32_t matrix_fragment_128 = 0;
    uint32_t matrix_fragment_129 = 0;
    uint32_t matrix_fragment_130 = 0;
    uint32_t matrix_fragment_131 = 0;
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
    uint32_t packed_half_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t packed_half_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t packed_half_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_address_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_address_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_block_index_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_reused_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];
    reused_word_1 = 0u;
    address_word_56 = parameters[parameter_LW_17];
    matrix_accumulator_79 = blockIdx.x;
    uniform_reused_word_12 = 0u;
    uniform_reused_word_13 = 0u;
    matrix_fragment_6 = threadIdx.x;
    packed_half_49 = parameters[parameter_LH_16];
    packed_half_77 = threadIdx.y;
    packed_half_76 = blockIdx.z;
    matrix_accumulator_81 = threadIdx.x & 31u;
    index_word_0 = address_word_56 + 0xffffffffu;
    matrix_fragment_73 = truncate_signed_power_of_two<2>(index_word_0);
    packed_half_59 = matrix_fragment_6 | packed_half_77;
    condition_1 = packed_half_59 != 0;
    address_word_58 = blockIdx.y;
    index_word_0 = matrix_fragment_73 + 0x1u;
    matrix_fragment_75 = packed_half_77 * 0x80u;
    const auto work = token_tile_work(matrix_accumulator_79, index_word_0);
    const auto copy = tile_copy_schedule<shared_bytes, 2>(!condition_1, blockDim.x * blockDim.y);
    uniform_condition_0 = copy.without_block_leader;
    uniform_condition_1 = copy.without_block_leader;
    uniform_index_word_4 = copy.wait_word_low;
    uniform_address_word_5 = copy.wait_word_high;
    uniform_address_word_6 = copy.barrier_byte_offset;
    packed_half_48 = work.group;
    matrix_fragment_4 = parameters[parameter_trained_weights_low_14];
    matrix_fragment_5 = parameters[parameter_trained_weights_high_15];
    address_word_2 = work.group * 256u + matrix_fragment_75;
    address_word_35 = address_word_2 * 8u;
    index_word_0 = uint32_t(int32_t(packed_half_49) >> 0x1fu);
    address_word_3 = packed_half_76 * 0x8000u + address_word_35;
    packed_half_34 = packed_half_49 & 0xfffffffcu;
    index_word_0 = uint32_t(combine_words(index_word_0, 0u) << 0x2u >> 32) + packed_half_49;
    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(address_word_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));
    if (!uniform_condition_0) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6);
#endif
    }
    condition_2 = uint32_t(packed_half_34) != uint32_t(0x4u);
    packed_half_57 = uint32_t(int32_t(index_word_0) >> 0x2u);
    split_address_words(address_word_32, address_word_33,
                        int64_t(int32_t(matrix_accumulator_81)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));
    address_word_3 = uint32_t(int32_t(address_word_56) >> 0x1fu);
    if (!uniform_condition_1) {
#if DLSS5_ENABLE_ASYNC_COPIES
        initialize_copy_barrier(shared_tile, uniform_address_word_6 + 0x8u);
#endif
    }
    phoenix_architecture::finish_compatibility_copies();
    condition_1 = int32_t(address_word_58) < int32_t(packed_half_57) || !condition_2;
    address_word_3 = uint32_t(combine_words(address_word_3, 0u) << 0x2u >> 32) + address_word_56;
    packed_half_51 = uint32_t(int32_t(address_word_3) >> 0x2u);
    packed_half_34 = work.tile;
    packed_half_49 = packed_half_51 * address_word_58;
    condition_0 = false;
    uniform_block_index_10 = blockIdx.z;
    address_word_3 = condition_2 ? packed_half_49 : 0u;
    index_word_0 = packed_half_34;
    uniform_address_word_5 = 0u;
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33),
                                        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70,
                                        matrix_fragment_71, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x200ull,
                                        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6,
                                        matrix_fragment_7, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x400ull,
                                        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66,
                                        packed_half_67, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x600ull,
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x800ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0xa00ull,
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0xc00ull,
                                        packed_half_36, address_word_37, address_word_38,
                                        address_word_39, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0xe00ull,
                                        packed_half_28, packed_half_29, packed_half_30,
                                        packed_half_31, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4000ull,
                                        address_word_24, address_word_25, address_word_26,
                                        address_word_27, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4200ull,
                                        packed_half_100, matrix_fragment_101, matrix_fragment_102,
                                        matrix_fragment_103, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4400ull,
                                        packed_half_8, packed_half_9, packed_half_10,
                                        address_word_11, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4600ull,
                                        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14,
                                        matrix_fragment_15, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(
        combine_words(address_word_32, address_word_33) + 0x4800ull, matrix_fragment_128,
        matrix_fragment_129, matrix_fragment_130, matrix_fragment_131, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4a00ull,
                                        packed_half_16, matrix_fragment_17, packed_half_18,
                                        matrix_fragment_19, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4c00ull,
                                        matrix_fragment_20, packed_half_21, packed_half_22,
                                        packed_half_23, memory_ranges, error_counts);
    load_four_words_cached<CheckBounds>(combine_words(address_word_32, address_word_33) + 0x4e00ull,
                                        matrix_fragment_40, matrix_fragment_41, packed_half_42,
                                        matrix_fragment_43, memory_ranges, error_counts);
    address_word_32 = address_word_58 * 0x2u;
    if (condition_1) {
        index_word_0 = address_word_56 & 0xfffffffcu;
        condition_2 = uint32_t(index_word_0) != uint32_t(0x4u);
        index_word_0 = 0u;
        condition_0 = int32_t(packed_half_34) < int32_t(packed_half_51) || !condition_2;
        if (condition_2) {
            index_word_0 = packed_half_34;
        }
    }
    if (condition_0) {
        address_word_33 = address_word_3 + index_word_0;

        packed_half_48 = packed_half_76 * 0x400u + matrix_fragment_75;
    }
    uniform_index_word_4 = 0x400u;
    index_word_0 = matrix_fragment_73 * address_word_32 + address_word_32;
    uniform_address_word_6 = (uniform_address_word_5 << 0x18u) + uniform_index_word_4;
    if (condition_0) {
        packed_half_50 = address_word_33 * 0x1000u + packed_half_48;
    }

    split_address_words(matrix_fragment_72, matrix_fragment_73, 0ull);
    index_word_0 = index_word_0 + matrix_accumulator_79;
    if (condition_0) {
        matrix_fragment_75 = uint32_t(int32_t(packed_half_50) >> 0x1fu);
    }
    packed_half_118 = (packed_half_77 << 0x9u) + uniform_address_word_6;
    packed_half_48 = 0u;
    address_word_33 = address_word_2 + 0x10u;
    if (condition_0) {
        matrix_fragment_72 = packed_half_50;

        matrix_fragment_73 = matrix_fragment_75;
    }
    if (!condition_0) {
        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
        packed_half_50 = matrix_accumulator_81 * 0x10u + packed_half_118;
        matrix_fragment_72 = uniform_address_word_6;
        matrix_fragment_73 = uniform_address_word_6;
        matrix_fragment_72 = permute_signed(matrix_fragment_72, matrix_fragment_73, 0x5410u);
        matrix_fragment_73 = matrix_fragment_72;
        matrix_fragment_74 = matrix_fragment_72;
        matrix_fragment_75 = matrix_fragment_72;
        store_four_shared_words<CheckBounds>(
            shared_tile, packed_half_50, matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
            matrix_fragment_75, shared_bytes, memory_ranges, error_counts);
    } else {
        condition_0 = int(threadIdx.x) == __ffs(__activemask()) - 1;
        if (condition_0) {
            matrix_fragment_74 = parameters[parameter_input_low_low_0];
            matrix_fragment_75 = parameters[parameter_input_low_high_1];
            uniform_address_word_6 = uniform_index_word_4 + 0x800u;
            check_single_lane<CheckBounds>(error_counts);
            uniform_address_word_14 = packed_half_118;
            uniform_address_word_15 = (uniform_address_word_5 << 0x18u) + uniform_address_word_6;
            uniform_address_word_6 = 0x20u;
            const uint64_t byte_address_0 =
                scaled_word_address(combine_words(matrix_fragment_74, matrix_fragment_75),
                                    combine_words(matrix_fragment_72, matrix_fragment_73));
            packed_half_50 = uint32_t(byte_address_0);
            matrix_fragment_72 = uint32_t(byte_address_0 >> 32);
            check_single_lane<CheckBounds>(error_counts);
            uniform_address_word_8 = packed_half_50;
            check_single_lane<CheckBounds>(error_counts);
            uniform_address_word_9 = matrix_fragment_72;
            packed_half_50 = 0x200u;
#if DLSS5_ENABLE_ASYNC_COPIES
            copy_global_to_shared<CheckBounds>(
                shared_tile, uniform_address_word_14,
                combine_words(uniform_address_word_8, uniform_address_word_9),
                uniform_address_word_6 * 16u, uniform_address_word_15, shared_bytes, memory_ranges,
                error_counts);
#else
            bulk_copy<CheckBounds>(shared_tile, uniform_address_word_14,
                                   combine_words(uniform_address_word_8, uniform_address_word_9),
                                   uniform_address_word_6 * 16u, shared_bytes, memory_ranges,
                                   error_counts);
#endif
        }
    }
    uniform_index_word_4 = uniform_index_word_4 + 0x800u;
    condition_0 = int32_t(packed_half_34) >= int32_t(packed_half_51);
    packed_half_119 = address_word_56 & 0xfffffffcu;
    uniform_index_word_4 = (uniform_address_word_5 << 0x18u) + uniform_index_word_4;
#if DLSS5_ENABLE_ASYNC_COPIES
    {
        const uint32_t operand_a = uniform_index_word_4;
        async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            async_tokens[operand_a >> 3 & 1]);
    }
#else
    matrix_fragment_72 = 0u;
#endif
#if DLSS5_ENABLE_ASYNC_COPIES
    condition_2 = wait_copy_barrier(shared_tile, uniform_index_word_4,
                                    async_tokens[(((uniform_index_word_4) >> 3) & 1)]);
#else
    phoenix_architecture::finish_compatibility_copies();
    condition_2 = true;
#endif
    if (!condition_2) {
        do {
#if DLSS5_ENABLE_ASYNC_COPIES
            condition_2 = wait_copy_barrier(shared_tile, uniform_index_word_4,
                                            async_tokens[(((uniform_index_word_4) >> 3) & 1)]);
#else
            phoenix_architecture::finish_compatibility_copies();
            condition_2 = true;
#endif
        } while (!condition_2);
    }
    packed_half_114 = parameters[parameter_trained_weights_low_14];
    packed_half_115 = parameters[parameter_trained_weights_high_15];
    condition_0 = uint32_t(packed_half_119) == uint32_t(0x4u) || !condition_0;
    matrix_accumulator_96 = 0;
    matrix_accumulator_97 = 0;
    matrix_accumulator_94 = 0;
    matrix_accumulator_95 = 0;
    matrix_accumulator_92 = 0;
    matrix_accumulator_93 = 0;
    matrix_accumulator_90 = 0;
    matrix_accumulator_91 = 0;
    matrix_accumulator_88 = 0;
    matrix_accumulator_89 = 0;
    matrix_accumulator_86 = 0;
    matrix_accumulator_87 = 0;
    packed_half_116 = parameters[parameter_input_low_low_0];
    packed_half_117 = parameters[parameter_input_low_high_1];
    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;
    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;
    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;
    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;
    matrix_accumulator_98 = 0;
    matrix_accumulator_99 = 0;
    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;
    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;
    condition_0 = condition_0 & condition_1;
    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;
    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;
    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;
    uniform_index_word_4 = 0u;
    do {
        uniform_address_word_9 = 0u;
        address_word_56 = threadIdx.x & 31u;
        uniform_address_word_5 = uniform_index_word_4 << (0x4u & 31u);
        uniform_address_word_8 = 0x400u;
        condition_2 = uint32_t(packed_half_119) != uint32_t(0x4u);
        uniform_index_word_7 = uniform_index_word_4;
        uniform_address_word_6 = uniform_address_word_5 & 0x400u;
        uniform_condition_0 = uint32_t(uniform_index_word_7) >= uint32_t(0x80u);

        uniform_index_word_16 = (uniform_address_word_9 << 0x18u) + uniform_address_word_8;
        uniform_address_word_8 = uniform_address_word_8 + 0x800u;
        uniform_address_word_5 = uniform_index_word_16 + uniform_address_word_6;
        uniform_address_word_9 = (uniform_address_word_9 << 0x18u) + uniform_address_word_8;
        uniform_address_word_8 = 0x400u ^ uniform_address_word_6;
        packed_half_50 = (address_word_56 << 0x4u) + uniform_address_word_5;
        uniform_address_word_5 = uniform_index_word_4 + 0x40u;
        uniform_index_word_4 = uint32_t(uint32_t(uniform_index_word_4) >> 0x6u);
        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_50, matrix_fragment_72, matrix_fragment_73, matrix_fragment_74,
            matrix_fragment_75, shared_bytes, memory_ranges, error_counts);
        uniform_index_word_11 = (uniform_block_index_10 << 0x8u) + uniform_address_word_5;
        uniform_index_word_4 = ~uniform_index_word_4 & 0x1u;
        uniform_index_word_7 = (uniform_index_word_4 << 0x3u) + uniform_address_word_9;
        uniform_index_word_4 = uniform_address_word_5;
        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_106,
                                 matrix_accumulator_107);
        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_104,
                                 matrix_accumulator_105);
        load_four_shared_words<CheckBounds>(shared_tile, packed_half_50 + 0x200u, matrix_fragment_4,
                                            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                            shared_bytes, memory_ranges, error_counts);
        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_110,
                                 matrix_accumulator_111);
        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_112,
                                 matrix_accumulator_113);
        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_108,
                                 matrix_accumulator_109);
        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, packed_half_67, matrix_accumulator_98,
                                 matrix_accumulator_99);
        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_78,
                                 matrix_accumulator_79);
        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_80,
                                 matrix_accumulator_81);
        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_82,
                                 matrix_accumulator_83);
        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_84,
                                 matrix_accumulator_85);
        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_86,
                                 matrix_accumulator_87);
        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_88,
                                 matrix_accumulator_89);
        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 packed_half_36, address_word_37, matrix_accumulator_90,
                                 matrix_accumulator_91);
        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 address_word_38, address_word_39, matrix_accumulator_92,
                                 matrix_accumulator_93);
        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 packed_half_28, packed_half_29, matrix_accumulator_94,
                                 matrix_accumulator_95);
        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 packed_half_30, packed_half_31, matrix_accumulator_96,
                                 matrix_accumulator_97);
        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_8, packed_half_9, matrix_accumulator_110,
                                 matrix_accumulator_111);
        packed_half_9 = !condition_1 ? packed_half_34 : 0u;
        packed_half_8 = uniform_index_word_11;
        packed_half_50 = !condition_2 ? packed_half_9 : packed_half_34;
        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 address_word_24, address_word_25, matrix_accumulator_112,
                                 matrix_accumulator_113);
        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 address_word_26, address_word_27, matrix_accumulator_108,
                                 matrix_accumulator_109);
        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_100, matrix_fragment_101, matrix_accumulator_106,
                                 matrix_accumulator_107);
        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_102, matrix_fragment_103, matrix_accumulator_104,
                                 matrix_accumulator_105);
        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_10, address_word_11, matrix_accumulator_98,
                                 matrix_accumulator_99);
        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_78,
                                 matrix_accumulator_79);
        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_80,
                                 matrix_accumulator_81);
        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_128, matrix_fragment_129, matrix_accumulator_82,
                                 matrix_accumulator_83);
        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_130, matrix_fragment_131, matrix_accumulator_84,
                                 matrix_accumulator_85);
        multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_16, matrix_fragment_17, matrix_accumulator_86,
                                 matrix_accumulator_87);
        multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_18, matrix_fragment_19, matrix_accumulator_88,
                                 matrix_accumulator_89);
        multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, packed_half_21, matrix_accumulator_90,
                                 matrix_accumulator_91);
        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_22, packed_half_23, matrix_accumulator_92,
                                 matrix_accumulator_93);
        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_94,
                                 matrix_accumulator_95);
        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 packed_half_42, matrix_fragment_43, matrix_accumulator_96,
                                 matrix_accumulator_97);
        if (condition_0) {
            matrix_fragment_4 = address_word_3 + packed_half_50;

            matrix_fragment_5 =
                uint32_t(combine_words(packed_half_8, 0u) << 0x1bu >> 32) + packed_half_77;

            matrix_fragment_7 = matrix_fragment_4 * 0x20u + matrix_fragment_5;
        }
        split_address_words(matrix_fragment_4, matrix_fragment_5, 0ull);
        if (condition_0) {
            matrix_fragment_7 = matrix_fragment_7 * 0x80u;

            matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

            matrix_fragment_4 = matrix_fragment_7;
        }
        if (!condition_0) {
            uniform_address_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));
            matrix_fragment_5 = packed_half_118 + uniform_address_word_8;
            packed_half_8 = uniform_address_word_5;
            matrix_fragment_7 = uniform_address_word_5;
            matrix_fragment_4 = address_word_56 * 0x10u + matrix_fragment_5;
            packed_half_8 = permute_signed(packed_half_8, matrix_fragment_7, 0x5410u);
            packed_half_9 = packed_half_8;
            packed_half_10 = packed_half_8;
            address_word_11 = packed_half_8;
            store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_4, packed_half_8,
                                                 packed_half_9, packed_half_10, address_word_11,
                                                 shared_bytes, memory_ranges, error_counts);
        } else {
            condition_2 = int(threadIdx.x) == __ffs(__activemask()) - 1;
            if (condition_2) {
                check_single_lane<CheckBounds>(error_counts);
                uniform_address_word_6 = packed_half_118;
                uniform_address_word_5 = 0x20u;
                // Later control flow also consumes this address-overflow predicate.
                condition_2 = (uint64_t(matrix_fragment_4) << 2) + packed_half_116 > 0xffffffffull;
                const uint64_t byte_address_1 =
                    scaled_word_address(combine_words(packed_half_116, packed_half_117),
                                        combine_words(matrix_fragment_4, matrix_fragment_5));
                matrix_fragment_6 = uint32_t(byte_address_1);
                matrix_fragment_4 = uint32_t(byte_address_1 >> 32);
                check_single_lane<CheckBounds>(error_counts);
                uniform_address_word_14 = matrix_fragment_6;
                check_single_lane<CheckBounds>(error_counts);
                uniform_address_word_15 = matrix_fragment_4;
                matrix_fragment_4 = 0x200u;
                uniform_address_word_6 = uniform_address_word_6 + uniform_address_word_8;
#if DLSS5_ENABLE_ASYNC_COPIES
                copy_global_to_shared<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_5 * 16u, uniform_index_word_7, shared_bytes, memory_ranges,
                    error_counts);
#else
                bulk_copy<CheckBounds>(
                    shared_tile, uniform_address_word_6,
                    combine_words(uniform_address_word_14, uniform_address_word_15),
                    uniform_address_word_5 * 16u, shared_bytes, memory_ranges, error_counts);
#endif
            }
        }
        matrix_fragment_4 = uniform_index_word_11;
        matrix_fragment_5 = matrix_fragment_4 * 0x80u + address_word_35;
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_114, packed_half_115));
        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_72, matrix_fragment_73),
                                            matrix_fragment_68, matrix_fragment_69,
                                            matrix_fragment_70, matrix_fragment_71, memory_ranges,
                                            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x400ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, packed_half_67, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x600ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x800ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0xa00ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0xc00ull, packed_half_36,
            address_word_37, address_word_38, address_word_39, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0xe00ull, packed_half_28,
            packed_half_29, packed_half_30, packed_half_31, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4000ull, address_word_24,
            address_word_25, address_word_26, address_word_27, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4200ull, packed_half_100,
            matrix_fragment_101, matrix_fragment_102, matrix_fragment_103, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4400ull, packed_half_8,
            packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4600ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4800ull, matrix_fragment_128,
            matrix_fragment_129, matrix_fragment_130, matrix_fragment_131, memory_ranges,
            error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4a00ull, packed_half_16,
            matrix_fragment_17, packed_half_18, matrix_fragment_19, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4c00ull, matrix_fragment_20,
            packed_half_21, packed_half_22, packed_half_23, memory_ranges, error_counts);
        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x4e00ull, matrix_fragment_40,
            matrix_fragment_41, packed_half_42, matrix_fragment_43, memory_ranges, error_counts);
#if DLSS5_ENABLE_ASYNC_COPIES
        {
            const uint32_t operand_a = uniform_index_word_7;
            async_tokens[operand_a >> 3 & 1] = arrive_copy_barrier(shared_tile, operand_a);
            split_address_words(matrix_fragment_74, matrix_fragment_75,
                                async_tokens[operand_a >> 3 & 1]);
        }
#else
        matrix_fragment_74 = 0u;
#endif
#if DLSS5_ENABLE_ASYNC_COPIES
        condition_2 = wait_copy_barrier(shared_tile, uniform_index_word_7,
                                        async_tokens[(((uniform_index_word_7) >> 3) & 1)]);
#else
        phoenix_architecture::finish_compatibility_copies();
        condition_2 = true;
#endif
        if (!condition_2) {
            do {
#if DLSS5_ENABLE_ASYNC_COPIES
                condition_2 = wait_copy_barrier(shared_tile, uniform_index_word_7,
                                                async_tokens[(((uniform_index_word_7) >> 3) & 1)]);
#else
                phoenix_architecture::finish_compatibility_copies();
                condition_2 = true;
#endif
            } while (!condition_2);
        }
    } while (!uniform_condition_0);
    packed_half_77 = (address_word_56 << 0x4u) + uniform_index_word_16;
    uniform_condition_0 = uint32_t(uniform_block_index_10) != 0u;
    load_four_shared_words<CheckBounds>(shared_tile, packed_half_77 + 0x400u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);
    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_106,
                             matrix_accumulator_107);
    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_104,
                             matrix_accumulator_105);
    load_four_shared_words<CheckBounds>(shared_tile, packed_half_77 + 0x600u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);
    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_112,
                             matrix_accumulator_113);
    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_108,
                             matrix_accumulator_109);
    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_110,
                             matrix_accumulator_111);
    multiply_accumulate_half(matrix_fragment_66, packed_half_67, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_66, packed_half_67, matrix_accumulator_98,
                             matrix_accumulator_99);
    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_78,
                             matrix_accumulator_79);
    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_80,
                             matrix_accumulator_81);
    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_82,
                             matrix_accumulator_83);
    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_84,
                             matrix_accumulator_85);
    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_86,
                             matrix_accumulator_87);
    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_88,
                             matrix_accumulator_89);
    multiply_accumulate_half(packed_half_36, address_word_37, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             packed_half_36, address_word_37, matrix_accumulator_90,
                             matrix_accumulator_91);
    multiply_accumulate_half(address_word_38, address_word_39, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             address_word_38, address_word_39, matrix_accumulator_92,
                             matrix_accumulator_93);
    multiply_accumulate_half(packed_half_28, packed_half_29, matrix_fragment_72, matrix_fragment_73,
                             matrix_fragment_74, matrix_fragment_75, packed_half_28, packed_half_29,
                             matrix_accumulator_94, matrix_accumulator_95);
    multiply_accumulate_half(packed_half_30, packed_half_31, matrix_fragment_72, matrix_fragment_73,
                             matrix_fragment_74, matrix_fragment_75, packed_half_30, packed_half_31,
                             matrix_accumulator_96, matrix_accumulator_97);
    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             address_word_24, address_word_25, matrix_accumulator_120,
                             matrix_accumulator_121);
    address_word_24 = parameters[parameter_counter_low_8];
    address_word_25 = parameters[parameter_counter_high_9];
    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             address_word_26, address_word_27, matrix_accumulator_122,
                             matrix_accumulator_123);
    multiply_accumulate_half(packed_half_100, matrix_fragment_101, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_100, matrix_fragment_101, matrix_accumulator_106,
                             matrix_accumulator_107);
    multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_104,
                             matrix_accumulator_105);
    multiply_accumulate_half(packed_half_8, packed_half_9, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_8, packed_half_9,
                             matrix_fragment_64, matrix_fragment_65);
    multiply_accumulate_half(packed_half_10, address_word_11, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_10, address_word_11,
                             matrix_fragment_66, packed_half_67);
    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                             matrix_fragment_61);
    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_62,
                             matrix_fragment_63);
    multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_128, matrix_fragment_129, matrix_fragment_52,
                             matrix_fragment_53);
    multiply_accumulate_half(matrix_fragment_130, matrix_fragment_131, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_130, matrix_fragment_131, matrix_fragment_54,
                             matrix_fragment_55);
    multiply_accumulate_half(packed_half_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_16, matrix_fragment_17, matrix_fragment_44,
                             matrix_fragment_45);
    multiply_accumulate_half(packed_half_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_18, matrix_fragment_19, matrix_fragment_46,
                             matrix_fragment_47);
    multiply_accumulate_half(matrix_fragment_20, packed_half_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, packed_half_21, packed_half_36, address_word_37);
    multiply_accumulate_half(packed_half_22, packed_half_23, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_22, packed_half_23,
                             address_word_38, address_word_39);
    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, packed_half_28,
                             packed_half_29);
    multiply_accumulate_half(packed_half_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_42, matrix_fragment_43, packed_half_30, packed_half_31);
    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));
    if (!uniform_condition_0) {
        condition_0 = int32_t(packed_half_34) >= int32_t(packed_half_51);
        condition_0 = int32_t(address_word_58) >= int32_t(packed_half_57) || condition_0;
        if (!condition_0) {
            matrix_fragment_5 = threadIdx.x & 31u;
            address_word_2 = parameters[parameter_scratch_secondary_low_12];
            address_word_3 = parameters[parameter_scratch_secondary_high_13];
            packed_half_34 = packed_half_34 + packed_half_49;
            address_word_35 = packed_half_34 * 0x1000u + address_word_35;
            split_address_words(address_word_2, address_word_3,
                                int64_t(int32_t(address_word_35)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_2, address_word_3));
            split_address_words(address_word_2, address_word_3,
                                int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x10u)) +
                                    combine_words(address_word_2, address_word_3));
            store_four_words_streaming<CheckBounds>(combine_words(address_word_2, address_word_3),
                                                    matrix_accumulator_120, matrix_accumulator_121,
                                                    matrix_accumulator_122, matrix_accumulator_123,
                                                    memory_ranges, error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0x200ull, packed_half_100,
                matrix_fragment_101, matrix_fragment_102, matrix_fragment_103, memory_ranges,
                error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0x400ull, packed_half_8,
                packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0x600ull, matrix_fragment_12,
                matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
                error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0x800ull, matrix_fragment_128,
                matrix_fragment_129, matrix_fragment_130, matrix_fragment_131, memory_ranges,
                error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0xa00ull, packed_half_16,
                matrix_fragment_17, packed_half_18, matrix_fragment_19, memory_ranges,
                error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0xc00ull, matrix_fragment_20,
                packed_half_21, packed_half_22, packed_half_23, memory_ranges, error_counts);
            store_four_words_streaming<CheckBounds>(
                combine_words(address_word_2, address_word_3) + 0xe00ull, matrix_fragment_40,
                matrix_fragment_41, packed_half_42, matrix_fragment_43, memory_ranges,
                error_counts);
        }
    } else {
        matrix_fragment_6 = threadIdx.z;
        condition_1 = (matrix_fragment_6 | packed_half_59) != 0;
        if (!condition_1) {
            matrix_fragment_7 = read_device_flag<CheckBounds>(
                combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges, error_counts);
            packed_half_76 = packed_half_76 + 0xffffffffu;
            condition_1 = int32_t(matrix_fragment_7) >= int32_t(packed_half_76);
            if (!condition_1) {
                do {
                    __nanosleep(0x40u);
                    matrix_fragment_7 = read_device_flag<CheckBounds>(
                        combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges,
                        error_counts);
                    condition_1 = int32_t(matrix_fragment_7) >= int32_t(packed_half_76);
                } while (!condition_1);
            }
        }
        __syncwarp(__activemask());
        phoenix_architecture::finish_compatibility_copies();
        uniform_condition_0 = uint32_t(uniform_block_index_10) >= uint32_t(0x3u);
        if (!uniform_condition_0) {
            condition_0 = int32_t(packed_half_34) >= int32_t(packed_half_51);
            condition_0 = int32_t(address_word_58) >= int32_t(packed_half_57) || condition_0;
            if (!condition_0) {
                address_word_37 = threadIdx.x & 31u;
                address_word_2 = packed_half_34 + packed_half_49;
                uniform_index_word_4 = parameters[parameter_scratch_secondary_low_12];
                uniform_address_word_5 = parameters[parameter_scratch_secondary_high_13];
                address_word_2 = address_word_2 * 0x1000u + address_word_35;
                matrix_fragment_4 = address_word_2 + 0x80u;
                matrix_fragment_6 = address_word_2 + 0x100u;
                address_word_24 = address_word_2 + 0x180u;
                address_word_3 = uint32_t(int32_t(address_word_2) >> 0x1fu);
                address_word_26 = address_word_2 + 0x200u;
                packed_half_28 = address_word_2 + 0x280u;
                matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);
                packed_half_30 = address_word_2 + 0x300u;
                matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);
                address_word_32 = address_word_2 + 0x380u;
                address_word_25 = uint32_t(int32_t(address_word_24) >> 0x1fu);
                address_word_27 = uint32_t(int32_t(address_word_26) >> 0x1fu);
                packed_half_29 = uint32_t(int32_t(packed_half_28) >> 0x1fu);
                packed_half_31 = uint32_t(int32_t(packed_half_30) >> 0x1fu);
                split_address_words(address_word_2, address_word_3,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_2, address_word_3));
                address_word_33 = uint32_t(int32_t(address_word_32) >> 0x1fu);
                split_address_words(matrix_fragment_4, matrix_fragment_5,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_4, matrix_fragment_5));
                const uint64_t byte_address_2 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_2, address_word_3));
                packed_half_34 = uint32_t(byte_address_2);
                split_address_words(matrix_fragment_6, matrix_fragment_7,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_6, matrix_fragment_7));
                const uint64_t byte_address_3 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_4, matrix_fragment_5));
                packed_half_36 = uint32_t(byte_address_3);
                address_word_35 = uint32_t(byte_address_2 >> 32);
                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_24, address_word_25));
                const uint64_t byte_address_4 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(matrix_fragment_6, matrix_fragment_7));
                address_word_38 = uint32_t(byte_address_4);
                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_26, address_word_27));
                const uint64_t byte_address_5 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_24, address_word_25));
                matrix_fragment_44 = uint32_t(byte_address_5);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(packed_half_34, address_word_35), matrix_accumulator_120,
                    matrix_accumulator_121, matrix_accumulator_122, matrix_accumulator_123,
                    memory_ranges, error_counts);
                address_word_39 = uint32_t(byte_address_4 >> 32);
                split_address_words(packed_half_28, packed_half_29,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_28, packed_half_29));
                const uint64_t byte_address_6 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_26, address_word_27));
                address_word_2 = uint32_t(byte_address_6);
                matrix_fragment_45 = uint32_t(byte_address_5 >> 32);
                split_address_words(packed_half_30, packed_half_31,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_30, packed_half_31));
                address_word_3 = uint32_t(byte_address_6 >> 32);
                split_address_words(address_word_32, address_word_33,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_32, address_word_33));
                address_word_37 = uint32_t(byte_address_3 >> 32);
                const uint64_t byte_address_7 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(packed_half_28, packed_half_29));
                matrix_fragment_4 = uint32_t(byte_address_7);
                const uint64_t byte_address_8 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(packed_half_30, packed_half_31));
                matrix_fragment_6 = uint32_t(byte_address_8);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(packed_half_36, address_word_37), packed_half_100,
                    matrix_fragment_101, matrix_fragment_102, matrix_fragment_103, memory_ranges,
                    error_counts);
                const uint64_t byte_address_9 =
                    scaled_word_address(combine_words(uniform_index_word_4, uniform_address_word_5),
                                        combine_words(address_word_32, address_word_33));
                address_word_24 = uint32_t(byte_address_9);
                matrix_fragment_5 = uint32_t(byte_address_7 >> 32);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_38, address_word_39), packed_half_8, packed_half_9,
                    packed_half_10, address_word_11, memory_ranges, error_counts);
                matrix_fragment_7 = uint32_t(byte_address_8 >> 32);
                address_word_25 = uint32_t(byte_address_9 >> 32);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_44, matrix_fragment_45), matrix_fragment_12,
                    matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
                    error_counts);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_2, address_word_3), matrix_fragment_128,
                    matrix_fragment_129, matrix_fragment_130, matrix_fragment_131, memory_ranges,
                    error_counts);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_4, matrix_fragment_5), packed_half_16,
                    matrix_fragment_17, packed_half_18, matrix_fragment_19, memory_ranges,
                    error_counts);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(matrix_fragment_6, matrix_fragment_7), matrix_fragment_20,
                    packed_half_21, packed_half_22, packed_half_23, memory_ranges, error_counts);
                atomic_add_four_half_pairs<CheckBounds>(
                    combine_words(address_word_24, address_word_25), matrix_fragment_40,
                    matrix_fragment_41, packed_half_42, matrix_fragment_43, memory_ranges,
                    error_counts);
            }
        } else {
            if (condition_0) {
                address_word_24 = parameters[parameter_scratch_secondary_low_12];
                address_word_25 = parameters[parameter_scratch_secondary_high_13];

                matrix_fragment_4 = address_word_3 + packed_half_50;

                matrix_fragment_6 = address_word_3 + packed_half_50;

                address_word_37 = address_word_3 + packed_half_50;

                address_word_38 = address_word_3 + packed_half_50;
            }
            matrix_fragment_44 = 0;
            matrix_fragment_45 = 0;
            if (condition_0) {
                matrix_fragment_7 = matrix_fragment_4 * 0x200u + address_word_33;

                matrix_fragment_4 = address_word_35 + 0x100u;

                packed_half_28 = parameters[parameter_scratch_secondary_low_12];
                packed_half_29 = parameters[parameter_scratch_secondary_high_13];

                matrix_fragment_5 = matrix_fragment_6 * 0x1000u + address_word_35;
            }
            matrix_fragment_46 = 0;
            matrix_fragment_47 = 0;
            if (condition_0) {
                matrix_fragment_7 = matrix_fragment_7 * 0x8u;

                address_word_37 = address_word_37 * 0x1000u + matrix_fragment_4;

                address_word_26 = parameters[parameter_scratch_secondary_low_12];
                address_word_27 = parameters[parameter_scratch_secondary_high_13];

                packed_half_30 = parameters[parameter_scratch_secondary_low_12];
                packed_half_31 = parameters[parameter_scratch_secondary_high_13];

                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_24, address_word_25));
            }
            matrix_fragment_4 = 0;
            matrix_fragment_5 = 0;
            matrix_fragment_52 = 0;
            matrix_fragment_53 = 0;
            matrix_fragment_54 = 0;
            matrix_fragment_55 = 0;
            if (condition_0) {
                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_24, address_word_25));

                address_word_58 = parameters[parameter_scratch_secondary_low_12];
                packed_half_59 = parameters[parameter_scratch_secondary_high_13];

                split_address_words(packed_half_28, packed_half_29,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_28, packed_half_29));

                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_26, address_word_27));
            }
            matrix_fragment_6 = 0;
            matrix_fragment_7 = 0;
            if (condition_0) {
                packed_half_49 = address_word_3 + packed_half_50;

                packed_half_36 = address_word_35 + 0x200u;

                address_word_37 = address_word_38 * 0x200u + address_word_33;

                matrix_fragment_62 = parameters[parameter_scratch_secondary_low_12];
                matrix_fragment_63 = parameters[parameter_scratch_secondary_high_13];

                split_address_words(packed_half_28, packed_half_29,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_28, packed_half_29));

                load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25),
                                                    matrix_fragment_4, matrix_fragment_5,
                                                    matrix_fragment_6, matrix_fragment_7,
                                                    memory_ranges, error_counts);

                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_26, address_word_27));

                address_word_37 = (address_word_37 << 0x3u) + 0x100u;

                load_four_words_cached<CheckBounds>(combine_words(packed_half_28, packed_half_29),
                                                    matrix_fragment_52, matrix_fragment_53,
                                                    matrix_fragment_54, matrix_fragment_55,
                                                    memory_ranges, error_counts);
            }
            packed_half_132 = 0;
            packed_half_133 = 0;
            packed_half_134 = 0;
            packed_half_135 = 0;
            matrix_fragment_66 = parameters[parameter_tensor_height_18];
            packed_half_67 = parameters[parameter_tensor_width_19];
            if (condition_0) {
                load_four_words_cached<CheckBounds>(combine_words(address_word_26, address_word_27),
                                                    matrix_fragment_44, matrix_fragment_45,
                                                    matrix_fragment_46, matrix_fragment_47,
                                                    memory_ranges, error_counts);

                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_30, packed_half_31));
            }
            address_word_38 = 0;
            address_word_39 = 0;
            if (condition_0) {
                packed_half_49 = packed_half_49 * 0x1000u + packed_half_36;
            }
            packed_half_36 = 0;
            address_word_37 = 0;
            if (condition_0) {
                packed_half_30 = parameters[parameter_scratch_secondary_low_12];
                packed_half_31 = parameters[parameter_scratch_secondary_high_13];

                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_24, address_word_25));

                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(packed_half_49)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_58, packed_half_59));

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_24, address_word_25), packed_half_36,
                    address_word_37, address_word_38, address_word_39, memory_ranges, error_counts);

                address_word_58 = address_word_3 + packed_half_50;

                packed_half_29 = address_word_58 * 0x200u + address_word_33;

                address_word_35 = address_word_35 + 0x300u;

                packed_half_28 = address_word_3 + packed_half_50;

                packed_half_29 = (packed_half_29 << 0x3u) + 0x200u;

                address_word_35 = packed_half_28 * 0x1000u + address_word_35;

                split_address_words(address_word_24, address_word_25,
                                    int64_t(int32_t(packed_half_29)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_30, packed_half_31));
            }
            packed_half_28 = 0;
            packed_half_29 = 0;
            packed_half_30 = 0;
            packed_half_31 = 0;
            if (condition_0) {
                split_address_words(address_word_58, packed_half_59,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_26, address_word_27));

                split_address_words(matrix_fragment_60, matrix_fragment_61,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_24, address_word_25));

                load_four_words_cached<CheckBounds>(
                    combine_words(address_word_58, packed_half_59), packed_half_132,
                    packed_half_133, packed_half_134, packed_half_135, memory_ranges, error_counts);

                split_address_words(matrix_fragment_62, matrix_fragment_63,
                                    int64_t(int32_t(address_word_35)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_62, matrix_fragment_63));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_60, matrix_fragment_61), packed_half_28,
                    packed_half_29, packed_half_30, packed_half_31, memory_ranges, error_counts);
            }
            address_word_24 = 0;
            address_word_25 = 0;
            address_word_26 = 0;
            address_word_27 = 0;
            if (condition_0) {
                split_address_words(matrix_fragment_62, matrix_fragment_63,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_62, matrix_fragment_63));

                address_word_58 = parameters[parameter_scratch_secondary_low_12];
                packed_half_59 = parameters[parameter_scratch_secondary_high_13];

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_62, matrix_fragment_63), address_word_24,
                    address_word_25, address_word_26, address_word_27, memory_ranges, error_counts);
            }
            address_word_35 = uint32_t(uint32_t(address_word_56) >> 0x1u);
            matrix_fragment_64 = address_word_56 & 0x3u;
            if (condition_0) {
                packed_half_50 = address_word_3 + packed_half_50;
            }
            address_word_3 = matrix_fragment_64 | 0x4u & address_word_35;
            if (condition_0) {
                address_word_35 = packed_half_50 * 0x200u + address_word_33;

                address_word_35 = (address_word_35 << 0x3u) + 0x300u;
            }
            packed_half_50 = 0;
            packed_half_51 = 0;
            packed_half_49 = 0u;
            matrix_accumulator_91 = blockIdx.y;
            address_word_125 = threadIdx.x & 31u;
            matrix_fragment_62 = address_word_2 + 0x38u;
            matrix_fragment_64 = address_word_2 + 0x48u;
            matrix_accumulator_78 = address_word_2 + 0x50u;
            packed_half_76 = address_word_2 + 0x60u;
            matrix_accumulator_122 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                               select_half_operand<2, false, false>(matrix_accumulator_122));
            matrix_accumulator_106 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                               select_half_operand<2, false, false>(matrix_accumulator_120));
            packed_half_114 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                               select_half_operand<2, false, false>(packed_half_8));
            packed_half_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                           select_half_operand<2, false, false>(packed_half_9));
            matrix_fragment_6 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                               select_half_operand<2, false, false>(matrix_fragment_101));
            matrix_fragment_45 = uint32_t(int32_t(matrix_fragment_66) >> 0x1fu);
            matrix_fragment_4 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                               select_half_operand<2, false, false>(matrix_accumulator_121));
            packed_half_116 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                               select_half_operand<2, false, false>(packed_half_10));
            packed_half_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                           select_half_operand<2, false, false>(address_word_11));
            if (condition_0) {
                split_address_words(packed_half_10, address_word_11,
                                    int64_t(int32_t(address_word_35)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_58, packed_half_59));
            }
            matrix_fragment_5 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                               select_half_operand<2, false, false>(matrix_accumulator_123));
            matrix_fragment_7 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                               select_half_operand<2, false, false>(matrix_fragment_103));
            matrix_fragment_47 = matrix_fragment_66 & 0xfffffffcu;
            address_word_35 =
                uint32_t(combine_words(matrix_fragment_45, 0u) << 0x2u >> 32) + matrix_fragment_66;
            matrix_accumulator_104 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                               select_half_operand<2, false, false>(packed_half_100));
            if (condition_0) {
                split_address_words(matrix_fragment_44, matrix_fragment_45,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_10, address_word_11));
            }
            condition_4 = uint32_t(matrix_fragment_47) != uint32_t(0x4u);
            address_word_35 = uint32_t(int32_t(address_word_35) >> 0x2u);
            packed_half_10 = uint32_t(int32_t(packed_half_67) >> 0x1fu);
            packed_half_118 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_36),
                               select_half_operand<2, false, false>(matrix_fragment_12));
            if (condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_44, matrix_fragment_45), packed_half_48,
                    packed_half_49, packed_half_50, packed_half_51, memory_ranges, error_counts);
            }
            packed_half_36 = packed_half_67 & 0xfffffffcu;
            condition_0 = int32_t(address_word_32) >= int32_t(address_word_35) && condition_4;
            matrix_fragment_12 =
                uint32_t(combine_words(packed_half_10, 0u) << 0x2u >> 32) + packed_half_67;
            matrix_fragment_102 =
                add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                               select_half_operand<2, false, false>(matrix_fragment_102));
            address_word_11 = packed_half_34 * 0x2u;
            condition_6 = uint32_t(matrix_fragment_47) != uint32_t(0x4u);
            matrix_fragment_46 = uint32_t(int32_t(matrix_fragment_12) >> 0x2u);
            condition_3 = uint32_t(packed_half_36) == uint32_t(0x4u) || condition_0;
            condition_1 = int32_t(address_word_32) < int32_t(address_word_35) || !condition_6;
            condition_2 = int32_t(address_word_11) < int32_t(matrix_fragment_46) || condition_3;
            matrix_fragment_12 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_39),
                               select_half_operand<2, false, false>(matrix_fragment_15));
            condition_2 = condition_2 & condition_1;
            address_word_39 = address_word_11 + 0x1u;
            condition_0 = uint32_t(packed_half_36) == uint32_t(0x4u) && !condition_0;
            packed_half_134 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                               select_half_operand<2, false, false>(matrix_fragment_130));
            condition_3 = int32_t(address_word_39) < int32_t(matrix_fragment_46) || condition_3;
            matrix_fragment_128 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_132),
                               select_half_operand<2, false, false>(matrix_fragment_128));
            matrix_fragment_63 = !condition_0 ? address_word_11 : 0u;
            matrix_fragment_130 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_28),
                               select_half_operand<2, false, false>(packed_half_16));
            matrix_fragment_15 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                               select_half_operand<2, false, false>(matrix_fragment_17));
            condition_1 = condition_3 & condition_1;
            packed_half_132 = add_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                             select_half_operand<2, false, false>(packed_half_18));
            if (condition_2) {
                packed_half_28 = parameters[parameter_input_encoder_skip_low_2];
                packed_half_29 = parameters[parameter_input_encoder_skip_high_3];
            }
            packed_half_18 = address_word_32 * matrix_fragment_46;
            condition_0 = uint32_t(matrix_fragment_47) != uint32_t(0x4u) && condition_2;
            condition_3 = uint32_t(matrix_fragment_47) != uint32_t(0x4u) && condition_2;
            packed_half_10 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_37),
                               select_half_operand<2, false, false>(matrix_fragment_13));
            address_word_37 = matrix_fragment_63;
            packed_half_126 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_24),
                               select_half_operand<2, false, false>(matrix_fragment_20));
            address_word_24 = condition_6 ? packed_half_18 : 0u;
            packed_half_16 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_31),
                               select_half_operand<2, false, false>(matrix_fragment_19));
            matrix_fragment_17 = matrix_accumulator_91 * 0x2u;
            packed_half_18 = add_half_pairs(select_half_operand<2, false, false>(address_word_25),
                                            select_half_operand<2, false, false>(packed_half_21));
            if (condition_2) {
                address_word_32 = address_word_32 * matrix_fragment_46 + address_word_37;
            }
            matrix_fragment_19 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_26),
                               select_half_operand<2, false, false>(packed_half_22));
            if (condition_2) {
                packed_half_21 = address_word_24 + address_word_37;

                packed_half_22 = matrix_fragment_46 * matrix_accumulator_91;
            }
            if (condition_1) {
                condition_5 = int32_t(matrix_fragment_17) >= int32_t(address_word_35);
            }
            if (condition_2) {
                address_word_32 = !condition_6 ? address_word_37 : address_word_32;
            }
            matrix_accumulator_120 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_38),
                               select_half_operand<2, false, false>(matrix_fragment_14));
            if (condition_0) {
                matrix_fragment_63 = packed_half_22 * 0x2u + matrix_fragment_63;
            }
            packed_half_34 = address_word_2 + 0x28u;
            if (condition_3) {
                address_word_37 = packed_half_22 * 0x2u + address_word_37;
            }
            packed_half_22 = address_word_125;
            if (condition_2) {
                packed_half_21 = packed_half_21 * 0x200u + address_word_2;
            }
            matrix_fragment_20 =
                add_half_pairs(select_half_operand<2, false, false>(address_word_27),
                               select_half_operand<2, false, false>(packed_half_23));
            if (condition_1) {
                packed_half_23 = condition_5 ? address_word_39 : 0u;
            }
            packed_half_22 = packed_half_22 & 0x7ffffffcu;
            if (condition_2) {
                address_word_27 = packed_half_21 * 0x4u;
            }
            if (condition_1) {
                condition_5 = uint32_t(packed_half_36) != uint32_t(0x4u);
            }
            matrix_fragment_66 = address_word_2 + 0x30u;
            if (condition_1) {
                address_word_24 = condition_4 ? packed_half_23 : 0u;
            }
            if (condition_2) {
                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(address_word_27)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_28, packed_half_29));
            }
            packed_half_29 = 0u - packed_half_22 + address_word_125;
            matrix_fragment_44 = address_word_2 + 0x58u;
            packed_half_22 = parameters[parameter_trained_weights_low_14];
            packed_half_23 = parameters[parameter_trained_weights_high_15];
            if (condition_1) {
                packed_half_21 = !condition_5 ? address_word_24 : address_word_39;
            }
            if (condition_2) {
                matrix_fragment_60 = address_word_32 * 0x200u + address_word_33;
            }
            packed_half_28 = address_word_2 + 0x8u;
            packed_half_30 = address_word_2 + 0x10u;
            address_word_32 = address_word_2 + 0x18u;
            packed_half_36 = address_word_2 + 0x20u;
            address_word_24 = parameters[parameter_tensor_height_18];
            address_word_25 = parameters[parameter_tensor_width_19];
            address_word_38 = address_word_2 + 0x40u;
            matrix_fragment_52 = address_word_2 + 0x68u;
            packed_half_100 = address_word_2 + 0x70u;
            address_word_58 = address_word_2 + 0x78u;
            if (!condition_2) {
                packed_half_31 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
            }
            packed_half_28 = packed_half_29 * 0x2u + packed_half_28;
            condition_0 = uint32_t(matrix_fragment_47) != uint32_t(0x4u) && condition_1;
            packed_half_30 = packed_half_29 * 0x2u + packed_half_30;
            matrix_accumulator_95 = address_word_11 + 0x1u;
            matrix_fragment_54 = packed_half_29 * 0x2u + address_word_32;
            if (condition_1) {
                matrix_fragment_74 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_75 = parameters[parameter_input_encoder_skip_high_3];
            }
            packed_half_36 = packed_half_29 * 0x2u + packed_half_36;
            packed_half_67 = packed_half_29 * 0x2u + packed_half_34;
            matrix_fragment_66 = packed_half_29 * 0x2u + matrix_fragment_66;
            matrix_fragment_62 = packed_half_29 * 0x2u + matrix_fragment_62;
            matrix_fragment_65 = packed_half_29 * 0x2u + address_word_38;
            matrix_fragment_64 = packed_half_29 * 0x2u + matrix_fragment_64;
            matrix_accumulator_78 = packed_half_29 * 0x2u + matrix_accumulator_78;
            packed_half_57 = packed_half_29 * 0x2u + matrix_fragment_44;
            packed_half_76 = packed_half_29 * 0x2u + packed_half_76;
            packed_half_77 = packed_half_29 * 0x2u + matrix_fragment_52;
            packed_half_100 = packed_half_29 * 0x2u + packed_half_100;
            matrix_fragment_101 = packed_half_29 * 0x2u + address_word_58;
            if (!condition_2) {
                packed_half_31 = permute_signed(packed_half_31, packed_half_31, 0x5410u);
            }
            packed_half_29 = packed_half_29 * 0x2u + address_word_2;
            matrix_fragment_47 = uint32_t(int32_t(packed_half_28) >> 0x1u);
            if (condition_1) {
                packed_half_34 = parameters[parameter_input_encoder_skip_low_2];
                address_word_35 = parameters[parameter_input_encoder_skip_high_3];
            }
            matrix_fragment_53 = uint32_t(int32_t(packed_half_30) >> 0x1u);
            matrix_fragment_45 = uint32_t(int32_t(packed_half_29) >> 0x1u);
            if (condition_2) {
                split_address_words(address_word_38, address_word_39,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_26, address_word_27));
            }
            if (!condition_2) {
                packed_half_28 = packed_half_31;

                packed_half_29 = packed_half_31;

                packed_half_30 = packed_half_31;
            }
            if (condition_0) {
                packed_half_21 = matrix_fragment_17 * matrix_fragment_46 + packed_half_21;
            }
            matrix_fragment_61 = uint32_t(int32_t(address_word_24) >> 0x1fu);
            split_address_words(matrix_fragment_46, matrix_fragment_47,
                                int64_t(int32_t(matrix_fragment_47)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_22, packed_half_23));
            address_word_32 = parameters[parameter_trained_weights_low_14];
            address_word_33 = parameters[parameter_trained_weights_high_15];
            if (condition_2) {
                load_four_words_cached<CheckBounds>(combine_words(address_word_38, address_word_39),
                                                    packed_half_28, packed_half_29, packed_half_30,
                                                    packed_half_31, memory_ranges, error_counts);
            }
            if (condition_1) {
                matrix_fragment_55 = packed_half_21 * 0x200u + address_word_2;
            }
            packed_half_21 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_46, matrix_fragment_47) + 0x80000ull, memory_ranges,
                error_counts);
            matrix_fragment_61 =
                uint32_t(combine_words(matrix_fragment_61, 0u) << 0x2u >> 32) + address_word_24;
            split_address_words(matrix_fragment_44, matrix_fragment_45,
                                int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_22, packed_half_23));
            if (condition_2) {
                address_word_26 = parameters[parameter_input_encoder_skip_low_2];
                address_word_27 = parameters[parameter_input_encoder_skip_high_3];
            }
            matrix_accumulator_80 = uint32_t(int32_t(matrix_fragment_61) >> 0x2u);
            split_address_words(matrix_fragment_52, matrix_fragment_53,
                                int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_22, packed_half_23));
            if (!condition_1) {
                address_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
            }
            packed_half_22 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_44, matrix_fragment_45) + 0x80000ull, memory_ranges,
                error_counts);
            if (condition_1) {
                matrix_fragment_55 = matrix_fragment_55 * 0x4u;
            }
            matrix_fragment_54 =
                uint32_t(combine_words(matrix_fragment_54, 0u) << 0x1u >> 32) + matrix_fragment_54;
            packed_half_23 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_52, matrix_fragment_53) + 0x80000ull, memory_ranges,
                error_counts);
            if (condition_1) {
                condition_0 = int32_t(matrix_fragment_17) >= int32_t(matrix_accumulator_80);

                split_address_words(packed_half_34, address_word_35,
                                    int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_34, address_word_35));
            }
            if (!condition_1) {
                matrix_fragment_47 = permute_signed(address_word_38, address_word_38, 0x5410u);
            }
            packed_half_59 = uint32_t(int32_t(matrix_fragment_54) >> 0x1u);
            matrix_fragment_54 = address_word_24 & 0xfffffffcu;
            matrix_accumulator_92 = address_word_25 & 0xfffffffcu;
            if (condition_1) {
                address_word_24 = condition_0 ? matrix_accumulator_95 : 0u;
            }
            if (!condition_1) {
                matrix_fragment_44 = matrix_fragment_47;

                matrix_fragment_45 = matrix_fragment_47;

                matrix_fragment_46 = matrix_fragment_47;
            }
            if (condition_1) {
                condition_5 = int32_t(matrix_fragment_17) >= int32_t(matrix_accumulator_80);
            }
            address_word_58 = uint32_t(int32_t(address_word_25) >> 0x1fu);
            if (condition_1) {
                split_address_words(packed_half_34, address_word_35,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_34, address_word_35));

                condition_0 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);

                address_word_38 = condition_4 ? address_word_24 : 0u;
            }
            address_word_58 =
                uint32_t(combine_words(address_word_58, 0u) << 0x2u >> 32) + address_word_25;
            if (condition_1) {
                address_word_39 = condition_5 ? matrix_accumulator_95 : 0u;

                load_four_words_cached<CheckBounds>(combine_words(packed_half_34, address_word_35),
                                                    matrix_fragment_44, matrix_fragment_45,
                                                    matrix_fragment_46, matrix_fragment_47,
                                                    memory_ranges, error_counts);
            }
            if (condition_2) {
                address_word_25 = (matrix_fragment_60 << 0x2u) + 0x40u;
            }
            split_address_words(address_word_32, address_word_33,
                                int64_t(int32_t(packed_half_59)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_32, address_word_33));
            condition_3 = uint32_t(matrix_fragment_54) != uint32_t(0x4u) && condition_1;
            if (condition_1) {
                matrix_fragment_55 = !condition_0 ? address_word_38 : matrix_accumulator_95;

                condition_0 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);
            }
            if (condition_2) {
                split_address_words(address_word_26, address_word_27,
                                    int64_t(int32_t(address_word_25)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_26, address_word_27));
            }
            address_word_24 = load_aligned_word<CheckBounds>(
                combine_words(address_word_32, address_word_33) + 0x80000ull, memory_ranges,
                error_counts);
            if (!condition_2) {
                address_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
            }
            if (condition_2) {
                matrix_fragment_60 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_61 = parameters[parameter_input_encoder_skip_high_3];
            }
            if (condition_1) {
                packed_half_34 = condition_4 ? address_word_39 : 0u;
            }
            if (!condition_2) {
                address_word_35 = permute_signed(address_word_35, address_word_35, 0x5410u);
            }
            if (condition_2) {
                split_address_words(matrix_fragment_52, matrix_fragment_53,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(address_word_26, address_word_27));
            }
            if (condition_1) {
                matrix_accumulator_79 = !condition_0 ? packed_half_34 : matrix_accumulator_95;
            }
            matrix_accumulator_94 = uint32_t(int32_t(address_word_58) >> 0x2u);
            if (condition_1) {
                address_word_38 = parameters[parameter_input_encoder_skip_low_2];
                address_word_39 = parameters[parameter_input_encoder_skip_high_3];
            }
            if (!condition_2) {
                address_word_32 = address_word_35;

                address_word_33 = address_word_35;

                packed_half_34 = address_word_35;
            }
            address_word_25 = (matrix_accumulator_91 << 0x1u) + 0x1u;
            if (condition_3) {
                matrix_fragment_55 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_fragment_55;
            }
            if (condition_1) {
                condition_0 = int32_t(matrix_fragment_17) >= int32_t(matrix_accumulator_80);
            }
            address_word_58 = parameters[parameter_trained_weights_low_14];
            packed_half_59 = parameters[parameter_trained_weights_high_15];
            if (condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_52, matrix_fragment_53), address_word_32,
                    address_word_33, packed_half_34, address_word_35, memory_ranges, error_counts);
            }
            condition_5 = uint32_t(matrix_fragment_54) != uint32_t(0x4u) && condition_1;
            condition_3 = int32_t(address_word_25) >= int32_t(matrix_accumulator_80);
            if (condition_1) {
                matrix_fragment_68 = condition_0 ? matrix_accumulator_95 : 0u;
            }
            condition_0 = uint32_t(matrix_fragment_54) != uint32_t(0x4u) && condition_1;
            if (condition_1) {
                matrix_fragment_69 = matrix_fragment_55 * 0x200u + address_word_2;
            }
            condition_3 = uint32_t(matrix_fragment_54) == uint32_t(0x4u) || !condition_3;
            address_word_26 = parameters[parameter_trained_weights_low_14];
            address_word_27 = parameters[parameter_trained_weights_high_15];
            matrix_fragment_54 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_55 = parameters[parameter_trained_weights_high_15];
            if (condition_5) {
                matrix_accumulator_79 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_79;
            }
            condition_5 = int32_t(address_word_25) >= int32_t(matrix_accumulator_80) && condition_4;
            if (condition_1) {
                matrix_fragment_70 = condition_4 ? matrix_fragment_68 : 0u;
            }
            if (condition_2) {
                address_word_25 = address_word_37 * 0x200u + address_word_2;
            }
            if (condition_1) {
                condition_4 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);
            }
            matrix_fragment_71 = uint32_t(int32_t(packed_half_36) >> 0x1u);
            matrix_fragment_52 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_53 = parameters[parameter_trained_weights_high_15];
            if (condition_1) {
                matrix_fragment_69 = (matrix_fragment_69 << 0x2u) + 0x80u;
            }
            if (condition_2) {
                address_word_25 = (address_word_25 << 0x2u) + 0x100u;
            }
            if (condition_1) {
                matrix_accumulator_81 = !condition_4 ? matrix_fragment_70 : matrix_accumulator_95;
            }
            packed_half_67 =
                uint32_t(combine_words(packed_half_67, 0u) << 0x1u >> 32) + packed_half_67;
            split_address_words(matrix_fragment_70, matrix_fragment_71,
                                int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_58, packed_half_59));
            matrix_fragment_73 = uint32_t(int32_t(matrix_fragment_66) >> 0x1u);
            address_word_58 = parameters[parameter_trained_weights_low_14];
            packed_half_59 = parameters[parameter_trained_weights_high_15];
            if (condition_1) {
                split_address_words(matrix_fragment_68, matrix_fragment_69,
                                    int64_t(int32_t(matrix_fragment_69)) * int64_t(int32_t(0x4u)) +
                                        combine_words(address_word_38, address_word_39));
            }
            if (!condition_1) {
                address_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
            }
            packed_half_67 = uint32_t(int32_t(packed_half_67) >> 0x1u);
            if (condition_2) {
                split_address_words(matrix_fragment_60, matrix_fragment_61,
                                    int64_t(int32_t(address_word_25)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_60, matrix_fragment_61));
            }
            if (!condition_1) {
                address_word_39 = permute_signed(address_word_39, address_word_39, 0x5410u);
            }
            address_word_25 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_70, matrix_fragment_71) + 0x80000ull, memory_ranges,
                error_counts);
            matrix_fragment_62 =
                uint32_t(combine_words(matrix_fragment_62, 0u) << 0x1u >> 32) + matrix_fragment_62;
            split_address_words(matrix_fragment_54, matrix_fragment_55,
                                int64_t(int32_t(packed_half_67)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_54, matrix_fragment_55));
            if (!condition_1) {
                packed_half_36 = address_word_39;
            }
            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(matrix_fragment_73)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_26, address_word_27));
            if (!condition_1) {
                address_word_37 = address_word_39;

                address_word_38 = address_word_39;
            }
            matrix_accumulator_85 = uint32_t(int32_t(matrix_fragment_62) >> 0x1u);
            if (condition_1) {
                split_address_words(matrix_fragment_68, matrix_fragment_69,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_68, matrix_fragment_69));
            }
            address_word_26 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_54, matrix_fragment_55) + 0x80000ull, memory_ranges,
                error_counts);
            condition_4 = uint32_t(matrix_accumulator_92) == uint32_t(0x4u) || condition_5;
            if (condition_2) {
                matrix_fragment_66 = parameters[parameter_input_encoder_skip_low_2];
                packed_half_67 = parameters[parameter_input_encoder_skip_high_3];
            }
            address_word_27 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x80000ull, memory_ranges,
                error_counts);
            split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                                int64_t(int32_t(matrix_accumulator_85)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_52, matrix_fragment_53));
            if (condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_68, matrix_fragment_69), packed_half_36,
                    address_word_37, address_word_38, address_word_39, memory_ranges, error_counts);
            }
            matrix_accumulator_84 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_84, matrix_accumulator_85) + 0x80000ull,
                memory_ranges, error_counts);
            if (condition_0) {
                matrix_accumulator_81 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_81;
            }
            condition_0 = int32_t(address_word_11) < int32_t(matrix_accumulator_94) || condition_4;
            condition_5 = uint32_t(matrix_accumulator_92) == uint32_t(0x4u) && !condition_5;
            if (!condition_2) {
                matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));
            }
            if (condition_2) {
                matrix_fragment_63 = matrix_fragment_63 * 0x200u + address_word_2;
            }
            if (condition_1) {
                matrix_fragment_68 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_69 = parameters[parameter_input_encoder_skip_high_3];
            }
            condition_0 = condition_0 & condition_3;
            if (condition_2) {
                split_address_words(matrix_fragment_70, matrix_fragment_71,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_60, matrix_fragment_61));
            }
            if (!condition_2) {
                matrix_fragment_55 =
                    permute_signed(matrix_fragment_53, matrix_fragment_53, 0x5410u);
            }
            if (condition_0) {
                matrix_accumulator_83 = matrix_accumulator_94 * matrix_accumulator_91;
            }
            if (condition_2) {
                matrix_fragment_63 = (matrix_fragment_63 << 0x2u) + 0x180u;
            }
            if (condition_1) {
                matrix_accumulator_79 = matrix_accumulator_79 * 0x200u + address_word_2;
            }
            matrix_fragment_65 = uint32_t(int32_t(matrix_fragment_65) >> 0x1u);
            if (condition_0) {
                matrix_fragment_52 = matrix_accumulator_83 * 0x2u + matrix_accumulator_94;

                matrix_fragment_62 = !condition_5 ? address_word_11 : 0u;

                matrix_fragment_61 = matrix_accumulator_94 * matrix_accumulator_91;
            }
            matrix_accumulator_99 = uint32_t(int32_t(matrix_accumulator_78) >> 0x1u);
            if (condition_0) {
                matrix_accumulator_110 = parameters[parameter_input_encoder_skip_low_2];
                matrix_accumulator_111 = parameters[parameter_input_encoder_skip_high_3];

                matrix_fragment_73 = condition_6 ? matrix_fragment_52 : 0u;
            }
            if (!condition_2) {
                matrix_fragment_52 = matrix_fragment_55;

                matrix_fragment_53 = matrix_fragment_55;

                matrix_fragment_54 = matrix_fragment_55;
            }
            if (condition_0) {
                matrix_accumulator_93 = matrix_fragment_62 + matrix_fragment_73;

                matrix_fragment_62 = matrix_fragment_61 * 0x2u + matrix_accumulator_94;
            }
            matrix_fragment_72 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_73 = parameters[parameter_trained_weights_high_15];
            if (condition_2) {
                split_address_words(matrix_fragment_66, packed_half_67,
                                    int64_t(int32_t(matrix_fragment_63)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_66, packed_half_67));

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_70, matrix_fragment_71), matrix_fragment_52,
                    matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
                    error_counts);
            }
            if (condition_0) {
                matrix_accumulator_82 = condition_6 ? matrix_fragment_62 : 0u;
            }
            matrix_fragment_60 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_61 = parameters[parameter_trained_weights_high_15];
            if (condition_1) {
                matrix_accumulator_83 = matrix_accumulator_81 * 0x200u + address_word_2;
            }
            if (condition_2) {
                split_address_words(matrix_fragment_70, matrix_fragment_71,
                                    int64_t(int32_t(address_word_56)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_66, packed_half_67));
            }
            matrix_fragment_62 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_63 = parameters[parameter_trained_weights_high_15];
            if (condition_0) {
                matrix_accumulator_81 = !condition_5 ? address_word_11 : 0u;
            }
            if (condition_1) {
                matrix_accumulator_79 = (matrix_accumulator_79 << 0x2u) + 0x100u;

                matrix_accumulator_83 = (matrix_accumulator_83 << 0x2u) + 0x180u;
            }
            matrix_fragment_66 = parameters[parameter_trained_weights_low_14];
            packed_half_67 = parameters[parameter_trained_weights_high_15];
            address_word_56 =
                uint32_t(combine_words(matrix_fragment_64, 0u) << 0x1u >> 32) + matrix_fragment_64;
            if (condition_1) {
                split_address_words(matrix_fragment_74, matrix_fragment_75,
                                    int64_t(int32_t(matrix_accumulator_79)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
            }
            if (condition_0) {
                matrix_accumulator_81 = matrix_accumulator_81 + matrix_accumulator_82;
            }
            if (condition_1) {
                split_address_words(matrix_fragment_68, matrix_fragment_69,
                                    int64_t(int32_t(matrix_accumulator_83)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_68, matrix_fragment_69));
            }
            matrix_accumulator_79 = uint32_t(int32_t(address_word_56) >> 0x1u);
            packed_half_48 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_48),
                               select_half_operand<2, false, false>(matrix_fragment_40));
            matrix_accumulator_85 =
                uint32_t(combine_words(packed_half_57, 0u) << 0x1u >> 32) + packed_half_57;
            split_address_words(matrix_accumulator_82, matrix_accumulator_83,
                                int64_t(int32_t(matrix_fragment_65)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_58, packed_half_59));
            condition_4 =
                int32_t(matrix_accumulator_95) < int32_t(matrix_accumulator_94) || condition_4;
            matrix_fragment_64 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_65 = parameters[parameter_trained_weights_high_15];
            if (condition_1) {
                split_address_words(matrix_accumulator_96, matrix_accumulator_97,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
            }
            matrix_accumulator_89 = uint32_t(int32_t(matrix_accumulator_85) >> 0x1u);
            split_address_words(matrix_fragment_72, matrix_fragment_73,
                                int64_t(int32_t(matrix_accumulator_79)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_72, matrix_fragment_73));
            packed_half_50 = add_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                                            select_half_operand<2, false, false>(packed_half_42));
            if (condition_0) {
                matrix_fragment_74 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_75 = parameters[parameter_input_encoder_skip_high_3];
            }
            split_address_words(matrix_accumulator_98, matrix_accumulator_99,
                                int64_t(int32_t(matrix_accumulator_99)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_60, matrix_fragment_61));
            matrix_fragment_60 =
                uint32_t(combine_words(packed_half_77, 0u) << 0x1u >> 32) + packed_half_77;
            matrix_accumulator_86 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_72, matrix_fragment_73) + 0x80000ull, memory_ranges,
                error_counts);
            packed_half_77 = uint32_t(int32_t(packed_half_76) >> 0x1u);
            split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                                int64_t(int32_t(matrix_accumulator_89)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_62, matrix_fragment_63));
            condition_3 = condition_4 & condition_3;
            matrix_accumulator_85 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_82, matrix_accumulator_83) + 0x80000ull,
                memory_ranges, error_counts);
            if (!condition_1) {
                packed_half_59 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
            }
            split_address_words(packed_half_76, packed_half_77,
                                int64_t(int32_t(packed_half_77)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_66, packed_half_67));
            packed_half_49 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_49),
                               select_half_operand<2, false, false>(matrix_fragment_41));
            matrix_accumulator_87 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_98, matrix_accumulator_99) + 0x80000ull,
                memory_ranges, error_counts);
            if (condition_0) {
                matrix_accumulator_112 = parameters[parameter_input_encoder_skip_low_2];
                matrix_accumulator_113 = parameters[parameter_input_encoder_skip_high_3];
            }
            if (!condition_1) {
                packed_half_67 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
            }
            matrix_accumulator_88 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_88, matrix_accumulator_89) + 0x80000ull,
                memory_ranges, error_counts);
            if (condition_0) {
                matrix_fragment_72 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_73 = parameters[parameter_input_encoder_skip_high_3];
            }
            matrix_accumulator_79 = uint32_t(int32_t(matrix_fragment_60) >> 0x1u);
            if (!condition_1) {
                packed_half_59 = permute_signed(packed_half_59, packed_half_59, 0x5410u);
            }
            if (condition_0) {
                matrix_accumulator_89 = matrix_accumulator_93 * 0x200u + address_word_2;
            }
            if (!condition_1) {
                packed_half_67 = permute_signed(packed_half_67, packed_half_67, 0x5410u);
            }
            split_address_words(matrix_accumulator_78, matrix_accumulator_79,
                                int64_t(int32_t(matrix_accumulator_79)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_64, matrix_fragment_65));
            if (!condition_2) {
                matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));
            }
            if (condition_3) {
                matrix_accumulator_108 = parameters[parameter_input_encoder_skip_low_2];
                matrix_accumulator_109 = parameters[parameter_input_encoder_skip_high_3];
            }
            if (!condition_1) {
                address_word_56 = packed_half_59;
            }
            if (condition_0) {
                matrix_accumulator_89 = matrix_accumulator_89 * 0x4u;
            }
            if (!condition_1) {
                packed_half_57 = packed_half_59;

                address_word_58 = packed_half_59;

                matrix_fragment_64 = packed_half_67;

                matrix_fragment_65 = packed_half_67;

                matrix_fragment_66 = packed_half_67;
            }
            if (condition_1) {
                split_address_words(matrix_accumulator_82, matrix_accumulator_83,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_68, matrix_fragment_69));
            }
            if (!condition_2) {
                matrix_fragment_63 =
                    permute_signed(matrix_fragment_63, matrix_fragment_63, 0x5410u);
            }
            if (condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_96, matrix_accumulator_97), address_word_56,
                    packed_half_57, address_word_58, packed_half_59, memory_ranges, error_counts);
            }
            if (condition_3) {
                matrix_accumulator_93 = matrix_fragment_17 + 0x1u;
            }
            if (condition_0) {
                matrix_accumulator_81 = matrix_accumulator_81 * 0x200u + address_word_2;

                split_address_words(matrix_fragment_74, matrix_fragment_75,
                                    int64_t(int32_t(matrix_accumulator_89)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
            }
            if (condition_1) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_82, matrix_accumulator_83), matrix_fragment_64,
                    matrix_fragment_65, matrix_fragment_66, packed_half_67, memory_ranges,
                    error_counts);
            }
            if (!condition_2) {
                matrix_fragment_60 = matrix_fragment_63;

                matrix_fragment_61 = matrix_fragment_63;
            }
            matrix_accumulator_89 = load_aligned_word<CheckBounds>(
                combine_words(packed_half_76, packed_half_77) + 0x80000ull, memory_ranges,
                error_counts);
            if (!condition_2) {
                matrix_fragment_62 = matrix_fragment_63;
            }
            if (condition_3) {
                condition_1 = int32_t(matrix_accumulator_93) >= int32_t(matrix_accumulator_80);
            }
            matrix_accumulator_90 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_78, matrix_accumulator_79) + 0x80000ull,
                memory_ranges, error_counts);
            if (condition_2) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_70, matrix_fragment_71), matrix_fragment_60,
                    matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
                    error_counts);
            }
            if (condition_0) {
                packed_half_77 = (matrix_accumulator_81 << 0x2u) + 0x80u;

                split_address_words(matrix_accumulator_78, matrix_accumulator_79,
                                    int64_t(int32_t(packed_half_77)) * int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_72, matrix_fragment_73));
            }
            if (condition_3) {
                matrix_fragment_72 = condition_1 ? matrix_accumulator_95 : 0u;
            }
            if (!condition_0) {
                matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));
            }
            if (condition_3) {
                matrix_fragment_40 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_94;

                condition_1 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);

                matrix_accumulator_82 = condition_6 ? matrix_fragment_72 : 0u;

                matrix_accumulator_97 = matrix_fragment_17 + 0x1u;
            }
            if (!condition_0) {
                matrix_fragment_71 =
                    permute_signed(matrix_fragment_71, matrix_fragment_71, 0x5410u);
            }
            if (condition_3) {
                packed_half_76 = parameters[parameter_input_encoder_skip_low_2];
                packed_half_77 = parameters[parameter_input_encoder_skip_high_3];

                matrix_fragment_40 = condition_6 ? matrix_fragment_40 : 0u;

                matrix_accumulator_81 =
                    !condition_1 ? matrix_accumulator_82 : matrix_accumulator_95;

                condition_2 = int32_t(matrix_accumulator_97) >= int32_t(matrix_accumulator_80);
            }
            if (!condition_0) {
                matrix_fragment_68 = matrix_fragment_71;

                matrix_fragment_69 = matrix_fragment_71;

                matrix_fragment_70 = matrix_fragment_71;
            }
            if (condition_0) {
                split_address_words(matrix_fragment_74, matrix_fragment_75,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_74, matrix_fragment_75));
            }
            if (!condition_0) {
                matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                   select_half_operand<2, false, false>(0u));
            }
            if (condition_3) {
                matrix_accumulator_81 = matrix_fragment_40 + matrix_accumulator_81;

                matrix_fragment_40 = condition_2 ? matrix_accumulator_95 : 0u;
            }
            if (condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_fragment_74, matrix_fragment_75), matrix_fragment_68,
                    matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
                    error_counts);
            }
            if (condition_3) {
                condition_1 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);

                packed_half_42 = matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_94;

                matrix_accumulator_82 = condition_6 ? matrix_fragment_40 : 0u;

                matrix_fragment_40 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_41 = parameters[parameter_input_encoder_skip_high_3];
            }
            if (!condition_0) {
                matrix_fragment_75 =
                    permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5410u);
            }
            if (condition_0) {
                split_address_words(
                    matrix_accumulator_78, matrix_accumulator_79,
                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_78, matrix_accumulator_79));
            }
            if (condition_3) {
                packed_half_42 = condition_6 ? packed_half_42 : 0u;
            }
            if (!condition_0) {
                matrix_fragment_72 = matrix_fragment_75;

                matrix_fragment_73 = matrix_fragment_75;

                matrix_fragment_74 = matrix_fragment_75;
            }
            if (condition_3) {
                matrix_accumulator_81 = matrix_accumulator_81 * 0x200u + address_word_2;

                matrix_accumulator_83 =
                    !condition_1 ? matrix_accumulator_82 : matrix_accumulator_95;

                condition_1 = int32_t(matrix_accumulator_93) >= int32_t(matrix_accumulator_80);

                matrix_accumulator_81 = matrix_accumulator_81 * 0x4u;
            }
            if (condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_78, matrix_accumulator_79), matrix_fragment_72,
                    matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
                    error_counts);
            }
            if (condition_3) {
                matrix_accumulator_83 = packed_half_42 + matrix_accumulator_83;

                condition_2 = int32_t(matrix_accumulator_97) >= int32_t(matrix_accumulator_80);

                split_address_words(matrix_accumulator_80, matrix_accumulator_81,
                                    int64_t(int32_t(matrix_accumulator_81)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(packed_half_76, packed_half_77));
            }
            if (!condition_3) {
                matrix_accumulator_79 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                      select_half_operand<2, false, false>(0u));
            }
            if (condition_3) {
                packed_half_42 = matrix_accumulator_83 * 0x200u + address_word_2;
            }
            if (!condition_3) {
                matrix_accumulator_79 =
                    permute_signed(matrix_accumulator_79, matrix_accumulator_79, 0x5410u);
            }
            if (condition_3) {
                matrix_accumulator_96 = condition_1 ? matrix_accumulator_95 : 0u;

                split_address_words(
                    matrix_accumulator_82, matrix_accumulator_83,
                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_80, matrix_accumulator_81));
            }
            if (!condition_3) {
                packed_half_76 = matrix_accumulator_79;

                packed_half_77 = matrix_accumulator_79;

                matrix_accumulator_78 = matrix_accumulator_79;
            }
            if (condition_3) {
                matrix_accumulator_81 = (packed_half_42 << 0x2u) + 0x80u;
            }
            if (condition_0) {
                matrix_accumulator_93 = matrix_accumulator_94 * matrix_accumulator_91;
            }
            if (condition_3) {
                condition_1 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);
            }
            if (!condition_3) {
                matrix_accumulator_80 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                      select_half_operand<2, false, false>(0u));
            }
            if (condition_3) {
                matrix_accumulator_96 = condition_6 ? matrix_accumulator_96 : 0u;

                matrix_accumulator_98 = condition_2 ? matrix_accumulator_95 : 0u;

                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_82, matrix_accumulator_83), packed_half_76,
                    packed_half_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
                    error_counts);
            }
            if (condition_0) {
                matrix_accumulator_93 = matrix_accumulator_93 * 0x2u + matrix_accumulator_94;
            }
            if (condition_3) {
                split_address_words(matrix_fragment_40, matrix_fragment_41,
                                    int64_t(int32_t(matrix_accumulator_81)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_40, matrix_fragment_41));

                matrix_accumulator_99 =
                    !condition_1 ? matrix_accumulator_96 : matrix_accumulator_95;

                condition_2 = uint32_t(matrix_accumulator_92) != uint32_t(0x4u);

                matrix_accumulator_96 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_94;
            }
            if (!condition_3) {
                matrix_accumulator_83 =
                    permute_signed(matrix_accumulator_80, matrix_accumulator_80, 0x5410u);
            }
            if (condition_3) {
                matrix_accumulator_98 = condition_6 ? matrix_accumulator_98 : 0u;
            }
            if (condition_0) {
                matrix_accumulator_97 = condition_6 ? matrix_accumulator_93 : 0u;
            }
            if (condition_3) {
                split_address_words(matrix_accumulator_92, matrix_accumulator_93,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_40, matrix_fragment_41));
            }
            if (!condition_3) {
                matrix_accumulator_80 = matrix_accumulator_83;

                matrix_accumulator_81 = matrix_accumulator_83;

                matrix_accumulator_82 = matrix_accumulator_83;
            }
            if (condition_0) {
                matrix_accumulator_91 = matrix_accumulator_94 * matrix_accumulator_91;
            }
            if (condition_3) {
                matrix_accumulator_96 = condition_6 ? matrix_accumulator_96 : 0u;

                matrix_fragment_103 = !condition_2 ? matrix_accumulator_98 : matrix_accumulator_95;

                matrix_accumulator_98 =
                    matrix_fragment_17 * matrix_accumulator_94 + matrix_accumulator_94;
            }
            if (condition_0) {
                packed_half_42 = !condition_5 ? address_word_11 : 0u;

                matrix_accumulator_95 = matrix_accumulator_91 * 0x2u + matrix_accumulator_94;
            }
            if (condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_92, matrix_accumulator_93),
                    matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_82,
                    matrix_accumulator_83, memory_ranges, error_counts);

                matrix_fragment_40 = parameters[parameter_input_encoder_skip_low_2];
                matrix_fragment_41 = parameters[parameter_input_encoder_skip_high_3];

                matrix_accumulator_98 = condition_6 ? matrix_accumulator_98 : 0u;
            }
            if (condition_0) {
                matrix_accumulator_95 = condition_6 ? matrix_accumulator_95 : 0u;
            }
            if (condition_3) {
                matrix_accumulator_93 = matrix_accumulator_96 + matrix_accumulator_99;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_122), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_99 = result_word;
            }
            if (condition_0) {
                matrix_accumulator_95 = packed_half_42 + matrix_accumulator_95;

                matrix_accumulator_97 = packed_half_42 + matrix_accumulator_97;
            }
            if (condition_3) {
                matrix_fragment_103 = matrix_accumulator_98 + matrix_fragment_103;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_106), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_98 = result_word;
            }
            packed_half_51 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                               select_half_operand<2, false, false>(matrix_fragment_43));
            if (condition_0) {
                matrix_fragment_43 = matrix_accumulator_95 * 0x200u + address_word_2;

                matrix_accumulator_91 = matrix_accumulator_97 * 0x200u + address_word_2;
            }
            matrix_accumulator_96 = unpack_fp8_bits(packed_half_28 >> 16);
            if (condition_0) {
                matrix_fragment_43 = (matrix_fragment_43 << 0x2u) + 0x180u;

                matrix_accumulator_91 = (matrix_accumulator_91 << 0x2u) + 0x100u;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_104), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_138 = result_word;
            }
            if (condition_3) {
                matrix_accumulator_93 = matrix_accumulator_93 * 0x200u + address_word_2;

                matrix_accumulator_95 = matrix_fragment_103 * 0x200u + address_word_2;
            }
            if (condition_0) {
                split_address_words(
                    matrix_accumulator_110, matrix_accumulator_111,
                    int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                        combine_words(matrix_accumulator_110, matrix_accumulator_111));
            }
            packed_half_42 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_43 = parameters[parameter_trained_weights_high_15];
            if (condition_0) {
                split_address_words(
                    matrix_accumulator_112, matrix_accumulator_113,
                    int64_t(int32_t(matrix_accumulator_91)) * int64_t(int32_t(0x4u)) +
                        combine_words(matrix_accumulator_112, matrix_accumulator_113));
            }
            matrix_accumulator_91 = 0x10u | address_word_3;
            matrix_accumulator_96 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_96),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(matrix_accumulator_99));
            matrix_accumulator_99 = 0x18u | address_word_3;
            if (condition_3) {
                matrix_accumulator_93 = (matrix_accumulator_93 << 0x2u) + 0x100u;

                matrix_accumulator_95 = (matrix_accumulator_95 << 0x2u) + 0x180u;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_122), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_107 = result_word;
            }
            matrix_accumulator_97 = unpack_fp8_bits(packed_half_28);
            if (condition_3) {
                split_address_words(
                    matrix_accumulator_108, matrix_accumulator_109,
                    int64_t(int32_t(matrix_accumulator_93)) * int64_t(int32_t(0x4u)) +
                        combine_words(matrix_accumulator_108, matrix_accumulator_109));
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_122), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_142 = result_word;
            }
            if (condition_3) {
                split_address_words(matrix_fragment_40, matrix_fragment_41,
                                    int64_t(int32_t(matrix_accumulator_95)) *
                                            int64_t(int32_t(0x4u)) +
                                        combine_words(matrix_fragment_40, matrix_fragment_41));
            }
            matrix_accumulator_97 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_97),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(matrix_accumulator_98));
            matrix_accumulator_98 = 0x8u | address_word_3;
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_106), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_136 = result_word;
            }
            if (condition_0) {
                split_address_words(
                    matrix_accumulator_112, matrix_accumulator_113,
                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_112, matrix_accumulator_113));
            }
            matrix_accumulator_93 = unpack_fp8_bits(packed_half_30);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_104), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_115 = result_word;
            }
            if (condition_0) {
                split_address_words(
                    matrix_accumulator_110, matrix_accumulator_111,
                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_110, matrix_accumulator_111));
            }
            condition_1 = int32_t(address_word_11) >= int32_t(matrix_accumulator_94);
            if (condition_3) {
                split_address_words(
                    matrix_accumulator_108, matrix_accumulator_109,
                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                        combine_words(matrix_accumulator_108, matrix_accumulator_109));
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_106), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_117 = result_word;
            }
            if (condition_3) {
                split_address_words(packed_half_124, address_word_125,
                                    int64_t(int32_t(address_word_125)) * int64_t(int32_t(0x10u)) +
                                        combine_words(matrix_fragment_40, matrix_fragment_41));
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_122), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_140 = result_word;
            }
            matrix_fragment_40 = parameters[parameter_trained_weights_low_14];
            matrix_fragment_41 = parameters[parameter_trained_weights_high_15];
            matrix_accumulator_95 = unpack_fp8_bits(packed_half_29);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_106), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_119 = result_word;
            }
            matrix_accumulator_94 = unpack_fp8_bits(packed_half_29 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_104), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_121 = result_word;
            }
            packed_half_29 = uint32_t(int32_t(packed_half_100) >> 0x1u);
            matrix_accumulator_93 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_93),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_138));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_102), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_138 = result_word;
            }
            matrix_accumulator_92 = unpack_fp8_bits(packed_half_31);
            split_address_words(packed_half_42, matrix_fragment_43,
                                int64_t(int32_t(packed_half_29)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_42, matrix_fragment_43));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_102), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_146 = result_word;
            }
            matrix_fragment_103 = unpack_fp8_bits(packed_half_31 >> 16);
            packed_half_31 = unpack_fp8_bits(matrix_fragment_45 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_104), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            matrix_accumulator_94 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_94),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(matrix_accumulator_107));
            matrix_accumulator_107 =
                uint32_t(combine_words(matrix_fragment_101, 0u) << 0x1u >> 32) +
                matrix_fragment_101;
            packed_half_100 = load_aligned_word<CheckBounds>(
                combine_words(packed_half_42, matrix_fragment_43) + 0x80000ull, memory_ranges,
                error_counts);
            matrix_accumulator_105 = unpack_fp8_bits(packed_half_30 >> 16);
            matrix_accumulator_104 = unpack_fp8_bits(matrix_fragment_47 >> 16);
            packed_half_42 = unpack_fp8_bits(matrix_fragment_47);
            matrix_fragment_47 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_31),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_142));
            if (!condition_0) {
                packed_half_31 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
            }
            packed_half_28 = unpack_fp8_bits(matrix_fragment_44);
            packed_half_29 = unpack_fp8_bits(matrix_fragment_44 >> 16);
            packed_half_30 = unpack_fp8_bits(matrix_fragment_45);
            matrix_accumulator_107 = uint32_t(int32_t(matrix_accumulator_107) >> 0x1u);
            if (!condition_0) {
                packed_half_31 = permute_signed(packed_half_31, packed_half_31, 0x5410u);
            }
            matrix_accumulator_95 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_95),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_136));
            matrix_fragment_101 = unpack_fp8_bits(matrix_fragment_46);
            matrix_accumulator_92 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_92),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_115));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_102), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_136 = result_word;
            }
            matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_46 >> 16);
            matrix_fragment_44 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_28),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_117));
            split_address_words(matrix_fragment_40, matrix_fragment_41,
                                int64_t(int32_t(matrix_accumulator_107)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_40, matrix_fragment_41));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_102), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_115 = result_word;
            }
            matrix_fragment_45 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_29),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_140));
            if (!condition_0) {
                packed_half_28 = packed_half_31;
            }
            matrix_fragment_46 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_30),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_119));
            if (!condition_0) {
                packed_half_29 = packed_half_31;

                packed_half_30 = packed_half_31;
            }
            matrix_accumulator_123 = unpack_fp8_bits(address_word_32);
            matrix_accumulator_107 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_40, matrix_fragment_41) + 0x80000ull, memory_ranges,
                error_counts);
            matrix_accumulator_122 = unpack_fp8_bits(address_word_33);
            matrix_fragment_102 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_42),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(matrix_accumulator_121));
            matrix_accumulator_121 = unpack_fp8_bits(address_word_32 >> 16);
            packed_half_127 = unpack_fp8_bits(address_word_35 >> 16);
            matrix_accumulator_106 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_103),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_138));
            address_word_32 = unpack_fp8_bits(address_word_33 >> 16);
            if (condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_112, matrix_accumulator_113), packed_half_28,
                    packed_half_29, packed_half_30, packed_half_31, memory_ranges, error_counts);
            }
            address_word_33 = unpack_fp8_bits(address_word_35);
            if (!condition_0) {
                packed_half_42 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_114), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                address_word_35 = result_word;
            }
            matrix_fragment_103 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_43),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_146));
            if (!condition_0) {
                matrix_fragment_43 = permute_signed(packed_half_42, packed_half_42, 0x5410u);
            }
            matrix_fragment_13 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                               select_half_operand<2, false, false>(matrix_fragment_129));
            packed_half_119 = unpack_fp8_bits(packed_half_34);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_116), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_129 = result_word;
            }
            packed_half_117 = unpack_fp8_bits(packed_half_34 >> 16);
            if (!condition_0) {
                matrix_fragment_40 = matrix_fragment_43;

                matrix_fragment_41 = matrix_fragment_43;

                packed_half_42 = matrix_fragment_43;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_114), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_34 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_118), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_140 = result_word;
            }
            if (condition_0) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_110, matrix_accumulator_111),
                    matrix_fragment_40, matrix_fragment_41, packed_half_42, matrix_fragment_43,
                    memory_ranges, error_counts);
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_120), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_152 = result_word;
            }
            matrix_accumulator_122 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_122),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(address_word_35));
            if (!condition_3) {
                address_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_120), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_137 = result_word;
            }
            if (!condition_3) {
                address_word_35 = permute_signed(address_word_35, address_word_35, 0x5410u);
            }
            matrix_fragment_14 =
                add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                               select_half_operand<2, false, false>(matrix_fragment_131));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_118), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_138 = result_word;
            }
            matrix_fragment_101 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_101),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_144));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_118), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_150 = result_word;
            }
            matrix_accumulator_123 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_123),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(packed_half_34));
            if (!condition_3) {
                packed_half_34 = address_word_35;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_118), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_135 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_120), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_142 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_accumulator_120), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            packed_half_118 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_33),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_140));
            if (!condition_3) {
                address_word_33 = address_word_35;
            }
            matrix_accumulator_110 = unpack_fp8_bits(address_word_39);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_114), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_146 = result_word;
            }
            address_word_39 = unpack_fp8_bits(address_word_39 >> 16);
            matrix_accumulator_120 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_32),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(matrix_fragment_129));
            if (!condition_3) {
                address_word_32 = address_word_35;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_114), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_131 = result_word;
            }
            matrix_accumulator_112 = unpack_fp8_bits(address_word_37 >> 16);
            matrix_accumulator_104 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_104),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_115));
            packed_half_115 = unpack_fp8_bits(packed_half_36);
            if (condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(matrix_accumulator_108, matrix_accumulator_109), address_word_32,
                    address_word_33, packed_half_34, address_word_35, memory_ranges, error_counts);
            }
            matrix_accumulator_113 = unpack_fp8_bits(packed_half_36 >> 16);
            packed_half_114 = unpack_fp8_bits(address_word_37);
            if (!condition_3) {
                address_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
            }
            packed_half_36 = unpack_fp8_bits(address_word_38 >> 16);
            matrix_accumulator_108 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_39),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_152));
            if (!condition_3) {
                address_word_39 = permute_signed(address_word_37, address_word_37, 0x5410u);
            }
            matrix_accumulator_111 = unpack_fp8_bits(address_word_38);
            matrix_accumulator_109 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_36),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_137));
            if (!condition_3) {
                packed_half_36 = address_word_39;

                address_word_37 = address_word_39;

                address_word_38 = address_word_39;
            }
            if (condition_3) {
                load_four_words_cached<CheckBounds>(
                    combine_words(packed_half_124, address_word_125), packed_half_36,
                    address_word_37, address_word_38, address_word_39, memory_ranges, error_counts);
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_116), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_148 = result_word;
            }
            matrix_accumulator_105 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_105),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_136));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_116), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_133 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_116), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_136 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_134), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_145 = result_word;
            }
            packed_half_139 = unpack_fp8_bits(matrix_fragment_55);
            packed_half_116 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_127),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_144));
            packed_half_127 = unpack_fp8_bits(matrix_fragment_55 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_128), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_141 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_134), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_55 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_134), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_143 = result_word;
            }
            packed_half_115 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_115),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(packed_half_146));
            packed_half_117 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_117),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_142));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_130), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_130), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_146 = result_word;
            }
            matrix_accumulator_113 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_113),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_148));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_132), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_142 = result_word;
            }
            matrix_accumulator_111 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_111),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_150));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_130), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_148 = result_word;
            }
            packed_half_140 = unpack_fp8_bits(matrix_fragment_53 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_128), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_150 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_130), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_152 = result_word;
            }
            packed_half_119 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_119),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_138));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_134), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_154 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_132), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_124 = result_word;
            }
            packed_half_114 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_114),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(matrix_fragment_131));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_132), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                address_word_125 = result_word;
            }
            matrix_accumulator_112 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_112),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_133));
            packed_half_137 = unpack_fp8_bits(matrix_fragment_52);
            matrix_accumulator_110 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_110),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_135));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_132), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_129 = result_word;
            }
            packed_half_135 = unpack_fp8_bits(matrix_fragment_52 >> 16);
            matrix_accumulator_121 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_121),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_136));
            packed_half_133 = unpack_fp8_bits(matrix_fragment_54);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_126), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_138 = result_word;
            }
            matrix_fragment_131 = unpack_fp8_bits(matrix_fragment_54 >> 16);
            packed_half_134 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_140),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(packed_half_145));
            matrix_fragment_52 = unpack_fp8_bits(address_word_56);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_128), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_147 = result_word;
            }
            matrix_fragment_54 = unpack_fp8_bits(address_word_56 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_128), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_156 = result_word;
            }
            packed_half_136 = unpack_fp8_bits(matrix_fragment_53);
            address_word_56 = unpack_fp8_bits(packed_half_57 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_126), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_128 = result_word;
            }
            matrix_fragment_53 = unpack_fp8_bits(packed_half_57);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_126), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_140 = result_word;
            }
            packed_half_57 = unpack_fp8_bits(address_word_58);
            matrix_fragment_54 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_54),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(matrix_fragment_55));
            matrix_fragment_130 = unpack_fp8_bits(packed_half_59);
            packed_half_136 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_136),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_141));
            matrix_fragment_55 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_56),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(packed_half_143));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_19), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_141 = result_word;
            }
            address_word_56 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_57),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_144));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_19), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_143 = result_word;
            }
            packed_half_57 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_130),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_146));
            matrix_fragment_131 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_131),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(packed_half_142));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_4), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_146 = result_word;
            }
            address_word_58 = unpack_fp8_bits(address_word_58 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_6), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_142 = result_word;
            }
            packed_half_59 = unpack_fp8_bits(packed_half_59 >> 16);
            packed_half_132 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_139),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_148));
            packed_half_139 = unpack_fp8_bits(matrix_fragment_60);
            packed_half_137 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_137),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_150));
            packed_half_133 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_133),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_152));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_126), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_150 = result_word;
            }
            packed_half_135 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_135),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(packed_half_154));
            matrix_fragment_130 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_127),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(packed_half_124));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_4), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_152 = result_word;
            }
            address_word_58 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_58),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(address_word_125));
            packed_half_124 = unpack_fp8_bits(matrix_fragment_64);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_5), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_126 = result_word;
            }
            address_word_125 = unpack_fp8_bits(matrix_fragment_65);
            packed_half_59 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_59),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(matrix_fragment_129));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_5), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_154 = result_word;
            }
            matrix_fragment_129 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_139),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_138));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_7), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_127 = result_word;
            }
            packed_half_138 = unpack_fp8_bits(matrix_fragment_60 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_6), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            matrix_fragment_60 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_124),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(matrix_fragment_128));
            packed_half_124 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_125),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_140));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_7), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_151 = result_word;
            }
            packed_half_140 = unpack_fp8_bits(matrix_fragment_61 >> 16);
            packed_half_138 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_138),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_141));
            packed_half_139 = unpack_fp8_bits(matrix_fragment_61);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_8), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                matrix_fragment_128 = result_word;
            }
            matrix_fragment_61 = unpack_fp8_bits(matrix_fragment_68);
            packed_half_140 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_140),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_143));
            packed_half_143 = unpack_fp8_bits(matrix_fragment_70);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_9), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_141 = result_word;
            }
            matrix_fragment_52 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_52),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_147));
            address_word_125 = unpack_fp8_bits(matrix_fragment_69);
            matrix_fragment_61 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_61),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_146));
            packed_half_147 = unpack_fp8_bits(matrix_fragment_69 >> 16);
            matrix_fragment_69 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_143),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_142));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_10), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_146 = result_word;
            }
            matrix_fragment_68 = unpack_fp8_bits(matrix_fragment_68 >> 16);
            packed_half_149 = unpack_fp8_bits(matrix_fragment_71 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_12), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_145 = result_word;
            }
            packed_half_148 = unpack_fp8_bits(matrix_fragment_71);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_8), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_142 = result_word;
            }
            matrix_fragment_71 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_125),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_152));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_9), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_143 = result_word;
            }
            matrix_fragment_68 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_68),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_126));
            address_word_125 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_147),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_154));
            packed_half_127 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_149),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_127));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_4), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_147 = result_word;
            }
            packed_half_126 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_148),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_144));
            matrix_fragment_70 = unpack_fp8_bits(matrix_fragment_70 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_4), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_149 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_5), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            packed_half_139 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_139),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_150));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_10), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_148 = result_word;
            }
            matrix_fragment_70 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_70),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_151));
            packed_half_150 = unpack_fp8_bits(matrix_fragment_72);
            matrix_fragment_72 = unpack_fp8_bits(matrix_fragment_72 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_12), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_151 = result_word;
            }
            matrix_fragment_128 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_150),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(matrix_fragment_128));
            packed_half_150 = unpack_fp8_bits(matrix_fragment_74);
            matrix_fragment_4 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_72),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_141));
            matrix_fragment_74 = unpack_fp8_bits(matrix_fragment_74 >> 16);
            matrix_fragment_72 = unpack_fp8_bits(matrix_fragment_73);
            packed_half_141 = unpack_fp8_bits(matrix_fragment_73 >> 16);
            packed_half_152 = unpack_fp8_bits(matrix_fragment_75);
            packed_half_153 = unpack_fp8_bits(matrix_fragment_75 >> 16);
            matrix_fragment_75 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_150),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_146));
            packed_half_155 = unpack_fp8_bits(packed_half_77);
            matrix_fragment_74 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_74),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_145));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_5), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_150 = result_word;
            }
            matrix_fragment_73 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_72),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(packed_half_142));
            packed_half_145 = unpack_fp8_bits(packed_half_76);
            matrix_fragment_72 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_141),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_143));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_6), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_142 = result_word;
            }
            packed_half_154 = unpack_fp8_bits(packed_half_76 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_6), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_146 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_7), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_141 = result_word;
            }
            matrix_fragment_5 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_145),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_147));
            packed_half_76 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_155),
                select_half_operand<2, false, false>(packed_half_22),
                select_half_operand<2, false, false>(packed_half_149));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_7), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_143 = result_word;
            }
            packed_half_22 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_154),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_144));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_8), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_154 = result_word;
            }
            matrix_fragment_6 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_153),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_151));
            matrix_fragment_7 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_152),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_148));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_8), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_152 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_9), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_149 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_9), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_151 = result_word;
            }
            packed_half_77 = unpack_fp8_bits(packed_half_77 >> 16);
            packed_half_153 = unpack_fp8_bits(matrix_accumulator_78);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_12), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_145 = result_word;
            }
            packed_half_155 = unpack_fp8_bits(matrix_accumulator_79);
            matrix_accumulator_78 = unpack_fp8_bits(matrix_accumulator_78 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_10), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            packed_half_21 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_77),
                select_half_operand<2, false, false>(packed_half_21),
                select_half_operand<2, false, false>(packed_half_150));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_12), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_147 = result_word;
            }
            packed_half_77 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_155),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_146));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_10), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_148 = result_word;
            }
            matrix_accumulator_79 = unpack_fp8_bits(matrix_accumulator_79 >> 16);
            packed_half_8 = unpack_fp8_bits(matrix_accumulator_81);
            matrix_fragment_12 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_153),
                select_half_operand<2, false, false>(packed_half_23),
                select_half_operand<2, false, false>(packed_half_142));
            packed_half_23 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_78),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_141));
            matrix_accumulator_78 = unpack_fp8_bits(matrix_accumulator_80);
            address_word_24 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_79),
                select_half_operand<2, false, false>(address_word_24),
                select_half_operand<2, false, false>(packed_half_143));
            matrix_accumulator_79 = unpack_fp8_bits(matrix_accumulator_80 >> 16);
            packed_half_9 = unpack_fp8_bits(matrix_accumulator_81 >> 16);
            matrix_accumulator_78 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_78),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(packed_half_154));
            address_word_25 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_8),
                select_half_operand<2, false, false>(address_word_25),
                select_half_operand<2, false, false>(packed_half_152));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_19), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_152 = result_word;
            }
            matrix_accumulator_79 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_79),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_149));
            address_word_26 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_9),
                select_half_operand<2, false, false>(address_word_26),
                select_half_operand<2, false, false>(packed_half_151));
            matrix_accumulator_81 = unpack_fp8_bits(matrix_accumulator_82 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_19), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_151 = result_word;
            }
            packed_half_10 = unpack_fp8_bits(matrix_accumulator_83);
            matrix_accumulator_80 = unpack_fp8_bits(matrix_accumulator_82);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_48), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_157 = result_word;
            }
            matrix_accumulator_83 = unpack_fp8_bits(matrix_accumulator_83 >> 16);
            matrix_accumulator_81 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_81),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_145));
            matrix_fragment_53 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_53),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_156));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_18), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_154 = result_word;
            }
            matrix_accumulator_80 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_80),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_144));
            matrix_accumulator_84 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_83),
                select_half_operand<2, false, false>(matrix_accumulator_84),
                select_half_operand<2, false, false>(packed_half_147));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_18), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_9 = result_word;
            }
            address_word_27 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_10),
                select_half_operand<2, false, false>(address_word_27),
                select_half_operand<2, false, false>(packed_half_148));
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_18), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_142 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_18), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_143 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_50), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_162 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_13), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_160 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_13), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_156 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_13), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_141 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_13), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_83 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_48), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_18 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_50), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_161 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_14), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_13 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_14), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_155 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_14), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_10 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_14), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_82 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_20), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_153 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_20), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_8 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_20), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_144 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_20), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_146 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_15), "r"(address_word_3), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_14 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_15), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_19 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_15), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_145 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(matrix_fragment_15), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_148 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_16), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                matrix_fragment_15 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_16), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_20 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_16), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_147 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_16), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_149 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_50), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_16 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_48), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_165 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_50), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_163 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_49), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_159 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_51), "r"(address_word_3), "r"(0x1fu), "r"(__activemask()));
                packed_half_158 = result_word;
            }
            packed_half_50 = unpack_fp8_bits(matrix_fragment_65 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_48), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_164 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_49), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_150 = result_word;
            }
            address_word_3 = unpack_fp8_bits(matrix_fragment_64 >> 16);
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_51), "r"(matrix_accumulator_91), "r"(0x1fu),
                      "r"(__activemask()));
                packed_half_48 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_49), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_fragment_65 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_49), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_91 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_51), "r"(matrix_accumulator_98), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_98 = result_word;
            }
            {
                uint32_t result_word;
                asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
                    : "=r"(result_word)
                    : "r"(packed_half_51), "r"(matrix_accumulator_99), "r"(0x1fu),
                      "r"(__activemask()));
                matrix_accumulator_99 = result_word;
            }
            packed_half_50 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_50),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_151));
            packed_half_151 = unpack_fp8_bits(matrix_fragment_63);
            packed_half_51 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_3),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_152));
            address_word_3 = unpack_fp8_bits(matrix_fragment_66);
            packed_half_152 = unpack_fp8_bits(matrix_fragment_62);
            matrix_fragment_63 = unpack_fp8_bits(matrix_fragment_63 >> 16);
            packed_half_49 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_3),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_157));
            packed_half_157 = unpack_fp8_bits(matrix_fragment_62 >> 16);
            address_word_3 = unpack_fp8_bits(matrix_fragment_66 >> 16);
            matrix_fragment_62 = unpack_fp8_bits(packed_half_67);
            matrix_fragment_66 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_63),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_162));
            matrix_fragment_64 = unpack_fp8_bits(packed_half_67 >> 16);
            matrix_fragment_63 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_3),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_161));
            address_word_3 = unpack_fp8_bits(packed_half_28);
            matrix_fragment_62 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_62),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_18));
            packed_half_18 = unpack_fp8_bits(packed_half_28 >> 16);
            packed_half_28 = unpack_fp8_bits(packed_half_30 >> 16);
            matrix_fragment_64 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_64),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_16));
            packed_half_16 = unpack_fp8_bits(packed_half_30);
            matrix_fragment_15 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_28),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(matrix_fragment_15));
            packed_half_28 = unpack_fp8_bits(packed_half_31);
            matrix_fragment_13 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_18),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(matrix_fragment_13));
            packed_half_18 = unpack_fp8_bits(packed_half_29 >> 16);
            matrix_fragment_14 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_16),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(matrix_fragment_14));
            packed_half_16 = unpack_fp8_bits(packed_half_29);
            matrix_fragment_19 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_28),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(matrix_fragment_19));
            packed_half_29 = unpack_fp8_bits(packed_half_31 >> 16);
            packed_half_30 = unpack_fp8_bits(matrix_fragment_40 >> 16);
            packed_half_28 = unpack_fp8_bits(matrix_fragment_40);
            matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_41);
            packed_half_29 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_29),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(matrix_fragment_20));
            matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_41 >> 16);
            matrix_fragment_20 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_30),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_153));
            packed_half_31 = unpack_fp8_bits(packed_half_42);
            packed_half_30 = unpack_fp8_bits(packed_half_42 >> 16);
            matrix_fragment_40 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_40),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_9));
            packed_half_42 = unpack_fp8_bits(matrix_fragment_43);
            packed_half_67 = unpack_fp8_bits(matrix_fragment_43 >> 16);
            matrix_fragment_43 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_fragment_41),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_8));
            packed_half_8 = parameters[parameter_tensor_height_18];
            packed_half_9 = parameters[parameter_tensor_width_19];
            matrix_fragment_41 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_67),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_48));
            packed_half_48 = unpack_fp8_bits(address_word_32);
            packed_half_67 = unpack_fp8_bits(address_word_33);
            address_word_32 = unpack_fp8_bits(address_word_32 >> 16);
            address_word_33 = unpack_fp8_bits(address_word_33 >> 16);
            address_word_3 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_3),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_160));
            packed_half_16 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_16),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_156));
            packed_half_48 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_48),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(packed_half_141));
            matrix_accumulator_85 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_67),
                select_half_operand<2, false, false>(matrix_accumulator_85),
                select_half_operand<2, false, false>(matrix_accumulator_83));
            packed_half_67 = unpack_fp8_bits(packed_half_34);
            packed_half_18 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_18),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(packed_half_155));
            packed_half_34 = unpack_fp8_bits(packed_half_34 >> 16);
            address_word_32 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_32),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(packed_half_10));
            matrix_accumulator_82 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_33),
                select_half_operand<2, false, false>(matrix_accumulator_86),
                select_half_operand<2, false, false>(matrix_accumulator_82));
            matrix_accumulator_86 = uint32_t(int32_t(packed_half_9) >> 0x1fu);
            packed_half_147 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_34),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(packed_half_147));
            matrix_accumulator_83 = unpack_fp8_bits(address_word_35);
            matrix_accumulator_86 =
                uint32_t(combine_words(matrix_accumulator_86, 0u) << 0x2u >> 32) + packed_half_9;
            packed_half_34 = unpack_fp8_bits(packed_half_36);
            address_word_35 = unpack_fp8_bits(address_word_35 >> 16);
            packed_half_36 = unpack_fp8_bits(packed_half_36 >> 16);
            matrix_accumulator_86 = uint32_t(int32_t(matrix_accumulator_86) >> 0x2u);
            matrix_accumulator_88 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_35),
                select_half_operand<2, false, false>(matrix_accumulator_88),
                select_half_operand<2, false, false>(packed_half_149));
            packed_half_9 = uint32_t(int32_t(packed_half_8) >> 0x1fu);
            packed_half_144 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_36),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_144));
            address_word_35 = unpack_fp8_bits(address_word_37);
            packed_half_36 = matrix_fragment_17 * matrix_accumulator_86 + matrix_accumulator_86;
            packed_half_145 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_67),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_145));
            packed_half_9 =
                uint32_t(combine_words(packed_half_9, 0u) << 0x2u >> 32) + packed_half_8;
            matrix_accumulator_87 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(matrix_accumulator_83),
                select_half_operand<2, false, false>(matrix_accumulator_87),
                select_half_operand<2, false, false>(packed_half_148));
            address_word_37 = unpack_fp8_bits(address_word_37 >> 16);
            packed_half_28 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_28),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_154));
            matrix_accumulator_83 = address_word_11 + 0x1u;
            packed_half_142 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_34),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_142));
            packed_half_8 = matrix_fragment_17 * matrix_accumulator_86 + address_word_11;
            matrix_accumulator_89 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_35),
                select_half_operand<2, false, false>(matrix_accumulator_89),
                select_half_operand<2, false, false>(packed_half_143));
            address_word_11 = address_word_11 + packed_half_36;
            address_word_2 = address_word_2 * 0x4u;
            packed_half_34 = parameters[parameter_output_low_4];
            address_word_35 = parameters[parameter_output_high_5];
            matrix_accumulator_90 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_37),
                select_half_operand<2, false, false>(matrix_accumulator_90),
                select_half_operand<2, false, false>(packed_half_146));
            packed_half_67 = unpack_fp8_bits(address_word_38);
            address_word_37 = packed_half_8 * 0x800u + address_word_2;
            packed_half_10 = unpack_fp8_bits(address_word_38 >> 16);
            address_word_11 = address_word_11 * 0x800u + address_word_2;
            address_word_38 = uint32_t(int32_t(packed_half_9) >> 0x2u);
            address_word_2 = threadIdx.x & 31u;
            condition_2 = int32_t(matrix_fragment_17) >= int32_t(address_word_38) || condition_1;
            address_word_33 = unpack_fp8_bits(address_word_39);
            condition_0 = int32_t(matrix_accumulator_83) >= int32_t(matrix_accumulator_86);
            address_word_39 = unpack_fp8_bits(address_word_39 >> 16);
            packed_half_9 = matrix_fragment_17 + 0x1u;
            packed_half_152 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_152),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_165));
            condition_3 = int32_t(matrix_fragment_17) >= int32_t(address_word_38) || condition_0;
            packed_half_151 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_151),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_164));
            packed_half_157 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_157),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_163));
            condition_1 = int32_t(packed_half_9) >= int32_t(address_word_38) || condition_1;
            split_address_words(packed_half_36, address_word_37,
                                int64_t(int32_t(address_word_37)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_34, address_word_35));
            packed_half_31 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_31),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_159));
            condition_0 = int32_t(packed_half_9) >= int32_t(address_word_38) || condition_0;
            packed_half_30 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_30),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(packed_half_158));
            split_address_words(packed_half_34, address_word_35,
                                int64_t(int32_t(address_word_11)) * int64_t(int32_t(0x4u)) +
                                    combine_words(packed_half_34, address_word_35));
            packed_half_42 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_42),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(packed_half_150));
            matrix_fragment_65 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_67),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(matrix_fragment_65));
            matrix_accumulator_91 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_33),
                select_half_operand<2, false, false>(packed_half_100),
                select_half_operand<2, false, false>(matrix_accumulator_91));
            matrix_accumulator_98 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(packed_half_10),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(matrix_accumulator_98));
            matrix_accumulator_99 = fused_multiply_add_half_pairs(
                select_half_operand<2, false, false>(address_word_39),
                select_half_operand<2, false, false>(matrix_accumulator_107),
                select_half_operand<2, false, false>(matrix_accumulator_99));
            if (!condition_2) {
                packed_half_9 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_130),
                                  select_half_operand<2, false, false>(0u));
                split_address_words(address_word_38, address_word_39,
                                    int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_36, address_word_37));
                packed_half_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_131),
                                  select_half_operand<2, false, false>(0u));
                packed_half_132 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_132),
                                  select_half_operand<2, false, false>(packed_half_9));
                packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_134),
                                              select_half_operand<2, false, false>(0u));
                packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                              select_half_operand<2, false, false>(0u));
                address_word_11 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                  select_half_operand<2, false, false>(0u));
                matrix_fragment_66 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                  select_half_operand<2, false, false>(0u));
                packed_half_136 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                  select_half_operand<2, false, false>(packed_half_9));
                packed_half_133 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                                  select_half_operand<2, false, false>(packed_half_10));
                packed_half_137 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                                  select_half_operand<2, false, false>(packed_half_8));
                packed_half_9 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                  select_half_operand<2, false, false>(0u));
                packed_half_118 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                  select_half_operand<2, false, false>(address_word_11));
                packed_half_151 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                  select_half_operand<2, false, false>(matrix_fragment_66));
                address_word_11 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                  select_half_operand<2, false, false>(0u));
                matrix_accumulator_94 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                  select_half_operand<2, false, false>(0u));
                packed_half_8 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                  select_half_operand<2, false, false>(0u));
                packed_half_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                                  select_half_operand<2, false, false>(0u));
                matrix_fragment_66 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                  select_half_operand<2, false, false>(0u));
                matrix_accumulator_86 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                  select_half_operand<2, false, false>(0u));
                matrix_accumulator_122 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                  select_half_operand<2, false, false>(packed_half_9));
                address_word_11 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                  select_half_operand<2, false, false>(address_word_11));
                packed_half_9 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                  select_half_operand<2, false, false>(matrix_accumulator_94));
                packed_half_8 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                  select_half_operand<2, false, false>(packed_half_8));
                packed_half_10 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                  select_half_operand<2, false, false>(packed_half_10));
                matrix_accumulator_123 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                  select_half_operand<2, false, false>(matrix_fragment_66));
                packed_half_119 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                  select_half_operand<2, false, false>(matrix_accumulator_86));
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_38, address_word_39), packed_half_8, packed_half_9,
                    packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_157 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                  select_half_operand<2, false, false>(0u));
                packed_half_140 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                  select_half_operand<2, false, false>(0u));
                packed_half_138 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                                  select_half_operand<2, false, false>(0u));
                packed_half_150 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                  select_half_operand<2, false, false>(packed_half_157));
                packed_half_139 =
                    pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                                  select_half_operand<2, false, false>(packed_half_140));
                packed_half_138 =
                    pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_129),
                                  select_half_operand<2, false, false>(packed_half_138));
                packed_half_8 = matrix_accumulator_123;
                packed_half_9 = matrix_accumulator_122;
                packed_half_10 = packed_half_119;
                address_word_11 = packed_half_118;
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_38, address_word_39) + 0x200ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = packed_half_137;
                packed_half_9 = packed_half_136;
                packed_half_10 = packed_half_133;
                address_word_11 = packed_half_132;
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_38, address_word_39) + 0x400ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = packed_half_138;
                packed_half_9 = packed_half_139;
                packed_half_10 = packed_half_150;
                address_word_11 = packed_half_151;
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_38, address_word_39) + 0x600ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
            }
            address_word_38 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_47 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_54 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_55 =
                pack_fp8_pair(select_half_operand<2, false, false>(address_word_58),
                              select_half_operand<2, false, false>(0u));
            address_word_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_59),
                                            select_half_operand<2, false, false>(0u));
            packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                          select_half_operand<2, false, false>(0u));
            packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(0u));
            address_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_57),
                                            select_half_operand<2, false, false>(0u));
            packed_half_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_45 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                              select_half_operand<2, false, false>(0u));
            packed_half_10 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_101),
                              select_half_operand<2, false, false>(0u));
            address_word_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_103 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_103),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_104 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_44 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                              select_half_operand<2, false, false>(0u));
            address_word_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                            select_half_operand<2, false, false>(0u));
            matrix_accumulator_113 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_112 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_111 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_46 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_108 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_52 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_53 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                              select_half_operand<2, false, false>(0u));
            address_word_56 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_56),
                                            select_half_operand<2, false, false>(0u));
            matrix_fragment_60 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                              select_half_operand<2, false, false>(0u));
            packed_half_124 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                            select_half_operand<2, false, false>(0u));
            packed_half_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_51),
                                           select_half_operand<2, false, false>(0u));
            packed_half_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_50),
                                           select_half_operand<2, false, false>(0u));
            packed_half_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_49),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_62 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_64 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                              select_half_operand<2, false, false>(0u));
            if (!condition_3) {
                address_word_11 = permute_signed(address_word_11, matrix_accumulator_104, 0x5410u);
                split_address_words(packed_half_36, address_word_37,
                                    int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_36, address_word_37));
                packed_half_10 = permute_signed(packed_half_10, matrix_fragment_103, 0x5410u);
                packed_half_9 = permute_signed(packed_half_9, address_word_38, 0x5410u);
                packed_half_8 = permute_signed(packed_half_8, matrix_fragment_45, 0x5410u);
                matrix_fragment_17 =
                    permute_signed(matrix_fragment_17, matrix_accumulator_108, 0x5410u);
                address_word_38 =
                    permute_signed(matrix_accumulator_111, matrix_fragment_46, 0x5410u);
                store_four_words_streaming<CheckBounds>(
                    combine_words(packed_half_36, address_word_37) + 0x2000ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                address_word_33 = permute_signed(address_word_33, matrix_accumulator_112, 0x5410u);
                matrix_fragment_44 =
                    permute_signed(matrix_fragment_44, matrix_accumulator_113, 0x5410u);
                address_word_39 = permute_signed(address_word_39, address_word_58, 0x5410u);
                matrix_fragment_46 = permute_signed(address_word_56, matrix_fragment_55, 0x5410u);
                matrix_fragment_45 =
                    permute_signed(matrix_fragment_53, matrix_fragment_54, 0x5410u);
                packed_half_50 = permute_signed(matrix_fragment_52, matrix_fragment_47, 0x5410u);
                matrix_fragment_52 = permute_signed(packed_half_49, matrix_fragment_62, 0x5410u);
                matrix_fragment_47 = permute_signed(packed_half_59, matrix_fragment_64, 0x5410u);
                packed_half_49 = permute_signed(packed_half_124, packed_half_57, 0x5410u);
                packed_half_8 = matrix_fragment_44;
                packed_half_9 = address_word_33;
                packed_half_10 = address_word_38;
                address_word_11 = matrix_fragment_17;
                matrix_fragment_54 = permute_signed(matrix_fragment_60, packed_half_51, 0x5410u);
                store_four_words_streaming<CheckBounds>(
                    combine_words(packed_half_36, address_word_37) + 0x2200ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = packed_half_50;
                packed_half_9 = matrix_fragment_45;
                packed_half_10 = matrix_fragment_46;
                address_word_11 = address_word_39;
                store_four_words_streaming<CheckBounds>(
                    combine_words(packed_half_36, address_word_37) + 0x2400ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = matrix_fragment_54;
                packed_half_9 = packed_half_49;
                packed_half_10 = matrix_fragment_52;
                address_word_11 = matrix_fragment_47;
                store_four_words_streaming<CheckBounds>(
                    combine_words(packed_half_36, address_word_37) + 0x2600ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
            }
            matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_18),
                                               select_half_operand<2, false, false>(0u));
            packed_half_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                           select_half_operand<2, false, false>(0u));
            packed_half_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                           select_half_operand<2, false, false>(0u));
            packed_half_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_29),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_31),
                                               select_half_operand<2, false, false>(0u));
            address_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<2, false, false>(0u));
            matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                               select_half_operand<2, false, false>(0u));
            matrix_fragment_15 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                              select_half_operand<2, false, false>(0u));
            packed_half_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_28),
                                           select_half_operand<2, false, false>(0u));
            packed_half_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                           select_half_operand<2, false, false>(0u));
            packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(0u));
            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                              select_half_operand<2, false, false>(0u));
            packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                          select_half_operand<2, false, false>(0u));
            packed_half_36 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_125),
                                           select_half_operand<2, false, false>(0u));
            packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                           select_half_operand<2, false, false>(0u));
            address_word_33 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                              select_half_operand<2, false, false>(0u));
            address_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_126),
                                            select_half_operand<2, false, false>(0u));
            address_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_127),
                                            select_half_operand<2, false, false>(0u));
            matrix_fragment_128 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_128),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_73 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_72 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                              select_half_operand<2, false, false>(0u));
            address_word_39 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_7 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_3),
                                               select_half_operand<2, false, false>(0u));
            matrix_fragment_45 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                              select_half_operand<2, false, false>(0u));
            packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                           select_half_operand<2, false, false>(0u));
            packed_half_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_43 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_42),
                                               select_half_operand<2, false, false>(0u));
            packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                           select_half_operand<2, false, false>(0u));
            if (!condition_1) {
                address_word_3 = permute_signed(matrix_fragment_7, matrix_fragment_6, 0x5410u);
                split_address_words(matrix_fragment_6, matrix_fragment_7,
                                    int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_34, address_word_35));
                address_word_11 = permute_signed(address_word_11, address_word_38, 0x5410u);
                packed_half_10 = permute_signed(packed_half_10, address_word_33, 0x5410u);
                packed_half_9 = permute_signed(packed_half_9, packed_half_36, 0x5410u);
                packed_half_8 = permute_signed(packed_half_8, matrix_fragment_17, 0x5410u);
                matrix_fragment_4 = permute_signed(matrix_fragment_4, address_word_39, 0x5410u);
                matrix_fragment_13 =
                    permute_signed(matrix_fragment_73, matrix_fragment_72, 0x5410u);
                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_6, matrix_fragment_7), packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                matrix_fragment_14 = permute_signed(matrix_fragment_128, address_word_37, 0x5410u);
                matrix_fragment_15 = permute_signed(matrix_fragment_15, packed_half_18, 0x5410u);
                packed_half_16 = permute_signed(packed_half_16, packed_half_49, 0x5410u);
                matrix_fragment_17 =
                    permute_signed(matrix_fragment_46, matrix_fragment_47, 0x5410u);
                packed_half_18 = permute_signed(matrix_fragment_44, matrix_fragment_45, 0x5410u);
                matrix_fragment_19 = permute_signed(matrix_fragment_19, packed_half_30, 0x5410u);
                matrix_fragment_20 = permute_signed(matrix_fragment_20, packed_half_31, 0x5410u);
                packed_half_28 = permute_signed(packed_half_28, matrix_fragment_43, 0x5410u);
                packed_half_29 = permute_signed(packed_half_29, packed_half_50, 0x5410u);
                packed_half_8 = matrix_fragment_14;
                packed_half_9 = matrix_fragment_13;
                packed_half_10 = matrix_fragment_4;
                address_word_11 = address_word_3;
                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_6, matrix_fragment_7) + 0x200ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = packed_half_18;
                packed_half_9 = matrix_fragment_17;
                packed_half_10 = packed_half_16;
                address_word_11 = matrix_fragment_15;
                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_6, matrix_fragment_7) + 0x400ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
                packed_half_8 = packed_half_29;
                packed_half_9 = packed_half_28;
                packed_half_10 = matrix_fragment_20;
                address_word_11 = matrix_fragment_19;
                store_four_words_streaming<CheckBounds>(
                    combine_words(matrix_fragment_6, matrix_fragment_7) + 0x600ull, packed_half_8,
                    packed_half_9, packed_half_10, address_word_11, memory_ranges, error_counts);
            }
            matrix_fragment_4 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                              select_half_operand<2, false, false>(0u));
            packed_half_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_21),
                                          select_half_operand<2, false, false>(0u));
            matrix_fragment_6 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                              select_half_operand<2, false, false>(0u));
            packed_half_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_22),
                                          select_half_operand<2, false, false>(0u));
            matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_76),
                                              select_half_operand<2, false, false>(0u));
            packed_half_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_23),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_77),
                                              select_half_operand<2, false, false>(0u));
            address_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_24),
                                            select_half_operand<2, false, false>(0u));
            address_word_11 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_12 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                              select_half_operand<2, false, false>(0u));
            packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_25),
                                           select_half_operand<2, false, false>(0u));
            matrix_fragment_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(address_word_26),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_80 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_81 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                              select_half_operand<2, false, false>(0u));
            address_word_27 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_27),
                                            select_half_operand<2, false, false>(0u));
            matrix_accumulator_84 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_48),
                                               select_half_operand<2, false, false>(0u));
            address_word_32 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_32),
                                            select_half_operand<2, false, false>(0u));
            matrix_fragment_14 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_17 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                              select_half_operand<2, false, false>(0u));
            packed_half_145 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                            select_half_operand<2, false, false>(0u));
            packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                           select_half_operand<2, false, false>(0u));
            matrix_accumulator_87 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_88 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_19 =
                pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                              select_half_operand<2, false, false>(0u));
            packed_half_144 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                            select_half_operand<2, false, false>(0u));
            packed_half_18 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                              select_half_operand<2, false, false>(0u));
            packed_half_21 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_65 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_98 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                              select_half_operand<2, false, false>(0u));
            matrix_accumulator_91 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                              select_half_operand<2, false, false>(0u));
            matrix_fragment_20 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                              select_half_operand<2, false, false>(0u));
            if (!condition_0) {
                split_address_words(address_word_2, address_word_3,
                                    int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                        combine_words(packed_half_34, address_word_35));
                matrix_fragment_7 = permute_signed(matrix_fragment_7, address_word_24, 0x5410u);
                matrix_fragment_6 = permute_signed(matrix_fragment_6, packed_half_23, 0x5410u);
                matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_8, 0x5410u);
                matrix_fragment_4 = permute_signed(matrix_fragment_4, packed_half_9, 0x5410u);
                packed_half_8 = permute_signed(address_word_27, matrix_accumulator_84, 0x5410u);
                packed_half_9 =
                    permute_signed(matrix_accumulator_80, matrix_accumulator_81, 0x5410u);
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_2, address_word_3) + 0x2000ull, matrix_fragment_4,
                    matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                    error_counts);
                packed_half_10 = permute_signed(packed_half_10, matrix_fragment_13, 0x5410u);
                address_word_11 = permute_signed(address_word_11, matrix_fragment_12, 0x5410u);
                matrix_fragment_12 =
                    permute_signed(matrix_accumulator_87, matrix_accumulator_88, 0x5410u);
                matrix_fragment_13 = permute_signed(packed_half_145, packed_half_16, 0x5410u);
                matrix_fragment_14 =
                    permute_signed(matrix_fragment_14, matrix_fragment_17, 0x5410u);
                matrix_fragment_15 = permute_signed(matrix_fragment_15, address_word_32, 0x5410u);
                packed_half_16 = permute_signed(matrix_accumulator_91, matrix_fragment_20, 0x5410u);
                matrix_fragment_17 =
                    permute_signed(matrix_fragment_65, matrix_accumulator_98, 0x5410u);
                packed_half_18 = permute_signed(packed_half_18, packed_half_21, 0x5410u);
                matrix_fragment_4 = address_word_11;
                matrix_fragment_5 = packed_half_10;
                matrix_fragment_6 = packed_half_9;
                matrix_fragment_7 = packed_half_8;
                matrix_fragment_19 = permute_signed(matrix_fragment_19, packed_half_144, 0x5410u);
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_2, address_word_3) + 0x2200ull, matrix_fragment_4,
                    matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                    error_counts);
                matrix_fragment_4 = matrix_fragment_15;
                matrix_fragment_5 = matrix_fragment_14;
                matrix_fragment_6 = matrix_fragment_13;
                matrix_fragment_7 = matrix_fragment_12;
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_2, address_word_3) + 0x2400ull, matrix_fragment_4,
                    matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                    error_counts);
                matrix_fragment_4 = matrix_fragment_19;
                matrix_fragment_5 = packed_half_18;
                matrix_fragment_6 = matrix_fragment_17;
                matrix_fragment_7 = packed_half_16;
                store_four_words_streaming<CheckBounds>(
                    combine_words(address_word_2, address_word_3) + 0x2600ull, matrix_fragment_4,
                    matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
                    error_counts);
            }
        }
    }
    address_word_2 = threadIdx.z;
    address_word_3 = threadIdx.x;
    matrix_fragment_4 = threadIdx.y;
    phoenix_architecture::finish_compatibility_copies();
    condition_0 = (address_word_2 | address_word_3 | matrix_fragment_4) != 0;
    if (condition_0) {
        return;
    }
    matrix_fragment_5 = blockIdx.z;
    address_word_2 = parameters[parameter_counter_low_8];
    address_word_3 = parameters[parameter_counter_high_9];
    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));
    condition_0 = uint32_t(matrix_fragment_5) != uint32_t(0x3u);
    __threadfence();
    write_device_flag<CheckBounds>(combine_words(address_word_2, address_word_3), matrix_fragment_5,
                                   memory_ranges, error_counts);
    if (condition_0) {
        return;
    }
    address_word_2 = parameters[parameter_publication_low_10];
    address_word_3 = parameters[parameter_publication_high_11];
    split_address_words(address_word_2, address_word_3,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_2, address_word_3));
    __threadfence();
    write_device_flag<CheckBounds>(combine_words(address_word_2, address_word_3), 0u, memory_ranges,
                                   error_counts);
    return;
}

// The selected budget balances resident blocks and instruction scheduling.
extern "C" __global__
__maxnreg__(237) void phoenix_decoder_input_upsampling(KernelArguments parameters) {
    run_decoder_input_upsampling<false>(parameters.words, nullptr, nullptr);
}
