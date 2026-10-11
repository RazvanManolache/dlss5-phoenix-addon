// DLSS5 Phoenix output/reconstruction: native frame-boundary CUDA source.

// Read neural features, reconstruct output channels and write the output surface.

// CUDA 13.1: preserve predicate forms here; simplifying their group changes compiled scheduling.
// Word slots are reused across phases; changing their lifetime or arithmetic order can change GPU code.

// The 64-bit texture handles contain low-word texture and high-word sampler indices.

// Compile to PTX, select independent texture mode, then assemble with a 128-register limit.

#define DLSS5_D3D12_SAMPLER

#include "../../common/texture_sampling.cuh"

#include "../../common/frame_boundary_memory.cuh"

// The by-value parameter block matches the runtime's existing binary launch interface.

struct alignas(8) KernelArguments {

    uint32_t words[46];
};

static_assert(sizeof(KernelArguments) == 184, "Frame-boundary argument layout");

template <bool CheckBounds>

__device__ __forceinline__ void run_output_reconstruction(const uint32_t* parameters,

                                                          const MemoryRange* memory_ranges,

                                                          uint32_t* error_counts) {
    // Shared storage holds the warp tile. Address pairs and packed arithmetic remain bit-exact.

    uint32_t packed_word_0 = 0;
    uint32_t packed_word_1 = 0;
    uint32_t packed_word_2 = 0;
    uint32_t packed_word_3 = 0;
    uint32_t packed_word_4 = 0;
    uint32_t packed_word_5 = 0;
    uint32_t packed_word_6 = 0;
    uint32_t packed_word_7 = 0;
    uint32_t packed_word_8 = 0;
    uint32_t packed_word_9 = 0;
    uint32_t packed_word_10 = 0;
    uint32_t packed_word_11 = 0;
    uint32_t packed_word_12 = 0;
    uint32_t packed_word_13 = 0;
    uint32_t packed_word_14 = 0;
    uint32_t packed_word_15 = 0;
    uint32_t packed_word_16 = 0;
    uint32_t packed_word_17 = 0;
    uint32_t packed_word_18 = 0;
    uint32_t packed_word_19 = 0;
    uint32_t packed_word_20 = 0;
    uint32_t packed_word_21 = 0;
    uint32_t packed_word_22 = 0;
    uint32_t packed_word_23 = 0;
    uint32_t packed_word_24 = 0;
    uint32_t packed_word_25 = 0;
    uint32_t packed_word_26 = 0;
    uint32_t packed_word_27 = 0;
    uint32_t packed_word_28 = 0;
    uint32_t packed_word_29 = 0;
    uint32_t packed_word_30 = 0;
    uint32_t packed_word_31 = 0;
    uint32_t packed_word_32 = 0;
    uint32_t packed_word_33 = 0;
    uint32_t packed_word_34 = 0;
    uint32_t packed_word_35 = 0;
    uint32_t packed_word_36 = 0;
    uint32_t packed_word_37 = 0;
    uint32_t packed_word_38 = 0;
    uint32_t packed_word_39 = 0;
    uint32_t packed_word_40 = 0;
    uint32_t packed_word_41 = 0;
    uint32_t packed_word_42 = 0;
    uint32_t packed_word_43 = 0;
    uint32_t packed_word_44 = 0;
    uint32_t packed_word_45 = 0;
    uint32_t packed_word_46 = 0;
    uint32_t packed_word_47 = 0;
    uint32_t packed_word_48 = 0;
    uint32_t packed_word_49 = 0;
    uint32_t packed_word_50 = 0;
    uint32_t packed_word_51 = 0;
    uint32_t packed_word_52 = 0;
    uint32_t packed_word_53 = 0;
    uint32_t packed_word_54 = 0;
    uint32_t packed_word_55 = 0;
    uint32_t packed_word_56 = 0;
    uint32_t packed_word_57 = 0;
    uint32_t packed_word_58 = 0;
    uint32_t packed_word_59 = 0;
    uint32_t packed_word_60 = 0;
    uint32_t packed_word_61 = 0;
    uint32_t packed_word_62 = 0;
    uint32_t packed_word_63 = 0;
    uint32_t packed_word_64 = 0;
    uint32_t packed_word_65 = 0;
    uint32_t packed_word_66 = 0;
    uint32_t packed_word_67 = 0;
    uint32_t packed_word_68 = 0;
    uint32_t packed_word_69 = 0;
    uint32_t packed_word_70 = 0;
    uint32_t packed_word_71 = 0;
    uint32_t packed_word_72 = 0;
    uint32_t packed_word_73 = 0;
    uint32_t packed_word_74 = 0;
    uint32_t packed_word_75 = 0;
    uint32_t packed_word_76 = 0;
    uint32_t packed_word_77 = 0;
    uint32_t packed_word_78 = 0;
    uint32_t packed_word_79 = 0;
    uint32_t packed_word_80 = 0;
    uint32_t packed_word_81 = 0;
    uint32_t packed_word_82 = 0;
    uint32_t packed_word_83 = 0;
    uint32_t packed_word_84 = 0;
    uint32_t packed_word_85 = 0;
    uint32_t packed_word_86 = 0;
    uint32_t packed_word_87 = 0;
    uint32_t packed_word_88 = 0;
    uint32_t packed_word_89 = 0;
    uint32_t packed_word_90 = 0;
    uint32_t packed_word_91 = 0;
    uint32_t packed_word_92 = 0;
    uint32_t packed_word_93 = 0;
    uint32_t packed_word_94 = 0;
    uint32_t packed_word_95 = 0;
    uint32_t packed_word_96 = 0;
    uint32_t packed_word_97 = 0;
    uint32_t packed_word_98 = 0;
    uint32_t packed_word_99 = 0;
    uint32_t packed_word_100 = 0;
    uint32_t packed_word_101 = 0;
    uint32_t packed_word_102 = 0;
    uint32_t packed_word_103 = 0;
    uint32_t packed_word_104 = 0;
    uint32_t packed_word_105 = 0;
    uint32_t packed_word_106 = 0;
    uint32_t packed_word_107 = 0;
    uint32_t packed_word_108 = 0;
    uint32_t packed_word_109 = 0;
    uint32_t packed_word_110 = 0;
    uint32_t packed_word_111 = 0;
    uint32_t packed_word_112 = 0;
    uint32_t packed_word_113 = 0;
    uint32_t packed_word_114 = 0;
    uint32_t packed_word_115 = 0;
    uint32_t packed_word_116 = 0;
    uint32_t packed_word_117 = 0;
    uint32_t packed_word_118 = 0;
    uint32_t packed_word_119 = 0;
    uint32_t packed_word_120 = 0;
    uint32_t packed_word_121 = 0;
    uint32_t packed_word_122 = 0;
    uint32_t packed_word_123 = 0;
    uint32_t packed_word_124 = 0;
    uint32_t packed_word_125 = 0;
    uint32_t packed_word_126 = 0;
    uint32_t packed_word_127 = 0;
    uint32_t packed_word_128 = 0;
    uint32_t packed_word_129 = 0;
    uint32_t packed_word_130 = 0;
    uint32_t packed_word_131 = 0;
    uint32_t packed_word_132 = 0;
    uint32_t packed_word_133 = 0;
    uint32_t packed_word_134 = 0;
    uint32_t packed_word_135 = 0;
    uint32_t packed_word_136 = 0;
    uint32_t packed_word_137 = 0;
    uint32_t packed_word_138 = 0;
    uint32_t packed_word_139 = 0;
    uint32_t packed_word_140 = 0;
    uint32_t packed_word_141 = 0;
    uint32_t packed_word_142 = 0;
    uint32_t packed_word_143 = 0;
    uint32_t packed_word_144 = 0;
    uint32_t packed_word_145 = 0;
    uint32_t packed_word_146 = 0;
    uint32_t packed_word_147 = 0;
    uint32_t packed_word_148 = 0;
    uint32_t packed_word_149 = 0;
    uint32_t packed_word_150 = 0;
    uint32_t packed_word_151 = 0;
    uint32_t packed_word_152 = 0;
    uint32_t packed_word_153 = 0;
    uint32_t packed_word_154 = 0;
    uint32_t packed_word_155 = 0;
    uint32_t packed_word_156 = 0;
    uint32_t packed_word_157 = 0;
    uint32_t packed_word_158 = 0;
    uint32_t packed_word_159 = 0;
    uint32_t packed_word_160 = 0;
    uint32_t packed_word_161 = 0;
    uint32_t packed_word_162 = 0;
    uint32_t packed_word_163 = 0;
    uint32_t packed_word_164 = 0;
    uint32_t packed_word_165 = 0;
    uint32_t packed_word_255 = 0;
    uint32_t uniform_word_4 = 0;
    uint32_t uniform_word_5 = 0;
    uint32_t uniform_word_6 = 0;
    uint32_t uniform_word_7 = 0;
    uint32_t uniform_word_8 = 0;
    uint32_t uniform_word_9 = 0;
    uint32_t uniform_word_10 = 0;
    uint32_t uniform_word_11 = 0;
    uint32_t uniform_word_12 = 0;
    uint32_t uniform_word_13 = 0;
    uint32_t uniform_word_14 = 0;
    uint32_t uniform_word_15 = 0;
    uint32_t uniform_word_79 = 0;
    uint32_t angle_bits_1 = 0;
    uint64_t copy_tokens[2] = {0, 0};
    bool condition_0 = false, condition_1 = false, condition_2 = false, condition_3 = false,
         condition_4 = false, condition_5 = false, condition_6 = false, uniform_condition_0 = false,
         uniform_condition_1 = false, uniform_condition_2 = false, uniform_condition_3 = false,
         uniform_condition_4 = false, uniform_condition_5 = false, uniform_condition_6 = false;
    constexpr uint32_t shared_bytes = 4;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];
    packed_word_1 = 0u;
    packed_word_0 = blockIdx.y;
    packed_word_20 = parameters[8];
    condition_0 = true;

    packed_word_164 = threadIdx.x & 31u;
    packed_word_5 = blockIdx.x;
    packed_word_10 = parameters[10];
    packed_word_11 = parameters[11];
    packed_word_19 = parameters[9];
    packed_word_2 = packed_word_20 & 0xfffffffeu;
    packed_word_3 = uint32_t(combine_words(packed_word_20, 0u) << 0x1u >> 32) + packed_word_20;
    condition_2 = uint32_t(packed_word_2) != uint32_t(0x2u);
    packed_word_8 = uint32_t(int32_t(packed_word_3) >> 0x1u);
    packed_word_0 = packed_word_0 * 0x8u + packed_word_11;
    packed_word_4 = packed_word_164;
    packed_word_2 = uint32_t(combine_words(packed_word_0, 0u) << 0x1u >> 32) + packed_word_0;
    packed_word_165 = packed_word_164;
    packed_word_10 = packed_word_5 * 0x8u + packed_word_10;
    packed_word_2 = uint32_t(int32_t(packed_word_2) >> 0x1u);
    packed_word_3 = uint32_t(int32_t(packed_word_165) >> 0x2u);
    packed_word_7 =
        uint32_t(int64_t(int32_t(packed_word_4)) * (1ll << 0x1cu) >> 32) + packed_word_2;
    packed_word_2 = uint32_t(combine_words(packed_word_10, 0u) << 0x1u >> 32) + packed_word_10;
    packed_word_4 = uint32_t(combine_words(packed_word_3, 0u) << 0x2u >> 32) + packed_word_3;
    if (condition_2) {
        condition_1 = int32_t(packed_word_7) >= int32_t(packed_word_8);
    }
    packed_word_15 = packed_word_7 + 0x2u;
    packed_word_2 = uint32_t(int32_t(packed_word_2) >> 0x1u);
    packed_word_4 = packed_word_4 & 0xfffffffcu;
    if (condition_2) {
        condition_0 = int32_t(packed_word_7) > -1 && !condition_1;

        condition_3 = int32_t(packed_word_15) >= int32_t(packed_word_8);
    }
    condition_1 = true;
    packed_word_6 = packed_word_2 + packed_word_3 + (0u - packed_word_4);
    if (condition_2) {
        packed_word_9 = condition_0 ? 0u : 0x1u;

        condition_1 = int32_t(packed_word_15) > -1 && !condition_3;
    }
    packed_word_5 = permute_signed(0u, packed_word_5, 0x7610u);
    packed_word_2 = uint32_t(combine_words(packed_word_19, 0u) << 0x1u >> 32) + packed_word_19;
    if (condition_2) {
        packed_word_5 = permute_signed(packed_word_9, packed_word_5, 0x7610u);
    }
    packed_word_3 = permute_signed(0u, packed_word_3, 0x7610u);
    packed_word_18 = uint32_t(int32_t(packed_word_2) >> 0x1u);
    packed_word_14 = packed_word_6;
    if (condition_2) {
        packed_word_4 = condition_1 ? 0u : 0x1u;
    }
    packed_word_11 = packed_word_6;
    if (condition_0) {
        packed_word_2 = packed_word_19 & 0xfffffffeu;
        packed_word_11 = 0u;
        condition_3 = uint32_t(packed_word_2) != uint32_t(0x2u);
        if (condition_3) {
            condition_4 = int32_t(packed_word_6) >= int32_t(packed_word_18);

            condition_0 = int32_t(packed_word_6) > -1;

            packed_word_11 = packed_word_6;

            packed_word_2 = !condition_4 ? packed_word_5 : 0x1u;

            packed_word_2 = condition_0 ? packed_word_2 : 0x1u;

            packed_word_5 = permute_signed(packed_word_2, packed_word_5, 0x7610u);
        }
    }

    packed_word_5 = packed_word_5 & 0xffffu;
    if (condition_2) {
        packed_word_3 = permute_signed(packed_word_4, packed_word_3, 0x7610u);
    }
    if (condition_1) {
        packed_word_2 = packed_word_19 & 0xfffffffeu;
        packed_word_14 = 0u;
        condition_0 = uint32_t(packed_word_2) != uint32_t(0x2u);
        if (condition_0) {
            condition_3 = int32_t(packed_word_6) >= int32_t(packed_word_18);

            condition_1 = int32_t(packed_word_6) > -1;

            packed_word_14 = packed_word_6;

            packed_word_2 = !condition_3 ? packed_word_3 : 0x1u;

            packed_word_2 = condition_1 ? packed_word_2 : 0x1u;

            packed_word_3 = permute_signed(packed_word_2, packed_word_3, 0x7610u);
        }
    }

    packed_word_3 = packed_word_3 & 0xffffu;

    condition_4 = uint32_t(packed_word_5) != 0u;
    if (condition_2) {
        condition_5 = int32_t(packed_word_7) >= int32_t(packed_word_8);
    }
    condition_1 = true;
    condition_3 = uint32_t(packed_word_3) != 0u;
    packed_word_9 = 0u;
    if (condition_2) {
        condition_1 = int32_t(packed_word_7) > -1 && !condition_5;

        packed_word_9 = packed_word_7;

        condition_6 = int32_t(packed_word_15) >= int32_t(packed_word_8);
    }
    packed_word_16 = permute_signed(0u, packed_word_16, 0x7610u);
    if (condition_2) {
        packed_word_17 = condition_1 ? 0u : 0x1u;
    }
    condition_0 = true;
    if (!condition_4) {
        packed_word_2 = parameters[0];
        packed_word_3 = parameters[1];

        packed_word_13 = packed_word_165 & 0xfffffffcu;

        packed_word_9 = packed_word_9 * packed_word_18;
    }
    if (condition_2) {
        condition_0 = int32_t(packed_word_15) > -1 && !condition_6;

        packed_word_16 = permute_signed(packed_word_17, packed_word_16, 0x7610u);
    }
    if (!condition_4) {
        packed_word_12 = 0u - packed_word_13 + packed_word_164;
    }
    packed_word_13 = packed_word_6;
    if (!condition_3) {
        packed_word_4 = parameters[0];
        packed_word_5 = parameters[1];
    }
    if (!condition_4) {
        packed_word_12 = packed_word_9 * 0x4u + packed_word_12;
    }
    if (condition_1) {
        packed_word_9 = packed_word_19 & 0xfffffffeu;
        packed_word_13 = 0u;
        condition_1 = uint32_t(packed_word_9) != uint32_t(0x2u);
        if (condition_1) {
            condition_6 = int32_t(packed_word_6) >= int32_t(packed_word_18);

            condition_5 = int32_t(packed_word_6) > -1;

            packed_word_13 = packed_word_6;

            packed_word_9 = !condition_6 ? packed_word_16 : 0x1u;

            packed_word_9 = condition_5 ? packed_word_9 : 0x1u;

            packed_word_16 = permute_signed(packed_word_9, packed_word_16, 0x7610u);
        }
    }

    packed_word_17 = packed_word_16 & 0xffffu;
    if (!condition_4) {
        packed_word_9 = packed_word_11 * 0x4u + packed_word_12;
    }
    if (condition_2) {
        packed_word_21 = condition_0 ? 0u : 0x1u;
    }
    packed_word_16 = permute_signed(0u, packed_word_16, 0x7610u);
    uniform_word_4 = 0u;
    uniform_word_5 = 0u;

    condition_1 = uint32_t(packed_word_17) != 0u;
    packed_word_11 = 0u;
    if (condition_2) {
        packed_word_16 = permute_signed(packed_word_21, packed_word_16, 0x7610u);
    }
    if (!condition_4) {
        split_address_words(packed_word_2, packed_word_3,
                            int64_t(int32_t(packed_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_2, packed_word_3));
    }
    packed_word_38 = 0u;
    if (condition_2) {
        packed_word_11 = packed_word_15;
    }
    if (!condition_3) {
        packed_word_17 = packed_word_165 & 0xfffffffcu;
    }
    packed_word_12 = packed_word_6;
    if (condition_0) {
        packed_word_9 = packed_word_19 & 0xfffffffeu;
        packed_word_12 = 0u;
        condition_0 = uint32_t(packed_word_9) != uint32_t(0x2u);
        if (condition_0) {
            condition_6 = int32_t(packed_word_6) >= int32_t(packed_word_18);

            condition_5 = int32_t(packed_word_6) > -1;

            packed_word_12 = packed_word_6;

            packed_word_9 = !condition_6 ? packed_word_16 : 0x1u;

            packed_word_9 = condition_5 ? packed_word_9 : 0x1u;

            packed_word_16 = permute_signed(packed_word_9, packed_word_16, 0x7610u);
        }
    }

    if (!condition_3) {
        packed_word_6 = 0u - packed_word_17 + packed_word_164;
    }
    packed_word_16 = packed_word_16 & 0xffffu;
    if (!condition_3) {
        packed_word_11 = packed_word_11 * packed_word_18;
    }
    packed_word_9 = 0u;
    if (condition_2) {
        packed_word_9 = packed_word_7;
    }
    if (!condition_4) {
        packed_word_38 = read_boundary_word<CheckBounds>(
            combine_words(packed_word_2, packed_word_3), memory_ranges, error_counts);
    }
    if (!condition_3) {
        packed_word_11 = packed_word_11 * 0x4u + packed_word_6;
    }
    condition_0 = uint32_t(packed_word_16) != 0u;
    if (!condition_1) {
        packed_word_6 = parameters[0];
        packed_word_7 = parameters[1];
    }
    packed_word_17 = 0u;
    if (condition_2) {
        packed_word_17 = packed_word_15;
    }
    if (!condition_1) {
        packed_word_9 = packed_word_8 + packed_word_9;
    }
    if (!condition_3) {
        packed_word_11 = packed_word_14 * 0x4u + packed_word_11;
    }
    packed_word_39 = 0u;
    packed_word_23 = packed_word_165 & 0xfffffffcu;
    packed_word_36 = 0;
    packed_word_37 = 0;
    if (!condition_1) {
        packed_word_3 = packed_word_165 & 0xfffffffcu;

        packed_word_2 = packed_word_9 * packed_word_18;
    }
    packed_word_14 = packed_word_20 & 0xfffffffcu;
    if (!condition_3) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_4, packed_word_5));
    }
    if (!condition_0) {
        packed_word_17 = packed_word_8 + packed_word_17;
    }
    if (!condition_1) {
        packed_word_3 = 0u - packed_word_3 + packed_word_164;
    }
    if (!condition_0) {
        packed_word_8 = parameters[0];
        packed_word_9 = parameters[1];
    }
    packed_word_15 = 0u - packed_word_23 + packed_word_164;
    if (!condition_3) {
        packed_word_39 = read_boundary_word<CheckBounds>(
            combine_words(packed_word_4, packed_word_5), memory_ranges, error_counts);
    }
    packed_word_11 = uint32_t(int32_t(packed_word_20) >> 0x1fu);
    if (!condition_1) {
        packed_word_2 = packed_word_2 * 0x4u + packed_word_3;
    }
    condition_4 = uint32_t(packed_word_14) != uint32_t(0x4u);
    if (!condition_0) {
        packed_word_17 = packed_word_17 * packed_word_18;
    }
    packed_word_11 = uint32_t(combine_words(packed_word_11, 0u) << 0x2u >> 32) + packed_word_20;
    if (!condition_1) {
        packed_word_3 = packed_word_13 * 0x4u + packed_word_2;
    }
    packed_word_32 = packed_word_19 & 0xfffffffcu;
    if (!condition_0) {
        packed_word_17 = packed_word_17 * 0x4u + packed_word_15;
    }
    packed_word_21 = uint32_t(int32_t(packed_word_11) >> 0x2u);
    if (!condition_1) {
        split_address_words(packed_word_2, packed_word_3,
                            int64_t(int32_t(packed_word_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_6, packed_word_7));
    }
    packed_word_5 = uint32_t(int32_t(packed_word_0) >> 0x1fu);
    uniform_word_8 = parameters[8];
    uniform_word_9 = parameters[9];
    packed_word_4 = uint32_t(int32_t(packed_word_19) >> 0x1fu);
    if (!condition_0) {
        packed_word_17 = packed_word_12 * 0x4u + packed_word_17;
    }
    packed_word_5 = uint32_t(combine_words(packed_word_5, 0u) << 0x2u >> 32) + packed_word_0;
    if (!condition_1) {
        packed_word_37 = read_boundary_word<CheckBounds>(
            combine_words(packed_word_2, packed_word_3), memory_ranges, error_counts);
    }
    packed_word_7 = uint32_t(int32_t(packed_word_10) >> 0x1fu);
    uniform_word_6 = parameters[43];
    packed_word_6 = uint32_t(int32_t(packed_word_5) >> 0x2u);
    packed_word_12 = uint32_t(combine_words(packed_word_4, 0u) << 0x2u >> 32) + packed_word_19;
    uniform_word_7 = parameters[44];
    packed_word_7 = uint32_t(combine_words(packed_word_7, 0u) << 0x2u >> 32) + packed_word_10;
    if (condition_4) {
        condition_2 = int32_t(packed_word_6) >= int32_t(packed_word_21);
    }
    packed_word_35 = uint32_t(int32_t(packed_word_12) >> 0x2u);
    packed_word_7 = uint32_t(int32_t(packed_word_7) >> 0x2u);
    if (!condition_0) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_8, packed_word_9));
    }
    condition_1 = false;
    if (condition_4) {
        condition_1 = int32_t(packed_word_0) < int32_t(0u - 0x3u) || condition_2;
    }
    condition_2 = int32_t(packed_word_7) >= int32_t(packed_word_35);
    if (!condition_0) {
        packed_word_36 = read_boundary_word<CheckBounds>(
            combine_words(packed_word_4, packed_word_5), memory_ranges, error_counts);
    }
    condition_0 = true;
    condition_6 = uint32_t(packed_word_32) == uint32_t(0x4u) || condition_1;
    condition_2 = int32_t(packed_word_10) > int32_t(0u - 0x4u) && !condition_2;
    if (condition_4) {
        condition_0 = !condition_1;
    }
    packed_word_34 = packed_word_7 + 0x1u;
    if (condition_4) {
        condition_5 = int32_t(packed_word_6) >= int32_t(packed_word_21);
    }
    condition_1 = condition_1 ^ condition_6;
    condition_0 = condition_0 & (condition_6 | condition_2);
    condition_6 = false;
    condition_3 = int32_t(packed_word_34) >= int32_t(packed_word_35);
    if (condition_4) {
        condition_6 = int32_t(packed_word_0) < int32_t(0u - 0x3u) || condition_5;
    }
    packed_word_9 = !condition_1 ? packed_word_7 : 0u;
    condition_1 = true;
    condition_3 = int32_t(packed_word_10) > int32_t(0u - 0x8u) && !condition_3;
    condition_5 = uint32_t(packed_word_32) == uint32_t(0x4u) || condition_6;
    if (condition_4) {
        condition_1 = !condition_6;
    }
    if (condition_0) {
        packed_word_2 = parameters[2];
        packed_word_3 = parameters[3];
    }
    condition_1 = condition_1 & (condition_5 | condition_3);
    packed_word_8 = 0u;
    if (condition_4) {
        packed_word_8 = packed_word_6 * packed_word_35;
    }
    condition_5 = condition_6 ^ condition_5;
    if (condition_1) {
        packed_word_4 = parameters[2];
        packed_word_5 = parameters[3];
    }
    packed_word_10 = 0u;
    if (condition_0) {
        packed_word_8 = packed_word_9 + packed_word_8;
    }
    if (condition_4) {
        packed_word_10 = packed_word_6 * packed_word_35;
    }
    packed_word_11 = !condition_5 ? packed_word_34 : 0u;
    if (condition_0) {
        packed_word_9 = packed_word_8 * 0x80u;
    }
    if (condition_1) {
        packed_word_8 = packed_word_11 + packed_word_10;
    }
    if (!condition_0) {
        packed_word_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                       select_half_operand<2, false, false>(0u));
    }
    if (condition_0) {
        split_address_words(packed_word_2, packed_word_3,
                            int64_t(int32_t(packed_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_2, packed_word_3));
    }
    if (!condition_1) {
        packed_word_31 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                       select_half_operand<2, false, false>(0u));
    }
    packed_word_10 = parameters[6];
    packed_word_11 = parameters[7];
    if (condition_4) {
        packed_word_12 = packed_word_6 + 0x1u;
    }
    if (condition_1) {
        packed_word_9 = packed_word_8 * 0x80u;
    }
    if (!condition_0) {
        packed_word_27 = permute_signed(packed_word_27, packed_word_27, 0x5410u);
    }
    if (!condition_1) {
        packed_word_31 = permute_signed(packed_word_31, packed_word_31, 0x5410u);
    }
    if (condition_4) {
        condition_6 = int32_t(packed_word_12) >= int32_t(packed_word_21);
    }
    if (condition_1) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_4, packed_word_5));
    }
    if (!condition_0) {
        packed_word_24 = packed_word_27;

        packed_word_25 = packed_word_27;

        packed_word_26 = packed_word_27;
    }
    if (!condition_1) {
        packed_word_28 = packed_word_31;

        packed_word_29 = packed_word_31;

        packed_word_30 = packed_word_31;
    }
    if (condition_0) {
        split_address_words(packed_word_2, packed_word_3,
                            int64_t(int32_t(packed_word_164)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_2, packed_word_3));
    }
    condition_5 = false;
    if (condition_4) {
        condition_5 = int32_t(packed_word_0) < int32_t(0u - 0x7u) || condition_6;
    }
    if (condition_1) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_164)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_4, packed_word_5));
    }
    if (condition_0) {
        load_four_boundary_words<CheckBounds, 1>(combine_words(packed_word_2, packed_word_3),
                                                 packed_word_24, packed_word_25, packed_word_26,
                                                 packed_word_27, memory_ranges, error_counts);
    }
    condition_0 = true;
    if (condition_1) {
        load_four_boundary_words<CheckBounds, 1>(combine_words(packed_word_4, packed_word_5),
                                                 packed_word_28, packed_word_29, packed_word_30,
                                                 packed_word_31, memory_ranges, error_counts);
    }
    condition_1 = uint32_t(packed_word_32) == uint32_t(0x4u) || condition_5;
    if (condition_4) {
        condition_0 = !condition_5;

        packed_word_20 = packed_word_6 + 0x1u;
    }
    condition_2 = condition_0 & (condition_1 | condition_2);
    condition_1 = condition_5 ^ condition_1;
    if (condition_4) {
        condition_0 = int32_t(packed_word_20) >= int32_t(packed_word_21);
    }
    packed_word_7 = !condition_1 ? packed_word_7 : 0u;
    condition_1 = false;
    if (condition_4) {
        condition_1 = int32_t(packed_word_0) < int32_t(0u - 0x7u) || condition_0;
    }
    condition_0 = true;
    if (condition_2) {
        packed_word_4 = parameters[2];
        packed_word_5 = parameters[3];
    }
    condition_5 = uint32_t(packed_word_32) == uint32_t(0x4u) || condition_1;
    if (condition_4) {
        condition_0 = !condition_1;
    }
    split_address_words(packed_word_12, packed_word_13,
                        int64_t(int32_t(packed_word_15)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_word_10, packed_word_11));
    packed_word_15 = packed_word_164 + 0x4u + (0u - packed_word_23);
    packed_word_17 = packed_word_164 + 0x8u + (0u - packed_word_23);
    condition_0 = condition_0 & (condition_5 | condition_3);
    split_address_words(packed_word_14, packed_word_15,
                        uint64_t(uint32_t(packed_word_15)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_10, packed_word_11));
    packed_word_2 = 0u;
    packed_word_9 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_12, packed_word_13) + 0x2050ull, memory_ranges, error_counts);
    split_address_words(packed_word_16, packed_word_17,
                        uint64_t(uint32_t(packed_word_17)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_10, packed_word_11));
    packed_word_8 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_14, packed_word_15) + 0x2050ull, memory_ranges, error_counts);
    if (condition_4) {
        packed_word_2 = packed_word_6 * packed_word_35 + packed_word_35;
    }
    packed_word_3 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_16, packed_word_17) + 0x2050ull, memory_ranges, error_counts);
    packed_word_19 = packed_word_164 + 0xcu + (0u - packed_word_23);
    if (condition_0) {
        packed_word_32 = parameters[2];
        packed_word_33 = parameters[3];
    }
    if (condition_2) {
        packed_word_2 = packed_word_7 + packed_word_2;
    }
    packed_word_103 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_12, packed_word_13) + 0x2090ull, memory_ranges, error_counts);
    if (!condition_2) {
        packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                       select_half_operand<2, false, false>(0u));
    }
    split_address_words(packed_word_18, packed_word_19,
                        uint64_t(uint32_t(packed_word_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_10, packed_word_11));
    packed_word_99 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_14, packed_word_15) + 0x2090ull, memory_ranges, error_counts);
    condition_1 = condition_1 ^ condition_5;
    if (condition_2) {
        packed_word_7 = packed_word_2 * 0x80u;
    }
    packed_word_96 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_16, packed_word_17) + 0x2090ull, memory_ranges, error_counts);
    if (!condition_2) {
        packed_word_23 = permute_signed(packed_word_23, packed_word_23, 0x5410u);
    }
    if (condition_2) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_4, packed_word_5));
    }
    packed_word_7 = 0u;
    packed_word_2 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_18, packed_word_19) + 0x2050ull, memory_ranges, error_counts);
    packed_word_34 = !condition_1 ? packed_word_34 : 0u;
    if (condition_4) {
        packed_word_7 = packed_word_6 * packed_word_35 + packed_word_35;
    }
    if (!condition_2) {
        packed_word_20 = packed_word_23;
    }
    packed_word_88 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_18, packed_word_19) + 0x2090ull, memory_ranges, error_counts);
    if (!condition_2) {
        packed_word_21 = packed_word_23;

        packed_word_22 = packed_word_23;
    }
    if (condition_2) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_164)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_4, packed_word_5));
    }
    if (condition_0) {
        packed_word_7 = packed_word_34 + packed_word_7;
    }
    {
        uint32_t loaded_word;
        loaded_word = packed_word_164 & 0x10u;
        condition_3 = loaded_word != 0;
    }
    packed_word_100 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_12, packed_word_13) + 0x2010ull, memory_ranges, error_counts);
    if (condition_0) {
        packed_word_7 = packed_word_7 * 0x80u;
    }
    if (condition_2) {
        load_four_boundary_words<CheckBounds, 1>(combine_words(packed_word_4, packed_word_5),
                                                 packed_word_20, packed_word_21, packed_word_22,
                                                 packed_word_23, memory_ranges, error_counts);
    }
    split_address_words(packed_word_10, packed_word_11,
                        int64_t(int32_t(packed_word_164)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_word_10, packed_word_11));
    packed_word_91 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_16, packed_word_17) + 0x2010ull, memory_ranges, error_counts);
    if (condition_0) {
        split_address_words(packed_word_32, packed_word_33,
                            int64_t(int32_t(packed_word_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_32, packed_word_33));
    }
    load_four_boundary_words<CheckBounds, 0>(combine_words(packed_word_10, packed_word_11),
                                             packed_word_40, packed_word_41, packed_word_42,
                                             packed_word_43, memory_ranges, error_counts);
    if (!condition_0) {
        packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));
    }
    packed_word_89 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_18, packed_word_19) + 0x2010ull, memory_ranges, error_counts);
    if (!condition_0) {
        packed_word_7 = permute_signed(packed_word_4, packed_word_4, 0x5410u);
    }
    if (condition_0) {
        split_address_words(packed_word_44, packed_word_45,
                            int64_t(int32_t(packed_word_164)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_32, packed_word_33));
    }
    if (!condition_0) {
        packed_word_4 = packed_word_7;

        packed_word_5 = packed_word_7;

        packed_word_6 = packed_word_7;
    }
    if (condition_0) {
        load_four_boundary_words<CheckBounds, 1>(combine_words(packed_word_44, packed_word_45),
                                                 packed_word_4, packed_word_5, packed_word_6,
                                                 packed_word_7, memory_ranges, error_counts);
    }
    packed_word_32 = packed_word_164 * 0x2u;
    packed_word_33 = uint32_t(uint32_t(packed_word_164) >> 0x1u);
    packed_word_47 = packed_word_32 & 0x8u;
    packed_word_33 = packed_word_33 & 0x4u;
    packed_word_32 = packed_word_47 | 0x13u & packed_word_164;
    packed_word_46 = packed_word_33 | 0x13u & packed_word_164;
    packed_word_51 = packed_word_32 | packed_word_33;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x200ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    packed_word_47 = packed_word_46 | packed_word_47;
    packed_word_55 = 0x10u ^ packed_word_51;
    packed_word_53 = 0x8u ^ packed_word_47;
    packed_word_57 = 0x18u ^ packed_word_47;
    {
        uint32_t loaded_word;
        loaded_word = packed_word_164 & 0x4u;
        condition_0 = loaded_word != 0;
    }
    packed_word_46 = unpack_fp8_bits(packed_word_38);
    packed_word_38 = unpack_fp8_bits(packed_word_38 >> 16);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_46), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_45 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_46), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_48 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_46), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_47 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_46), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_50 = loaded_word;
    }
    packed_word_44 = unpack_fp8_bits(packed_word_39);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_49 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_58 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_52 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_63 = loaded_word;
    }
    packed_word_39 = unpack_fp8_bits(packed_word_39 >> 16);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_54 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_59 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_56 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_61 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_39), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_60 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_39), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_65 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_39), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_62 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_39), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_67 = loaded_word;
    }
    packed_word_46 = !condition_3 ? packed_word_45 : packed_word_48;
    packed_word_48 = !condition_3 ? packed_word_48 : packed_word_45;
    packed_word_45 = !condition_3 ? packed_word_47 : packed_word_50;
    packed_word_38 = !condition_3 ? packed_word_49 : packed_word_58;
    packed_word_58 = !condition_3 ? packed_word_58 : packed_word_49;
    packed_word_49 = !condition_3 ? packed_word_52 : packed_word_63;
    packed_word_47 = !condition_3 ? packed_word_50 : packed_word_47;
    packed_word_63 = !condition_3 ? packed_word_63 : packed_word_52;
    packed_word_50 = !condition_0 ? packed_word_46 : packed_word_45;
    packed_word_46 = !condition_0 ? packed_word_45 : packed_word_46;
    packed_word_45 = !condition_0 ? packed_word_38 : packed_word_49;
    packed_word_39 = !condition_0 ? packed_word_49 : packed_word_38;
    packed_word_44 = !condition_3 ? packed_word_54 : packed_word_59;
    packed_word_52 = !condition_0 ? packed_word_48 : packed_word_47;
    packed_word_54 = !condition_3 ? packed_word_59 : packed_word_54;
    packed_word_38 = unpack_fp8_bits(packed_word_37);
    packed_word_48 = !condition_0 ? packed_word_47 : packed_word_48;
    packed_word_59 = !condition_3 ? packed_word_56 : packed_word_61;
    packed_word_37 = unpack_fp8_bits(packed_word_37 >> 16);
    packed_word_47 = !condition_0 ? packed_word_58 : packed_word_63;
    packed_word_49 = !condition_0 ? packed_word_63 : packed_word_58;
    packed_word_63 = !condition_3 ? packed_word_60 : packed_word_65;
    packed_word_65 = !condition_3 ? packed_word_65 : packed_word_60;
    packed_word_61 = !condition_3 ? packed_word_61 : packed_word_56;
    packed_word_60 = !condition_3 ? packed_word_62 : packed_word_67;
    packed_word_62 = !condition_3 ? packed_word_67 : packed_word_62;
    packed_word_56 = !condition_0 ? packed_word_44 : packed_word_59;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_37), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_72 = loaded_word;
    }
    packed_word_104 = !condition_0 ? packed_word_59 : packed_word_44;
    packed_word_44 = unpack_fp8_bits(packed_word_36);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_37), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_75 = loaded_word;
    }
    packed_word_36 = unpack_fp8_bits(packed_word_36 >> 16);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_37), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_74 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_37), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_77 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_64 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_67 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_66 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_69 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_70 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_73 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_68 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_71 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_36), "r"(packed_word_51), "r"(0x1fu), "r"(__activemask()));
        packed_word_76 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_36), "r"(packed_word_53), "r"(0x1fu), "r"(__activemask()));
        packed_word_78 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_36), "r"(packed_word_55), "r"(0x1fu), "r"(__activemask()));
        packed_word_79 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_36), "r"(packed_word_57), "r"(0x1fu), "r"(__activemask()));
        packed_word_81 = loaded_word;
    }
    packed_word_58 = !condition_0 ? packed_word_54 : packed_word_61;
    packed_word_54 = !condition_0 ? packed_word_61 : packed_word_54;
    packed_word_44 = !condition_3 ? packed_word_72 : packed_word_75;
    packed_word_51 = !condition_3 ? packed_word_74 : packed_word_77;
    packed_word_38 = !condition_3 ? packed_word_64 : packed_word_67;
    packed_word_61 = !condition_3 ? packed_word_66 : packed_word_69;
    packed_word_53 = !condition_0 ? packed_word_44 : packed_word_51;
    packed_word_55 = !condition_0 ? packed_word_51 : packed_word_44;
    packed_word_36 = !condition_0 ? packed_word_38 : packed_word_61;
    packed_word_51 = !condition_3 ? packed_word_70 : packed_word_73;
    packed_word_38 = !condition_0 ? packed_word_61 : packed_word_38;
    packed_word_44 = !condition_3 ? packed_word_68 : packed_word_71;
    packed_word_73 = !condition_3 ? packed_word_73 : packed_word_70;
    packed_word_37 = !condition_0 ? packed_word_65 : packed_word_62;
    packed_word_65 = !condition_0 ? packed_word_62 : packed_word_65;
    packed_word_61 = !condition_3 ? packed_word_76 : packed_word_79;
    packed_word_64 = !condition_3 ? packed_word_67 : packed_word_64;
    packed_word_70 = !condition_3 ? packed_word_78 : packed_word_81;
    packed_word_69 = !condition_3 ? packed_word_69 : packed_word_66;
    packed_word_72 = !condition_3 ? packed_word_75 : packed_word_72;
    packed_word_77 = !condition_3 ? packed_word_77 : packed_word_74;
    packed_word_62 = !condition_0 ? packed_word_44 : packed_word_51;
    packed_word_66 = !condition_0 ? packed_word_51 : packed_word_44;
    packed_word_68 = !condition_3 ? packed_word_71 : packed_word_68;
    packed_word_51 = !condition_0 ? packed_word_61 : packed_word_70;
    packed_word_59 = !condition_0 ? packed_word_63 : packed_word_60;
    packed_word_76 = !condition_3 ? packed_word_79 : packed_word_76;
    packed_word_81 = !condition_3 ? packed_word_81 : packed_word_78;
    packed_word_61 = !condition_0 ? packed_word_70 : packed_word_61;
    packed_word_85 = unpack_fp8_bits(packed_word_24);
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_63 = !condition_0 ? packed_word_60 : packed_word_63;
    packed_word_70 = unpack_fp8_bits(packed_word_28 >> 16);
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_93 = unpack_fp8_bits(packed_word_30);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_60 = !condition_0 ? packed_word_64 : packed_word_69;
    packed_word_57 = !condition_0 ? packed_word_72 : packed_word_77;
    packed_word_64 = !condition_0 ? packed_word_69 : packed_word_64;
    packed_word_77 = !condition_0 ? packed_word_77 : packed_word_72;
    packed_word_72 = !condition_0 ? packed_word_68 : packed_word_73;
    packed_word_74 = !condition_0 ? packed_word_73 : packed_word_68;
    packed_word_84 = unpack_fp8_bits(packed_word_25);
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_69 = unpack_fp8_bits(packed_word_25 >> 16);
    packed_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_79 = unpack_fp8_bits(packed_word_27);
    packed_word_56 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_56),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_83 = unpack_fp8_bits(packed_word_27 >> 16);
    packed_word_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_67 = !condition_0 ? packed_word_76 : packed_word_81;
    packed_word_73 = !condition_0 ? packed_word_81 : packed_word_76;
    packed_word_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_81 = unpack_fp8_bits(packed_word_29);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<2, false, false>(packed_word_9));
    packed_word_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_50));
    packed_word_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_70),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_39));
    packed_word_93 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_38));
    packed_word_68 = unpack_fp8_bits(packed_word_24 >> 16);
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_78 = unpack_fp8_bits(packed_word_26);
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                        select_half_operand<2, false, false>(packed_word_3));
    packed_word_82 = unpack_fp8_bits(packed_word_26 >> 16);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1000ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    packed_word_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_48));
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_71 = unpack_fp8_bits(packed_word_29 >> 16);
    packed_word_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_80 = unpack_fp8_bits(packed_word_28);
    packed_word_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_94 = unpack_fp8_bits(packed_word_30 >> 16);
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_92 = unpack_fp8_bits(packed_word_31);
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_95 = unpack_fp8_bits(packed_word_31 >> 16);
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_101 = unpack_fp8_bits(packed_word_20);
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_57),
                                        select_half_operand<2, false, false>(packed_word_2));
    packed_word_98 = unpack_fp8_bits(packed_word_23);
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_86 = unpack_fp8_bits(packed_word_23 >> 16);
    packed_word_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_45));
    packed_word_102 = unpack_fp8_bits(packed_word_21);
    packed_word_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_24));
    packed_word_97 = unpack_fp8_bits(packed_word_22);
    packed_word_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_53));
    packed_word_76 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_14, packed_word_15) + 0x2010ull, memory_ranges, error_counts);
    packed_word_83 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_8));
    packed_word_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_26));
    packed_word_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_60));
    packed_word_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_68),
                                   select_half_operand<2, false, false>(0u));
    packed_word_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_25));
    packed_word_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_69),
                                   select_half_operand<2, false, false>(0u));
    packed_word_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_9));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_82),
                                  select_half_operand<2, false, false>(0u));
    packed_word_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_83),
                                  select_half_operand<2, false, false>(0u));
    packed_word_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_85),
                                   select_half_operand<2, false, false>(packed_word_28));
    packed_word_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_84),
                                   select_half_operand<2, false, false>(packed_word_29));
    packed_word_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_27));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_70),
                                  select_half_operand<2, false, false>(0u));
    packed_word_101 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_56));
    packed_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_3));
    // Accumulate packed FP8 activation/weight fragments in warp-wide Tensor Core operations.
    multiply_accumulate_half(packed_word_56, packed_word_57, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_40, packed_word_41, 0u,
                             0u);
    packed_word_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_3 = unpack_fp8_bits(packed_word_22 >> 16);
    packed_word_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_55));
    packed_word_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_24));
    packed_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_80),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_25));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_71),
                                  select_half_operand<2, false, false>(0u));
    packed_word_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_26));
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_98 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_27));
    packed_word_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(0u));
    packed_word_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(0u));
    packed_word_52 = unpack_fp8_bits(packed_word_4);
    packed_word_23 = unpack_fp8_bits(packed_word_4 >> 16);
    packed_word_90 = unpack_fp8_bits(packed_word_6);
    packed_word_22 = unpack_fp8_bits(packed_word_5);
    packed_word_4 = unpack_fp8_bits(packed_word_7);
    packed_word_102 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_75));
    packed_word_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_81),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_97 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_62));
    packed_word_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                   select_half_operand<2, false, false>(packed_word_26));
    packed_word_104 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_104));
    packed_word_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(packed_word_27));
    packed_word_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_45));
    packed_word_72 = unpack_fp8_bits(packed_word_20 >> 16);
    packed_word_103 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_44));
    packed_word_87 = unpack_fp8_bits(packed_word_6 >> 16);
    packed_word_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_46));
    packed_word_7 = unpack_fp8_bits(packed_word_7 >> 16);
    packed_word_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_4),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_47));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1200ull, packed_word_44, packed_word_45,
        packed_word_46, packed_word_47, memory_ranges, error_counts);
    packed_word_73 = unpack_fp8_bits(packed_word_21 >> 16);
    packed_word_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_59));
    packed_word_2 = unpack_fp8_bits(packed_word_5 >> 16);
    packed_word_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_67));
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_40, packed_word_41, 0u,
                             0u);
    packed_word_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_54));
    packed_word_88 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_7),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_60));
    packed_word_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_50));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                  select_half_operand<2, false, false>(0u));
    packed_word_99 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_48));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_88),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                  select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                  select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                  select_half_operand<2, false, false>(packed_word_49));
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_57), 0x44004400u);
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                  select_half_operand<2, false, false>(packed_word_5));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                  select_half_operand<2, false, false>(packed_word_2));
    packed_word_2 = 0x2b28u;
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x44004400u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0xc400c400u);
    packed_word_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0xc400c400u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_54),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x44004400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_58), 0x44004400u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_59), 0x44004400u);
    packed_word_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0xc400c400u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0xc400c400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_105), 0xc400c400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_54),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_107),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_108), 0xc400c400u);
    packed_word_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_107),
        select_half_operand<2, false, false>(packed_word_52), 0x3b283b28u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x400ull, packed_word_52, packed_word_53,
        packed_word_54, packed_word_55, memory_ranges, error_counts);
    packed_word_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_72),
                                   select_half_operand<2, false, false>(0u));
    packed_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_73),
                                   select_half_operand<2, false, false>(0u));
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(0u));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(0u));
    packed_word_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                   select_half_operand<2, false, false>(packed_word_20));
    packed_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_102),
                                   select_half_operand<2, false, false>(packed_word_21));
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(packed_word_22));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_23));
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_40, packed_word_41, 0u,
                             0u);
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_56), 0x44004400u);
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_48),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<2, false, false>(packed_word_49), 0x3b283b28u);
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_57),
                                          select_half_operand<2, false, false>(packed_word_109));
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_56),
                                          select_half_operand<2, false, false>(packed_word_106));
    packed_word_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0x44004400u);
    packed_word_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0x44004400u);
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                          select_half_operand<2, false, false>(packed_word_105));
    packed_word_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_56), 0xc400c400u);
    packed_word_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_57), 0xc400c400u);
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                          select_half_operand<2, false, false>(packed_word_108));
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_56),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_58),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x44004400u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_58),
        select_half_operand<2, false, false>(packed_word_59), 0x3b283b28u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_107), 0xc400c400u);
    packed_word_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_56),
        select_half_operand<2, false, false>(packed_word_57), 0x3b283b28u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x600ull, packed_word_56, packed_word_57,
        packed_word_58, packed_word_59, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_42, packed_word_43, 0u, 0u);
    multiply_accumulate_half(packed_word_64, packed_word_65, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_40, packed_word_41, 0u, 0u);
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_32, packed_word_33, 0u,
                             0u);
    packed_word_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_60), 0x3b283b28u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x44004400u);
    packed_word_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_34, packed_word_35, 0u,
                             0u);
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_118), 0xc400c400u);
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0x44004400u);
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_120), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0x44004400u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_32, packed_word_33, 0u,
                             0u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0xc400c400u);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                         select_half_operand<2, false, false>(packed_word_115));
    packed_word_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x44004400u);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_34, packed_word_35, 0u,
                             0u);
    packed_word_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_121),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                          select_half_operand<2, false, false>(packed_word_107));
    packed_word_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x44004400u);
    packed_word_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_66 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_120));
    packed_word_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0xc400c400u);
    packed_word_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_121),
        select_half_operand<2, false, false>(packed_word_122), 0x3b283b28u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_115), 0xc400c400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_60), 0x3b283b28u);
    packed_word_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x44004400u);
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_32, packed_word_33, 0u,
                             0u);
    packed_word_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_62),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_64 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                         select_half_operand<2, false, false>(packed_word_117));
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_62),
        select_half_operand<2, false, false>(packed_word_63), 0x3b283b28u);
    packed_word_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x44004400u);
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                         select_half_operand<2, false, false>(packed_word_118));
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_34, packed_word_35, 0u,
                             0u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_118), 0x3b283b28u);
    packed_word_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x44004400u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_115), 0xc400c400u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_62),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_122 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x44004400u);
    packed_word_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_120), 0xc400c400u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_62),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_119 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_122), 0xc400c400u);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_32, packed_word_33, 0u, 0u);
    packed_word_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_121),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_119), 0xc400c400u);
    packed_word_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_62), 0x3b283b28u);
    packed_word_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_121),
        select_half_operand<2, false, false>(packed_word_122), 0x3b283b28u);
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_62));
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x44004400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                          select_half_operand<2, false, false>(packed_word_122));
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x44004400u);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_34, packed_word_35, 0u, 0u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0xc400c400u);
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_63));
    packed_word_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0xc400c400u);
    packed_word_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_48),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_41), 0x3b283b28u);
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(0u));
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<2, false, false>(packed_word_62), 0x3b283b28u);
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(0u));
    packed_word_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_123),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_40), 0x3b283b28u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(packed_word_41));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_123),
        select_half_operand<2, false, false>(packed_word_124), 0x3b283b28u);
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_63),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_124 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                          select_half_operand<2, false, false>(packed_word_124));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(0u));
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_63),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(packed_word_49));
    packed_word_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x44004400u);
    packed_word_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_63),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x44004400u);
    packed_word_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                    select_half_operand<2, false, false>(0u));
    packed_word_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_106), 0xc400c400u);
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_63),
        select_half_operand<2, false, false>(packed_word_85), 0x3b283b28u);
    packed_word_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0xc400c400u);
    packed_word_49 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(packed_word_110));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_122),
                                   select_half_operand<2, false, false>(0u));
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(0u));
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(packed_word_63));
    packed_word_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x44004400u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_36, packed_word_37,
                             packed_word_48, packed_word_49);
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                        select_half_operand<2, false, false>(packed_word_51));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                        select_half_operand<2, false, false>(packed_word_62));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(packed_word_112));
    packed_word_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0xc400c400u);
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(0u));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_63));
    packed_word_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_105),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_109),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_105),
        select_half_operand<2, false, false>(packed_word_8), 0x3b283b28u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_113),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_109),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_113),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                          select_half_operand<2, false, false>(packed_word_106));
    packed_word_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_67),
                                   select_half_operand<2, false, false>(0u));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_66),
                                   select_half_operand<2, false, false>(0u));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                   select_half_operand<2, false, false>(0u));
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                   select_half_operand<2, false, false>(0u));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_64),
                                   select_half_operand<2, false, false>(packed_word_33));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(packed_word_51));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(packed_word_32));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(packed_word_35));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_75),
                                   select_half_operand<2, false, false>(0u));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1400ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    packed_word_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                    select_half_operand<2, false, false>(0u));
    packed_word_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_124),
                                    select_half_operand<2, false, false>(0u));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(0u));
    packed_word_115 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                    select_half_operand<2, false, false>(0u));
    packed_word_80 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_74),
                                   select_half_operand<2, false, false>(packed_word_61));
    packed_word_81 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                   select_half_operand<2, false, false>(packed_word_108));
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                   select_half_operand<2, false, false>(packed_word_109));
    packed_word_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                         select_half_operand<2, false, false>(packed_word_100));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                   select_half_operand<2, false, false>(packed_word_60));
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                        select_half_operand<2, false, false>(packed_word_100));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(packed_word_115));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                        select_half_operand<2, false, false>(packed_word_100));
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_83 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<2, false, false>(packed_word_89));
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_36, packed_word_37,
                             packed_word_80, packed_word_81);
    packed_word_70 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_70),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                         select_half_operand<2, false, false>(packed_word_76));
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_36, packed_word_37,
                             packed_word_74, packed_word_75);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_36, packed_word_37,
                             packed_word_8, packed_word_9);
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                         select_half_operand<2, false, false>(packed_word_76));
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_38, packed_word_39,
                             packed_word_68, packed_word_69);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_44, packed_word_45,
                             packed_word_78, packed_word_79);
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_46, packed_word_47,
                             packed_word_82, packed_word_83);
    packed_word_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_41 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                         select_half_operand<2, false, false>(packed_word_91));
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_38, packed_word_39,
                             packed_word_70, packed_word_71);
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_38, packed_word_39,
                             packed_word_72, packed_word_73);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_38, packed_word_39,
                             packed_word_36, packed_word_37);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1600ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_44, packed_word_45,
                             packed_word_40, packed_word_41);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x800ull, packed_word_40, packed_word_41,
        packed_word_42, packed_word_43, memory_ranges, error_counts);
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                          select_half_operand<2, false, false>(packed_word_91));
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                          select_half_operand<2, false, false>(packed_word_91));
    packed_word_90 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                         select_half_operand<2, false, false>(packed_word_91));
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_52, packed_word_53, 0u,
                             0u);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_44, packed_word_45,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_94 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_95 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                          select_half_operand<2, false, false>(packed_word_89));
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                          select_half_operand<2, false, false>(packed_word_89));
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                          select_half_operand<2, false, false>(packed_word_89));
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                          select_half_operand<2, false, false>(packed_word_89));
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_52, packed_word_53, 0u,
                             0u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x44004400u);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_46, packed_word_47,
                             packed_word_94, packed_word_95);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0xa00ull, packed_word_48, packed_word_49,
        packed_word_50, packed_word_51, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_46, packed_word_47,
                             packed_word_106, packed_word_107);
    packed_word_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_90), 0x44004400u);
    packed_word_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x44004400u);
    packed_word_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0xc400c400u);
    multiply_accumulate_half(packed_word_64, packed_word_65, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_52, packed_word_53, 0u,
                             0u);
    packed_word_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0xc400c400u);
    packed_word_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_87),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_66), 0x3b283b28u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_91), 0x44004400u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_44, packed_word_45,
                             packed_word_100, packed_word_101);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_87),
        select_half_operand<2, false, false>(packed_word_109), 0x3b283b28u);
    packed_word_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x44004400u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_46, packed_word_47,
                             packed_word_102, packed_word_103);
    packed_word_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_89), 0x44004400u);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_87), 0xc400c400u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_114), 0xc400c400u);
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_67),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_87), 0x3b283b28u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_52, packed_word_53, 0u, 0u);
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_67),
        select_half_operand<2, false, false>(packed_word_86), 0x3b283b28u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_3), 0x3b283b28u);
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                          select_half_operand<2, false, false>(packed_word_109));
    packed_word_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x44004400u);
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                          select_half_operand<2, false, false>(packed_word_86));
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_54, packed_word_55, 0u, 0u);
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                          select_half_operand<2, false, false>(packed_word_108));
    packed_word_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_91),
                                          select_half_operand<2, false, false>(packed_word_87));
    packed_word_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0x44004400u);
    packed_word_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_90), 0xc400c400u);
    packed_word_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_88), 0x44004400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0xc400c400u);
    packed_word_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0xc400c400u);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_112), 0xc400c400u);
    packed_word_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_89 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_97),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_96), 0x3b283b28u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_112), 0x3b283b28u);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_97),
        select_half_operand<2, false, false>(packed_word_98), 0x3b283b28u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_113),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_90 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                         select_half_operand<2, false, false>(packed_word_112));
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_113),
        select_half_operand<2, false, false>(packed_word_114), 0x3b283b28u);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_99), 0x3b283b28u);
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_88 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                         select_half_operand<2, false, false>(packed_word_114));
    packed_word_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_56, packed_word_57, 0u, 0u);
    packed_word_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_56), 0xc400c400u);
    packed_word_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_56), 0x3b283b28u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_57),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_112), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_57),
        select_half_operand<2, false, false>(packed_word_3), 0x3b283b28u);
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_46), 0x44004400u);
    packed_word_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_113),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_112), 0x3b283b28u);
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                        select_half_operand<2, false, false>(packed_word_3));
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0x44004400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_56), 0x3b283b28u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_113),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                          select_half_operand<2, false, false>(packed_word_99));
    packed_word_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x44004400u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_63),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_56, packed_word_57, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                          select_half_operand<2, false, false>(packed_word_98));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0xc400c400u);
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                          select_half_operand<2, false, false>(packed_word_119));
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_62),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_58, packed_word_59, 0u, 0u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0x44004400u);
    packed_word_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_63),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0x44004400u);
    packed_word_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x44004400u);
    packed_word_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_59), 0xc400c400u);
    packed_word_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0xc400c400u);
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_62),
        select_half_operand<2, false, false>(packed_word_54), 0x3b283b28u);
    packed_word_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x44004400u);
    packed_word_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_59),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0x44004400u);
    packed_word_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_61),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0xc400c400u);
    packed_word_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_59),
        select_half_operand<2, false, false>(packed_word_60), 0x3b283b28u);
    packed_word_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0xc400c400u);
    packed_word_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_61),
        select_half_operand<2, false, false>(packed_word_62), 0x3b283b28u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1800ull, packed_word_60, packed_word_61,
        packed_word_62, packed_word_63, memory_ranges, error_counts);
    packed_word_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_56), 0x44004400u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_120),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                         select_half_operand<2, false, false>(packed_word_59));
    packed_word_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_87), 0x44004400u);
    packed_word_58 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                         select_half_operand<2, false, false>(packed_word_58));
    packed_word_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_86), 0x44004400u);
    packed_word_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_120),
        select_half_operand<2, false, false>(packed_word_121), 0x3b283b28u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0xc400c400u);
    packed_word_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_57), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_65),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_46), 0xc400c400u);
    packed_word_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                          select_half_operand<2, false, false>(packed_word_121));
    packed_word_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0xc400c400u);
    packed_word_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_122),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_65),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    packed_word_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_46),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_122),
        select_half_operand<2, false, false>(packed_word_123), 0x3b283b28u);
    packed_word_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_64),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x44004400u);
    packed_word_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_46),
        select_half_operand<2, false, false>(packed_word_47), 0x3b283b28u);
    packed_word_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                    select_half_operand<2, false, false>(0u));
    packed_word_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_64),
        select_half_operand<2, false, false>(packed_word_65), 0x3b283b28u);
    packed_word_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x44004400u);
    packed_word_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_66),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_118 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                    select_half_operand<2, false, false>(0u));
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_119));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(packed_word_109));
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_66),
        select_half_operand<2, false, false>(packed_word_67), 0x3b283b28u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0xc400c400u);
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                   select_half_operand<2, false, false>(packed_word_118));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_58),
                                   select_half_operand<2, false, false>(0u));
    packed_word_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_118 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_59),
                                    select_half_operand<2, false, false>(0u));
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                         select_half_operand<2, false, false>(packed_word_53));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_54),
                                   select_half_operand<2, false, false>(packed_word_67));
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_58), 0x3b283b28u);
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_55),
                                   select_half_operand<2, false, false>(packed_word_118));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1a00ull, packed_word_52, packed_word_53,
        packed_word_54, packed_word_55, memory_ranges, error_counts);
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_56),
                                         select_half_operand<2, false, false>(packed_word_46));
    packed_word_88 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_88),
                                   select_half_operand<2, false, false>(0u));
    packed_word_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_108),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_123),
                                   select_half_operand<2, false, false>(0u));
    packed_word_119 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_57),
                                          select_half_operand<2, false, false>(packed_word_119));
    packed_word_88 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(packed_word_88));
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<2, false, false>(packed_word_59), 0x3b283b28u);
    packed_word_91 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_56));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0xe00ull, packed_word_56, packed_word_57,
        packed_word_58, packed_word_59, memory_ranges, error_counts);
    packed_word_89 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_89),
                                   select_half_operand<2, false, false>(0u));
    packed_word_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                    select_half_operand<2, false, false>(0u));
    packed_word_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x44004400u);
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                         select_half_operand<2, false, false>(packed_word_47));
    packed_word_89 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                   select_half_operand<2, false, false>(packed_word_89));
    packed_word_90 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_121));
    packed_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                   select_half_operand<2, false, false>(0u));
    packed_word_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x44004400u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0xc400c400u);
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_32, packed_word_33,
                             packed_word_84, packed_word_85);
    packed_word_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0xc400c400u);
    packed_word_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(packed_word_86));
    packed_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                   select_half_operand<2, false, false>(0u));
    packed_word_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_44),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_46), 0x3b283b28u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_44),
        select_half_operand<2, false, false>(packed_word_45), 0x3b283b28u);
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(0u));
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                    select_half_operand<2, false, false>(0u));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(packed_word_112));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(packed_word_114));
    multiply_accumulate_half(packed_word_112, packed_word_113, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_38, packed_word_39,
                             packed_word_82, packed_word_83);
    packed_word_117 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                    select_half_operand<2, false, false>(0u));
    packed_word_84 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(packed_word_117));
    packed_word_116 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_119),
                                    select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(packed_word_86));
    packed_word_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(packed_word_116));
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                          select_half_operand<2, false, false>(packed_word_118));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0xc00ull, packed_word_44, packed_word_45,
        packed_word_46, packed_word_47, memory_ranges, error_counts);
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                          select_half_operand<2, false, false>(packed_word_115));
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_88, packed_word_89,
                             packed_word_90, packed_word_91, packed_word_32, packed_word_33,
                             packed_word_80, packed_word_81);
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(0u));
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_88, packed_word_89,
                             packed_word_90, packed_word_91, packed_word_34, packed_word_35,
                             packed_word_70, packed_word_71);
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_96));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_88, packed_word_89,
                             packed_word_90, packed_word_91, packed_word_36, packed_word_37,
                             packed_word_92, packed_word_93);
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                   select_half_operand<2, false, false>(packed_word_98));
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_88, packed_word_89,
                             packed_word_90, packed_word_91, packed_word_38, packed_word_39,
                             packed_word_94, packed_word_95);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_40, packed_word_41, 0u,
                             0u);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_84, packed_word_85,
                             packed_word_86, packed_word_87, packed_word_32, packed_word_33,
                             packed_word_74, packed_word_75);
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_84, packed_word_85,
                             packed_word_86, packed_word_87, packed_word_34, packed_word_35,
                             packed_word_72, packed_word_73);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_84, packed_word_85,
                             packed_word_86, packed_word_87, packed_word_36, packed_word_37,
                             packed_word_100, packed_word_101);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_84, packed_word_85,
                             packed_word_86, packed_word_87, packed_word_38, packed_word_39,
                             packed_word_102, packed_word_103);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_40, packed_word_41, 0u,
                             0u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_88), 0x44004400u);
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_34, packed_word_35,
                             packed_word_68, packed_word_69);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_36, packed_word_37,
                             packed_word_78, packed_word_79);
    packed_word_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_82), 0x44004400u);
    packed_word_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0xc400c400u);
    multiply_accumulate_half(packed_word_64, packed_word_65, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_40, packed_word_41, 0u,
                             0u);
    packed_word_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_89), 0x44004400u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_38, packed_word_39,
                             packed_word_106, packed_word_107);
    packed_word_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0xc400c400u);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_32, packed_word_33,
                             packed_word_8, packed_word_9);
    packed_word_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_34, packed_word_35,
                             packed_word_76, packed_word_77);
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_36, packed_word_37,
                             packed_word_104, packed_word_105);
    packed_word_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_91),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_91),
        select_half_operand<2, false, false>(packed_word_96), 0x3b283b28u);
    packed_word_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_83), 0x44004400u);
    packed_word_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_86), 0x44004400u);
    packed_word_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0xc400c400u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_40, packed_word_41, 0u, 0u);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0xc400c400u);
    packed_word_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_96),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0x44004400u);
    packed_word_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_87), 0x44004400u);
    packed_word_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_67),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_85), 0x44004400u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_66), 0x3b283b28u);
    packed_word_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_104), 0xc400c400u);
    packed_word_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_96),
        select_half_operand<2, false, false>(packed_word_97), 0x3b283b28u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_42, packed_word_43, 0u, 0u);
    packed_word_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_98),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0xc400c400u);
    packed_word_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_67),
        select_half_operand<2, false, false>(packed_word_90), 0x3b283b28u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    packed_word_88 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0x44004400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_105),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_98),
        select_half_operand<2, false, false>(packed_word_96), 0x3b283b28u);
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_99),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                         select_half_operand<2, false, false>(packed_word_90));
    packed_word_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x44004400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_105),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_89 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_99),
        select_half_operand<2, false, false>(packed_word_104), 0x3b283b28u);
    packed_word_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_90), 0xc400c400u);
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<2, false, false>(packed_word_97));
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_110));
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_86), 0xc400c400u);
    packed_word_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                         select_half_operand<2, false, false>(packed_word_104));
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_90),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_50, packed_word_51, 0u,
                             0u);
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                          select_half_operand<2, false, false>(packed_word_105));
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_84), 0x3b283b28u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_90),
        select_half_operand<2, false, false>(packed_word_86), 0x3b283b28u);
    packed_word_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_85), 0x3b283b28u);
    packed_word_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x44004400u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0x44004400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_87),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x44004400u);
    packed_word_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x44004400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_85), 0xc400c400u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_87),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0xc400c400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_85),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_84), 0x3b283b28u);
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_50, packed_word_51, 0u,
                             0u);
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                        select_half_operand<2, false, false>(packed_word_90));
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x44004400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_85),
        select_half_operand<2, false, false>(packed_word_87), 0x3b283b28u);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_117));
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x44004400u);
    packed_word_86 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                          select_half_operand<2, false, false>(packed_word_104));
    packed_word_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    packed_word_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0xc400c400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_90 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_119));
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_41), 0x3b283b28u);
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x44004400u);
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_64),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_50, packed_word_51, 0u,
                             0u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x44004400u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_64),
        select_half_operand<2, false, false>(packed_word_65), 0x3b283b28u);
    packed_word_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0x44004400u);
    packed_word_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0x44004400u);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_48, packed_word_49, 0u, 0u);
    packed_word_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_42), 0x3b283b28u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_64),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_118), 0xc400c400u);
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_111));
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x44004400u);
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_64),
        select_half_operand<2, false, false>(packed_word_40), 0x3b283b28u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_40));
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_82), 0x44004400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_118), 0x3b283b28u);
    packed_word_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_83), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_85), 0x44004400u);
    packed_word_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0xc400c400u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_50, packed_word_51, 0u, 0u);
    packed_word_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0xc400c400u);
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x44004400u);
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_64), 0x3b283b28u);
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_67),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                         select_half_operand<2, false, false>(packed_word_65));
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                         select_half_operand<2, false, false>(packed_word_42));
    packed_word_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_66),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_117), 0xc400c400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_67),
        select_half_operand<2, false, false>(packed_word_111), 0x3b283b28u);
    packed_word_91 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(0u));
    packed_word_41 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                         select_half_operand<2, false, false>(packed_word_41));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(0u));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_43));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(0u));
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_66),
        select_half_operand<2, false, false>(packed_word_64), 0x3b283b28u);
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(0u));
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_88),
                                   select_half_operand<2, false, false>(packed_word_91));
    packed_word_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                    select_half_operand<2, false, false>(0u));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_99));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1c00ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_41),
                                   select_half_operand<2, false, false>(packed_word_66));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(packed_word_42));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(0u));
    packed_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                   select_half_operand<2, false, false>(0u));
    packed_word_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<2, false, false>(packed_word_40));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_83 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<2, false, false>(packed_word_111));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_37),
                                   select_half_operand<2, false, false>(packed_word_120));
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_60, packed_word_61,
                             packed_word_108, packed_word_109);
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(packed_word_39));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x1e00ull, packed_word_108, packed_word_109,
        packed_word_110, packed_word_111, memory_ranges, error_counts);
    packed_word_117 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                    select_half_operand<2, false, false>(0u));
    packed_word_119 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                          select_half_operand<2, false, false>(packed_word_119));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(0u));
    packed_word_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                   select_half_operand<2, false, false>(0u));
    packed_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_82),
                                   select_half_operand<2, false, false>(packed_word_117));
    packed_word_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x44004400u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_89),
                                   select_half_operand<2, false, false>(packed_word_40));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_41));
    packed_word_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x44004400u);
    packed_word_36 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                   select_half_operand<2, false, false>(packed_word_85));
    packed_word_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_82), 0xc400c400u);
    packed_word_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x44004400u);
    packed_word_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0xc400c400u);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_60, packed_word_61,
                             packed_word_80, packed_word_81);
    packed_word_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_82),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_119),
                                   select_half_operand<2, false, false>(0u));
    packed_word_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_85),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0xc400c400u);
    packed_word_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_121),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_82),
        select_half_operand<2, false, false>(packed_word_84), 0x3b283b28u);
    packed_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_83),
                                   select_half_operand<2, false, false>(packed_word_98));
    packed_word_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_80),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_85),
        select_half_operand<2, false, false>(packed_word_91), 0x3b283b28u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_121),
        select_half_operand<2, false, false>(packed_word_122), 0x3b283b28u);
    packed_word_84 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_84));
    packed_word_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_80),
        select_half_operand<2, false, false>(packed_word_81), 0x3b283b28u);
    packed_word_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_91));
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_60, packed_word_61,
                             packed_word_74, packed_word_75);
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_84),
                                   select_half_operand<2, false, false>(0u));
    packed_word_81 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                         select_half_operand<2, false, false>(packed_word_81));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(0u));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                   select_half_operand<2, false, false>(0u));
    packed_word_74 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(0u));
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(packed_word_51));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(packed_word_49));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_81),
                                   select_half_operand<2, false, false>(packed_word_74));
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_52, packed_word_53,
                             packed_word_78, packed_word_79);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_54, packed_word_55,
                             packed_word_112, packed_word_113);
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_52, packed_word_53,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_54, packed_word_55,
                             packed_word_94, packed_word_95);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_52, packed_word_53,
                             packed_word_100, packed_word_101);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_54, packed_word_55,
                             packed_word_102, packed_word_103);
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_52, packed_word_53,
                             packed_word_114, packed_word_115);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_54, packed_word_55,
                             packed_word_106, packed_word_107);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x22d0ull, packed_word_52, packed_word_53,
        packed_word_54, packed_word_55, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_60, packed_word_61,
                             packed_word_8, packed_word_9);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_62, packed_word_63,
                             packed_word_76, packed_word_77);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x20d0ull, packed_word_48, packed_word_49,
        packed_word_50, packed_word_51, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x26d0ull, packed_word_76, packed_word_77,
        packed_word_78, packed_word_79, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_62, packed_word_63,
                             packed_word_68, packed_word_69);
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_56, packed_word_57, 0u,
                             0u);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_67), 0x44004400u);
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_62, packed_word_63,
                             packed_word_72, packed_word_73);
    packed_word_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0xc400c400u);
    packed_word_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_106), 0x44004400u);
    packed_word_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_107), 0x44004400u);
    packed_word_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_72), 0xc400c400u);
    packed_word_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_65),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0xc400c400u);
    packed_word_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_73),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_65),
        select_half_operand<2, false, false>(packed_word_72), 0x3b283b28u);
    packed_word_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_73),
        select_half_operand<2, false, false>(packed_word_74), 0x3b283b28u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_75),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                          select_half_operand<2, false, false>(packed_word_74));
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_75),
        select_half_operand<2, false, false>(packed_word_100), 0x3b283b28u);
    packed_word_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                         select_half_operand<2, false, false>(packed_word_72));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x24d0ull, packed_word_72, packed_word_73,
        packed_word_74, packed_word_75, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_62, packed_word_63,
                             packed_word_70, packed_word_71);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_56, packed_word_57, 0u,
                             0u);
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_58, packed_word_59, 0u, 0u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_44, packed_word_45, 0u,
                             0u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_56, packed_word_57, 0u,
                             0u);
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_46, packed_word_47, 0u,
                             0u);
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_44, packed_word_45, 0u,
                             0u);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_24, packed_word_25,
                             packed_word_26, packed_word_27, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    packed_word_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0xc400c400u);
    packed_word_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0xc400c400u);
    packed_word_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_25),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_24), 0x3b283b28u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_27),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_25),
        select_half_operand<2, false, false>(packed_word_26), 0x3b283b28u);
    packed_word_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_66), 0x44004400u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_27),
        select_half_operand<2, false, false>(packed_word_64), 0x3b283b28u);
    multiply_accumulate_half(packed_word_56, packed_word_57, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_56, packed_word_57, 0u, 0u);
    packed_word_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0xc400c400u);
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_64 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_59), 0x44004400u);
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_25), 0x3b283b28u);
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_26));
    packed_word_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x44004400u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_27),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0xc400c400u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_27),
        select_half_operand<2, false, false>(packed_word_3), 0x3b283b28u);
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_66 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                         select_half_operand<2, false, false>(packed_word_3));
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_26),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_63), 0x3b283b28u);
    packed_word_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_26),
        select_half_operand<2, false, false>(packed_word_3), 0x3b283b28u);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x44004400u);
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_24));
    packed_word_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0xc400c400u);
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                          select_half_operand<2, false, false>(packed_word_25));
    packed_word_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_58), 0x44004400u);
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0xc400c400u);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_27),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_56), 0x44004400u);
    packed_word_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0xc400c400u);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_27),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_57), 0x44004400u);
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0xc400c400u);
    packed_word_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0xc400c400u);
    packed_word_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_3),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_113),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<2, false, false>(packed_word_20), 0x3b283b28u);
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                          select_half_operand<2, false, false>(packed_word_100));
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0xc400c400u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_21),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_26),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_113),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                          select_half_operand<2, false, false>(packed_word_20));
    packed_word_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x44004400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_21),
        select_half_operand<2, false, false>(packed_word_3), 0x3b283b28u);
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                          select_half_operand<2, false, false>(packed_word_61));
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_28), 0x44004400u);
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_26),
        select_half_operand<2, false, false>(packed_word_27), 0x3b283b28u);
    packed_word_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x44004400u);
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_44, packed_word_45, 0u, 0u);
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                        select_half_operand<2, false, false>(packed_word_119));
    packed_word_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0xc400c400u);
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_60));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_21), 0xc400c400u);
    packed_word_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_20),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_46, packed_word_47, 0u, 0u);
    packed_word_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_20),
        select_half_operand<2, false, false>(packed_word_21), 0x3b283b28u);
    packed_word_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_29), 0x44004400u);
    packed_word_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_30), 0x44004400u);
    packed_word_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_6), 0xc400c400u);
    packed_word_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_7), 0xc400c400u);
    packed_word_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_41),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_6),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_4), 0x3b283b28u);
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_31), 0x44004400u);
    packed_word_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_6),
        select_half_operand<2, false, false>(packed_word_7), 0x3b283b28u);
    packed_word_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_41),
        select_half_operand<2, false, false>(packed_word_5), 0x3b283b28u);
    packed_word_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x44004400u);
    packed_word_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_20),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0xc400c400u);
    packed_word_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_20),
        select_half_operand<2, false, false>(packed_word_6), 0x3b283b28u);
    packed_word_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x44004400u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_7 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                        select_half_operand<2, false, false>(packed_word_7));
    packed_word_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_22), 0x44004400u);
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                        select_half_operand<2, false, false>(packed_word_5));
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x44004400u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_41), 0x3b283b28u);
    packed_word_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0x44004400u);
    packed_word_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0xc400c400u);
    packed_word_6 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                        select_half_operand<2, false, false>(packed_word_6));
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0x44004400u);
    packed_word_21 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_21));
    packed_word_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_29), 0xc400c400u);
    packed_word_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_23), 0x44004400u);
    packed_word_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_28), 0xc400c400u);
    packed_word_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_20),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_4 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                        select_half_operand<2, false, false>(packed_word_4));
    packed_word_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0xc400c400u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_44),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_30),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_20),
        select_half_operand<2, false, false>(packed_word_28), 0x3b283b28u);
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_41));
    packed_word_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_29),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                  select_half_operand<2, false, false>(0u));
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_40),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_44),
        select_half_operand<2, false, false>(packed_word_45), 0x3b283b28u);
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_30),
        select_half_operand<2, false, false>(packed_word_36), 0x3b283b28u);
    packed_word_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_37),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_29),
        select_half_operand<2, false, false>(packed_word_20), 0x3b283b28u);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_43));
    packed_word_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<2, false, false>(packed_word_41), 0x3b283b28u);
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_4),
                                   select_half_operand<2, false, false>(packed_word_7));
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_45));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                  select_half_operand<2, false, false>(0u));
    packed_word_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_37),
        select_half_operand<2, false, false>(packed_word_30), 0x3b283b28u);
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                         select_half_operand<2, false, false>(packed_word_36));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                  select_half_operand<2, false, false>(0u));
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_100),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(0u));
    packed_word_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                         select_half_operand<2, false, false>(packed_word_28));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_67),
                                   select_half_operand<2, false, false>(packed_word_4));
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                         select_half_operand<2, false, false>(packed_word_30));
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(0u));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                        select_half_operand<2, false, false>(packed_word_20));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                  select_half_operand<2, false, false>(0u));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_56),
                                         select_half_operand<2, false, false>(packed_word_63));
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_64),
                                   select_half_operand<2, false, false>(0u));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x28d0ull, packed_word_44, packed_word_45,
        packed_word_46, packed_word_47, memory_ranges, error_counts);
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(0u));
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<2, false, false>(packed_word_106), 0x3b283b28u);
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                   select_half_operand<2, false, false>(packed_word_7));
    packed_word_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(packed_word_112));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                  select_half_operand<2, false, false>(0u));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                   select_half_operand<2, false, false>(packed_word_4));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_28),
                                  select_half_operand<2, false, false>(packed_word_23));
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_57),
                                          select_half_operand<2, false, false>(packed_word_106));
    packed_word_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0xc400c400u);
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_59),
                                  select_half_operand<2, false, false>(packed_word_6));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x2ad0ull, packed_word_56, packed_word_57,
        packed_word_58, packed_word_59, memory_ranges, error_counts);
    packed_word_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_28), 0xc400c400u);
    packed_word_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_25),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                   select_half_operand<2, false, false>(0u));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                  select_half_operand<2, false, false>(packed_word_30));
    packed_word_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_25),
        select_half_operand<2, false, false>(packed_word_28), 0x3b283b28u);
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_31),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_30));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(packed_word_37));
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_31),
        select_half_operand<2, false, false>(packed_word_36), 0x3b283b28u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_21),
                                   select_half_operand<2, false, false>(packed_word_40));
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_29));
    packed_word_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                    select_half_operand<2, false, false>(0u));
    packed_word_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                   select_half_operand<2, false, false>(0u));
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_36));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_66),
                                   select_half_operand<2, false, false>(packed_word_101));
    packed_word_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0x44004400u);
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                   select_half_operand<2, false, false>(packed_word_30));
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_34, packed_word_35,
                             packed_word_68, packed_word_69);
    packed_word_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0xc400c400u);
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(0u));
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_24),
        select_half_operand<0, false, true>(packed_word_2), 0x37283728u);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                  select_half_operand<2, false, false>(packed_word_22));
    multiply_accumulate_half(packed_word_20, packed_word_21, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_32, packed_word_33,
                             packed_word_96, packed_word_97);
    packed_word_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_24),
        select_half_operand<2, false, false>(packed_word_2), 0x3b283b28u);
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                   select_half_operand<2, false, false>(0u));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                        select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_108, packed_word_109,
                             packed_word_86, packed_word_87);
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_62));
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_110, packed_word_111,
                             packed_word_90, packed_word_91);
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(packed_word_3));
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_110, packed_word_111,
                             packed_word_94, packed_word_95);
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_61));
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_34, packed_word_35,
                             packed_word_70, packed_word_71);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_108, packed_word_109,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_32, packed_word_33,
                             packed_word_88, packed_word_89);
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_28),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                  select_half_operand<2, false, false>(0u));
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_20),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_21),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_32, packed_word_33,
                             packed_word_82, packed_word_83);
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_32, packed_word_33,
                             packed_word_8, packed_word_9);
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_34, packed_word_35,
                             packed_word_80, packed_word_81);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_110, packed_word_111,
                             packed_word_102, packed_word_103);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_108, packed_word_109,
                             packed_word_104, packed_word_105);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                  select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                  select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_34, packed_word_35,
                             packed_word_84, packed_word_85);
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_37),
                                   select_half_operand<2, false, false>(packed_word_4));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_40),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_22),
                                  select_half_operand<2, false, false>(packed_word_3));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_23),
                                  select_half_operand<2, false, false>(packed_word_2));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_41),
                                  select_half_operand<2, false, false>(packed_word_8));
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_110, packed_word_111,
                             packed_word_98, packed_word_99);
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_108, packed_word_109,
                             packed_word_114, packed_word_115);
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_32),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_33),
                                  select_half_operand<2, false, false>(0u));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_138, packed_word_139, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_52, packed_word_53, 0u, 0u);
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_136, packed_word_137, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_54, packed_word_55, 0u, 0u);
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_24),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_25),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                   select_half_operand<2, false, false>(packed_word_9));
    multiply_accumulate_half(packed_word_130, packed_word_131, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_48, packed_word_49, 0u, 0u);
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(packed_word_8));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_132, packed_word_133, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_50, packed_word_51, 0u, 0u);
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                  select_half_operand<2, false, false>(0u));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                  select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                  select_half_operand<2, false, false>(0u));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_26),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_27),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(packed_word_8));
    packed_word_86 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                         select_half_operand<2, false, false>(packed_word_136));
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_76, packed_word_77, 0u,
                             0u);
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                         select_half_operand<2, false, false>(packed_word_137));
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_76, packed_word_77, 0u, 0u);
    packed_word_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                      select_half_operand<2, false, false>(packed_word_132),
                                      select_half_operand<2, false, false>(packed_word_86));
    packed_word_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                      select_half_operand<2, false, false>(packed_word_133),
                                      select_half_operand<2, false, false>(packed_word_87));
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_76, packed_word_77, 0u,
                             0u);
    multiply_accumulate_half(packed_word_128, packed_word_129, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_76, packed_word_77, 0u,
                             0u);
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                         select_half_operand<2, false, false>(packed_word_138));
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                         select_half_operand<2, false, false>(packed_word_139));
    packed_word_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                      select_half_operand<2, false, false>(packed_word_130),
                                      select_half_operand<2, false, false>(packed_word_76));
    packed_word_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                      select_half_operand<2, false, false>(packed_word_131),
                                      select_half_operand<2, false, false>(packed_word_77));
    packed_word_86 = add_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                    select_half_operand<2, false, false>(packed_word_76));
    packed_word_89 = add_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                    select_half_operand<2, false, false>(packed_word_77));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_86), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_87 = loaded_word;
    }
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_52, packed_word_53, 0u,
                             0u);
    multiply_accumulate_half(packed_word_118, packed_word_119, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_96 = add_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                    select_half_operand<2, false, false>(packed_word_87));
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_48, packed_word_49, 0u,
                             0u);
    multiply_accumulate_half(packed_word_150, packed_word_151, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_50, packed_word_51, 0u,
                             0u);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_72, packed_word_73, 0u,
                             0u);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_72, packed_word_73, 0u, 0u);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_72, packed_word_73, 0u,
                             0u);
    multiply_accumulate_half(packed_word_126, packed_word_127, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_72, packed_word_73, 0u,
                             0u);
    packed_word_72 = parameters[6];
    packed_word_73 = parameters[7];
    packed_word_157 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_72, packed_word_73) + 0x4cd0ull, memory_ranges, error_counts);
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                        select_half_operand<2, false, false>(packed_word_98));
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_118),
                                        select_half_operand<2, false, false>(packed_word_118));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                      select_half_operand<2, false, false>(packed_word_102),
                                      select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                      select_half_operand<2, false, false>(packed_word_150),
                                      select_half_operand<2, false, false>(packed_word_3));
    packed_word_88 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                    select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_119),
                                        select_half_operand<2, false, false>(packed_word_119));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                        select_half_operand<2, false, false>(packed_word_99));
    packed_word_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_151),
                                      select_half_operand<2, false, false>(packed_word_151),
                                      select_half_operand<2, false, false>(packed_word_3));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_103),
                                      select_half_operand<2, false, false>(packed_word_2));
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_2));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_88), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_3 = loaded_word;
    }
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_78, packed_word_79, 0u,
                             0u);
    multiply_accumulate_half(packed_word_158, packed_word_159, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_78, packed_word_79, 0u, 0u);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_78, packed_word_79, 0u,
                             0u);
    multiply_accumulate_half(packed_word_140, packed_word_141, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_78, packed_word_79, 0u,
                             0u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_79 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_89), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_78 = loaded_word;
    }
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                   select_half_operand<2, false, false>(packed_word_3));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_113 = loaded_word;
    }
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_74, packed_word_75, 0u,
                             0u);
    packed_word_79 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                    select_half_operand<2, false, false>(packed_word_79));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_2 = loaded_word;
    }
    packed_word_112 = add_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                     select_half_operand<2, false, false>(packed_word_78));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_79), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_78 = loaded_word;
    }
    packed_word_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                         select_half_operand<2, false, false>(packed_word_80));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_74, packed_word_75, 0u, 0u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_112), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_115 = loaded_word;
    }
    packed_word_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                      select_half_operand<2, false, false>(packed_word_76),
                                      select_half_operand<2, false, false>(packed_word_73));
    packed_word_97 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                    select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_113));
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<2, false, false>(packed_word_82));
    packed_word_72 = permute_signed(packed_word_97, packed_word_97, 0x5432u);
    packed_word_123 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                      select_half_operand<2, false, false>(packed_word_100),
                                      select_half_operand<2, false, false>(packed_word_96));
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_74, packed_word_75, 0u,
                             0u);
    packed_word_78 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                    select_half_operand<2, false, false>(packed_word_78));
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_73));
    packed_word_97 = add_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                    select_half_operand<2, false, false>(packed_word_72));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                         select_half_operand<2, false, false>(packed_word_81));
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<2, false, false>(packed_word_83));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_123), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_120 = loaded_word;
    }
    packed_word_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                      select_half_operand<2, false, false>(packed_word_77),
                                      select_half_operand<2, false, false>(packed_word_79));
    packed_word_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                      select_half_operand<2, false, false>(packed_word_101),
                                      select_half_operand<2, false, false>(packed_word_72));
    multiply_accumulate_half(packed_word_134, packed_word_135, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_74, packed_word_75, 0u,
                             0u);
    packed_word_113 = permute_signed(packed_word_78, packed_word_78, 0x5432u);
    packed_word_75 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                    select_half_operand<2, false, false>(packed_word_79));
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_158),
                                         select_half_operand<2, false, false>(packed_word_158));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_75), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_122 = loaded_word;
    }
    packed_word_96 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                    select_half_operand<2, false, false>(packed_word_113));
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                         select_half_operand<2, false, false>(packed_word_90));
    packed_word_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_91),
                                         select_half_operand<2, false, false>(packed_word_91));
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_159),
                                         select_half_operand<2, false, false>(packed_word_159));
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_112),
                                   select_half_operand<2, false, false>(packed_word_115));
    packed_word_112 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                      select_half_operand<2, false, false>(packed_word_92),
                                      select_half_operand<2, false, false>(packed_word_74));
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                         select_half_operand<2, false, false>(packed_word_8));
    packed_word_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                      select_half_operand<2, false, false>(packed_word_94),
                                      select_half_operand<2, false, false>(packed_word_72));
    packed_word_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                      select_half_operand<2, false, false>(packed_word_95),
                                      select_half_operand<2, false, false>(packed_word_73));
    packed_word_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                      select_half_operand<2, false, false>(packed_word_93),
                                      select_half_operand<2, false, false>(packed_word_78));
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_84));
    packed_word_113 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_88),
                                      select_half_operand<2, false, false>(packed_word_74));
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(packed_word_79));
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                    select_half_operand<2, false, false>(packed_word_73));
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                         select_half_operand<2, false, false>(packed_word_85));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_128),
                                          select_half_operand<2, false, false>(packed_word_128));
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                          select_half_operand<2, false, false>(packed_word_140));
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_120));
    packed_word_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                      select_half_operand<2, false, false>(packed_word_86),
                                      select_half_operand<2, false, false>(packed_word_72));
    packed_word_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                      select_half_operand<2, false, false>(packed_word_87),
                                      select_half_operand<2, false, false>(packed_word_78));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_123), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_144 = loaded_word;
    }
    packed_word_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                      select_half_operand<2, false, false>(packed_word_89),
                                      select_half_operand<2, false, false>(packed_word_79));
    packed_word_116 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_126),
                                      select_half_operand<2, false, false>(packed_word_126),
                                      select_half_operand<2, false, false>(packed_word_114));
    packed_word_117 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                      select_half_operand<2, false, false>(packed_word_134),
                                      select_half_operand<2, false, false>(packed_word_115));
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_113),
                                    select_half_operand<2, false, false>(packed_word_72));
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_129),
                                          select_half_operand<2, false, false>(packed_word_129));
    packed_word_78 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                    select_half_operand<2, false, false>(packed_word_78));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_124 = loaded_word;
    }
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                          select_half_operand<2, false, false>(packed_word_141));
    packed_word_79 = add_half_pairs(select_half_operand<2, false, false>(packed_word_117),
                                    select_half_operand<2, false, false>(packed_word_116));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_125 = loaded_word;
    }
    packed_word_153 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_127),
                                      select_half_operand<2, false, false>(packed_word_127),
                                      select_half_operand<2, false, false>(packed_word_114));
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x4100410u);
    packed_word_97 = add_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                    select_half_operand<2, false, false>(packed_word_122));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_74), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_117 = loaded_word;
    }
    packed_word_115 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                      select_half_operand<2, false, false>(packed_word_135),
                                      select_half_operand<2, false, false>(packed_word_115));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_122 = loaded_word;
    }
    packed_word_153 = add_half_pairs(select_half_operand<2, false, false>(packed_word_115),
                                     select_half_operand<2, false, false>(packed_word_153));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_78), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_121 = loaded_word;
    }
    packed_word_145 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x4100410u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_79), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_142 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_153), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_120 = loaded_word;
    }
    packed_word_75 = half_to_float_bits((packed_word_114 & 65535u) * 65537u);
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_144));
    packed_word_146 = half_to_float_bits((packed_word_114 >> 16) * 65537u);
    packed_word_143 = half_to_float_bits((packed_word_145 & 65535u) * 65537u);
    packed_word_144 = half_to_float_bits((packed_word_145 >> 16) * 65537u);
    packed_word_75 = rsqrt_bits(packed_word_75);
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_117));
    multiply_accumulate_half(packed_word_112, packed_word_113, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_124));
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                    select_half_operand<2, false, false>(packed_word_125));
    packed_word_96 = rsqrt_bits(packed_word_146);
    packed_word_97 = add_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                    select_half_operand<2, false, false>(packed_word_122));
    packed_word_143 = rsqrt_bits(packed_word_143);
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_144 = rsqrt_bits(packed_word_144);
    multiply_accumulate_half(packed_word_116, packed_word_117, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_78 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                    select_half_operand<2, false, false>(packed_word_121));
    packed_word_79 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                    select_half_operand<2, false, false>(packed_word_142));
    packed_word_153 = add_half_pairs(select_half_operand<2, false, false>(packed_word_153),
                                     select_half_operand<2, false, false>(packed_word_120));
    multiply_accumulate_half(packed_word_124, packed_word_125, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_68 = permute_signed(packed_word_123, packed_word_123, 0x5432u);
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_68));
    packed_word_68 = permute_signed(packed_word_97, packed_word_97, 0x5432u);
    packed_word_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_123), 0x4100410u);
    packed_word_97 = add_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                    select_half_operand<2, false, false>(packed_word_68));
    packed_word_142 = pack_half(packed_word_96, packed_word_75);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_78), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_71 = loaded_word;
    }
    packed_word_69 = half_to_float_bits((packed_word_123 & 65535u) * 65537u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_79), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_70 = loaded_word;
    }
    packed_word_145 = half_to_float_bits((packed_word_123 >> 16) * 65537u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_153), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_68 = loaded_word;
    }
    packed_word_120 = pack_half(packed_word_144, packed_word_143);
    packed_word_143 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x4100410u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_74), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_121 = loaded_word;
    }
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_118),
                                         select_half_operand<2, false, false>(packed_word_142));
    packed_word_69 = rsqrt_bits(packed_word_69);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_75 = loaded_word;
    }
    packed_word_118 = rsqrt_bits(packed_word_145);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_144 = loaded_word;
    }
    packed_word_78 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                    select_half_operand<2, false, false>(packed_word_71));
    packed_word_79 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                    select_half_operand<2, false, false>(packed_word_70));
    packed_word_153 = add_half_pairs(select_half_operand<2, false, false>(packed_word_153),
                                     select_half_operand<2, false, false>(packed_word_68));
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_121));
    packed_word_145 = pack_half(packed_word_118, packed_word_69);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x2cd0ull, packed_word_68, packed_word_69,
        packed_word_70, packed_word_71, memory_ranges, error_counts);
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                         select_half_operand<2, false, false>(packed_word_142));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                         select_half_operand<2, false, false>(packed_word_142));
    packed_word_142 = half_to_float_bits((packed_word_143 & 65535u) * 65537u);
    packed_word_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_99 = permute_signed(packed_word_74, packed_word_74, 0x5432u);
    packed_word_143 = half_to_float_bits((packed_word_143 >> 16) * 65537u);
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                    select_half_operand<2, false, false>(packed_word_75));
    packed_word_142 = rsqrt_bits(packed_word_142);
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_99));
    packed_word_75 = rsqrt_bits(packed_word_143);
    packed_word_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x4100410u);
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_144));
    packed_word_161 = half_to_float_bits((packed_word_74 & 65535u) * 65537u);
    packed_word_144 = half_to_float_bits((packed_word_74 >> 16) * 65537u);
    packed_word_74 = permute_signed(packed_word_73, packed_word_73, 0x5432u);
    packed_word_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_74));
    multiply_accumulate_half(packed_word_122, packed_word_123, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_44, packed_word_45, 0u, 0u);
    packed_word_151 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_151),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_142 = pack_half(packed_word_75, packed_word_142);
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_119),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_75 = permute_signed(packed_word_72, packed_word_72, 0x5432u);
    packed_word_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                          select_half_operand<2, false, false>(packed_word_145));
    packed_word_80 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                    select_half_operand<2, false, false>(packed_word_75));
    multiply_accumulate_half(packed_word_120, packed_word_121, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_46, packed_word_47, 0u, 0u);
    packed_word_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                          select_half_operand<2, false, false>(packed_word_145));
    packed_word_161 = rsqrt_bits(packed_word_161);
    multiply_accumulate_half(packed_word_118, packed_word_119, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_56, packed_word_57, 0u, 0u);
    packed_word_76 = rsqrt_bits(packed_word_144);
    multiply_accumulate_half(packed_word_148, packed_word_149, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_58, packed_word_59, 0u, 0u);
    packed_word_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_73), 0x4100410u);
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                         select_half_operand<2, false, false>(packed_word_145));
    packed_word_161 = pack_half(packed_word_76, packed_word_161);
    packed_word_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                          select_half_operand<2, false, false>(packed_word_142));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x2ed0ull, packed_word_72, packed_word_73,
        packed_word_74, packed_word_75, memory_ranges, error_counts);
    packed_word_5 = half_to_float_bits((packed_word_4 >> 16) * 65537u);
    packed_word_7 = half_to_float_bits((packed_word_4 & 65535u) * 65537u);
    packed_word_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_80), 0x4100410u);
    packed_word_80 = rsqrt_bits(packed_word_5);
    packed_word_5 = permute_signed(packed_word_78, packed_word_78, 0x5432u);
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                          select_half_operand<2, false, false>(packed_word_145));
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_155 = half_to_float_bits((packed_word_4 & 65535u) * 65537u);
    packed_word_82 = half_to_float_bits((packed_word_4 >> 16) * 65537u);
    packed_word_4 = permute_signed(packed_word_79, packed_word_79, 0x5432u);
    packed_word_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_6), 0x4100410u);
    packed_word_7 = rsqrt_bits(packed_word_7);
    packed_word_143 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                     select_half_operand<2, false, false>(packed_word_4));
    packed_word_5 = half_to_float_bits((packed_word_6 & 65535u) * 65537u);
    packed_word_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_6 = half_to_float_bits((packed_word_6 >> 16) * 65537u);
    packed_word_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_143), 0x4100410u);
    packed_word_155 = rsqrt_bits(packed_word_155);
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                          select_half_operand<2, false, false>(packed_word_142));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x30d0ull, packed_word_76, packed_word_77,
        packed_word_78, packed_word_79, memory_ranges, error_counts);
    packed_word_4 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                        select_half_operand<2, false, false>(packed_word_142));
    packed_word_82 = rsqrt_bits(packed_word_82);
    packed_word_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                          select_half_operand<2, false, false>(packed_word_161));
    packed_word_142 = half_to_float_bits((packed_word_81 & 65535u) * 65537u);
    packed_word_90 = half_to_float_bits((packed_word_81 >> 16) * 65537u);
    packed_word_5 = rsqrt_bits(packed_word_5);
    packed_word_6 = rsqrt_bits(packed_word_6);
    packed_word_7 = pack_half(packed_word_80, packed_word_7);
    packed_word_94 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                         select_half_operand<2, false, false>(packed_word_161));
    packed_word_142 = rsqrt_bits(packed_word_142);
    packed_word_92 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                         select_half_operand<2, false, false>(packed_word_161));
    packed_word_90 = rsqrt_bits(packed_word_90);
    packed_word_161 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_158),
                                          select_half_operand<2, false, false>(packed_word_161));
    packed_word_155 = pack_half(packed_word_82, packed_word_155);
    packed_word_158 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_91),
                                          select_half_operand<2, false, false>(packed_word_7));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x32d0ull, packed_word_80, packed_word_81,
        packed_word_82, packed_word_83, memory_ranges, error_counts);
    packed_word_91 = pack_half(packed_word_6, packed_word_5);
    packed_word_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                          select_half_operand<2, false, false>(packed_word_7));
    packed_word_162 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                          select_half_operand<2, false, false>(packed_word_7));
    packed_word_159 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_159),
                                          select_half_operand<2, false, false>(packed_word_7));
    packed_word_7 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                        select_half_operand<2, false, false>(packed_word_91));
    packed_word_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(0u));
    packed_word_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                          select_half_operand<2, false, false>(packed_word_155));
    packed_word_160 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                          select_half_operand<2, false, false>(packed_word_155));
    packed_word_163 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                          select_half_operand<2, false, false>(packed_word_155));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_4),
                                  select_half_operand<2, false, false>(0u));
    packed_word_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                          select_half_operand<2, false, false>(packed_word_155));
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                        select_half_operand<2, false, false>(packed_word_91));
    packed_word_9 = pack_half(packed_word_90, packed_word_142);
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                        select_half_operand<2, false, false>(packed_word_91));
    packed_word_142 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                    select_half_operand<2, false, false>(packed_word_85));
    packed_word_6 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                        select_half_operand<2, false, false>(packed_word_91));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x34d0ull, packed_word_84, packed_word_85,
        packed_word_86, packed_word_87, memory_ranges, error_counts);
    packed_word_145 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_145),
                                    select_half_operand<2, false, false>(packed_word_4));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x36d0ull, packed_word_88, packed_word_89,
        packed_word_90, packed_word_91, memory_ranges, error_counts);
    packed_word_4 = pack_half(0u, packed_word_157);
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(0u));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x38d0ull, packed_word_92, packed_word_93,
        packed_word_94, packed_word_95, memory_ranges, error_counts);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_146),
                                    select_half_operand<2, false, false>(packed_word_99));
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                    select_half_operand<2, false, false>(0u));
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                    select_half_operand<2, false, false>(0u));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(0u));
    packed_word_147 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_147),
                                    select_half_operand<2, false, false>(packed_word_100));
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_102),
                                   select_half_operand<2, false, false>(packed_word_99));
    packed_word_144 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_144),
                                    select_half_operand<2, false, false>(packed_word_101));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                   select_half_operand<2, false, false>(0u));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x3ad0ull, packed_word_100, packed_word_101,
        packed_word_102, packed_word_103, memory_ranges, error_counts);
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_151 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_151),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_156),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_151),
                                   select_half_operand<2, false, false>(0u));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_150),
                                   select_half_operand<2, false, false>(packed_word_99));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_156),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                  select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                  select_half_operand<2, false, false>(packed_word_6));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_8),
                                  select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                  select_half_operand<2, false, false>(packed_word_8));
    multiply_accumulate_half(packed_word_150, packed_word_151, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_146, packed_word_147,
                             packed_word_68, packed_word_69);
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_161),
                                   select_half_operand<2, false, false>(0u));
    packed_word_143 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_143),
                                    select_half_operand<2, false, false>(packed_word_68));
    packed_word_68 = permute_signed(packed_word_153, packed_word_153, 0x5432u);
    multiply_accumulate_half(packed_word_156, packed_word_157, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_144, packed_word_145,
                             packed_word_70, packed_word_71);
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_162),
                                   select_half_operand<2, false, false>(0u));
    packed_word_68 = add_half_pairs(select_half_operand<2, false, false>(packed_word_153),
                                    select_half_operand<2, false, false>(packed_word_68));
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_155),
                                   select_half_operand<2, false, false>(0u));
    packed_word_152 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_152),
                                    select_half_operand<2, false, false>(packed_word_69));
    packed_word_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_163),
                                    select_half_operand<2, false, false>(packed_word_70));
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_160),
                                   select_half_operand<2, false, false>(0u));
    packed_word_70 = permute_signed(packed_word_3, packed_word_3, 0x5432u);
    packed_word_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_68), 0x4100410u);
    packed_word_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_154),
                                    select_half_operand<2, false, false>(packed_word_69));
    packed_word_69 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                    select_half_operand<2, false, false>(packed_word_70));
    packed_word_70 = half_to_float_bits((packed_word_68 >> 16) * 65537u);
    packed_word_71 = half_to_float_bits((packed_word_68 & 65535u) * 65537u);
    packed_word_5 = rsqrt_bits(packed_word_71);
    packed_word_3 = permute_signed(packed_word_2, packed_word_2, 0x5432u);
    packed_word_70 = rsqrt_bits(packed_word_70);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_69), 0x4100410u);
    packed_word_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_2), 0x4100410u);
    packed_word_68 = half_to_float_bits((packed_word_69 & 65535u) * 65537u);
    packed_word_69 = half_to_float_bits((packed_word_69 >> 16) * 65537u);
    packed_word_71 = half_to_float_bits((packed_word_8 & 65535u) * 65537u);
    packed_word_5 = pack_half(packed_word_70, packed_word_5);
    packed_word_70 = half_to_float_bits((packed_word_8 >> 16) * 65537u);
    packed_word_68 = rsqrt_bits(packed_word_68);
    packed_word_69 = rsqrt_bits(packed_word_69);
    packed_word_71 = rsqrt_bits(packed_word_71);
    packed_word_70 = rsqrt_bits(packed_word_70);
    packed_word_126 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_126),
                                          select_half_operand<2, false, false>(packed_word_9));
    packed_word_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                          select_half_operand<2, false, false>(packed_word_9));
    packed_word_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_128),
                                          select_half_operand<2, false, false>(packed_word_9));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                        select_half_operand<2, false, false>(packed_word_9));
    packed_word_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_159 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_159),
                                    select_half_operand<2, false, false>(0u));
    packed_word_69 = pack_half(packed_word_69, packed_word_68);
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                  select_half_operand<2, false, false>(0u));
    packed_word_70 = pack_half(packed_word_70, packed_word_71);
    packed_word_135 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_135),
                                    select_half_operand<2, false, false>(0u));
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_141),
                                   select_half_operand<2, false, false>(0u));
    packed_word_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<2, false, false>(packed_word_69));
    packed_word_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_158),
                                    select_half_operand<2, false, false>(packed_word_159));
    packed_word_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                          select_half_operand<2, false, false>(packed_word_69));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                  select_half_operand<2, false, false>(packed_word_9));
    packed_word_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                          select_half_operand<2, false, false>(packed_word_70));
    packed_word_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                          select_half_operand<2, false, false>(packed_word_70));
    multiply_accumulate_half(packed_word_2, packed_word_3, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_154, packed_word_155,
                             packed_word_76, packed_word_77);
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_127),
                                         select_half_operand<2, false, false>(packed_word_5));
    packed_word_77 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_134),
                                   select_half_operand<2, false, false>(0u));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_129),
                                        select_half_operand<2, false, false>(packed_word_5));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_126),
                                  select_half_operand<2, false, false>(packed_word_77));
    packed_word_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(packed_word_135));
    packed_word_77 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_68));
    packed_word_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                          select_half_operand<2, false, false>(packed_word_69));
    packed_word_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<0, false, false>(packed_word_4));
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_142, packed_word_143,
                             packed_word_72, packed_word_73);
    packed_word_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                          select_half_operand<2, false, false>(packed_word_69));
    packed_word_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                          select_half_operand<2, false, false>(packed_word_70));
    packed_word_139 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                          select_half_operand<2, false, false>(packed_word_70));
    packed_word_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_136),
                                   select_half_operand<2, false, false>(0u));
    packed_word_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                          select_half_operand<0, false, false>(packed_word_4));
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_152, packed_word_153,
                             packed_word_74, packed_word_75);
    packed_word_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_132),
                                  select_half_operand<2, false, false>(0u));
    packed_word_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_139 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                          select_half_operand<0, false, false>(packed_word_4));
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_138),
                                   select_half_operand<2, false, false>(packed_word_69));
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_130),
                                   select_half_operand<2, false, false>(packed_word_5));
    multiply_accumulate_half(packed_word_158, packed_word_159, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_6, packed_word_7,
                             packed_word_78, packed_word_79);
    packed_word_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_133),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_8, packed_word_9,
                             packed_word_80, packed_word_81);
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_131),
                                   select_half_operand<2, false, false>(packed_word_78));
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_96, packed_word_97,
                             packed_word_98, packed_word_99, packed_word_76, packed_word_77,
                             packed_word_82, packed_word_83);
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_137),
                                   select_half_operand<2, false, false>(0u));
    packed_word_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_139),
                                   select_half_operand<2, false, false>(packed_word_96));
    packed_word_5 = 0x29c0u;
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_146, packed_word_147,
                             packed_word_84, packed_word_85);
    packed_word_4 = permute_signed(packed_word_4, packed_word_5, 0x5410u);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_144, packed_word_145,
                             packed_word_86, packed_word_87);
    packed_word_150 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_150),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_156 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_156),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_142, packed_word_143,
                             packed_word_88, packed_word_89);
    packed_word_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_72),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_74),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_152, packed_word_153,
                             packed_word_90, packed_word_91);
    packed_word_151 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_151),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_157 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_157),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_73),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_75),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_134 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_150), 0x3c203c20u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_156), 0x3c203c20u);
    packed_word_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_2),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_72), 0x3c203c20u);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_154, packed_word_155,
                             packed_word_92, packed_word_93);
    packed_word_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x3c203c20u);
    packed_word_158 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_158),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_151 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_151), 0x3c203c20u);
    packed_word_159 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_159),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_157 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_157), 0x3c203c20u);
    packed_word_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_73), 0x3c203c20u);
    packed_word_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x3c203c20u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_6, packed_word_7,
                             packed_word_94, packed_word_95);
    packed_word_134 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_134), 0x3e473e47u);
    packed_word_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x3e473e47u);
    packed_word_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_72), 0x3e473e47u);
    packed_word_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x3e473e47u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0x3c203c20u);
    packed_word_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_80),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_151), 0x3e473e47u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_8, packed_word_9,
                             packed_word_100, packed_word_101);
    packed_word_157 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_157), 0x3e473e47u);
    packed_word_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_82),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_73), 0x3e473e47u);
    packed_word_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x3e473e47u);
    packed_word_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_2), 0x3c203c20u);
    packed_word_158 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_158), 0x3c203c20u);
    packed_word_159 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_159), 0x3c203c20u);
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_76, packed_word_77,
                             packed_word_102, packed_word_103);
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0x3e473e47u);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_98),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_2 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_2), 0x3e473e47u);
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_86),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_133 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_158), 0x3e473e47u);
    packed_word_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_88),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_134 = (packed_word_134 << 0x5u) + 0x7ff88000u;
    packed_word_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_90),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_97 = (packed_word_97 << 0x5u) + 0x7ff88000u;
    packed_word_93 = (packed_word_72 << 0x5u) + 0x7ff88000u;
    packed_word_95 = (packed_word_74 << 0x5u) + 0x7ff88000u;
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_99),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_159 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_159), 0x3e473e47u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_87),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_94 = (packed_word_94 << 0x5u) + 0x7ff88000u;
    packed_word_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_89),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_96 = (packed_word_157 << 0x5u) + 0x7ff88000u;
    packed_word_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_91),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_75 = (packed_word_73 << 0x5u) + 0x7ff88000u;
    packed_word_92 = (packed_word_92 << 0x5u) + 0x7ff88000u;
    packed_word_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_80), 0x3c203c20u);
    packed_word_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_82), 0x3c203c20u);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                   select_half_operand<2, false, false>(packed_word_95));
    packed_word_157 = (packed_word_3 << 0x5u) + 0x7ff88000u;
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_158 = (packed_word_2 << 0x5u) + 0x7ff88000u;
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                    select_half_operand<2, false, false>(packed_word_96));
    packed_word_133 = (packed_word_133 << 0x5u) + 0x7ff88000u;
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                    select_half_operand<2, false, false>(packed_word_92));
    packed_word_2 = (packed_word_159 << 0x5u) + 0x7ff88000u;
    packed_word_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_80), 0x3e473e47u);
    packed_word_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_82), 0x3e473e47u);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3c203c20u);
    packed_word_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_86), 0x3c203c20u);
    packed_word_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_88), 0x3c203c20u);
    packed_word_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_90), 0x3c203c20u);
    packed_word_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3c203c20u);
    packed_word_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_87), 0x3c203c20u);
    packed_word_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_89), 0x3c203c20u);
    packed_word_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_91), 0x3c203c20u);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3e473e47u);
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_158),
                                   select_half_operand<2, false, false>(packed_word_133));
    packed_word_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_86), 0x3e473e47u);
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                    select_half_operand<2, false, false>(packed_word_73));
    packed_word_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_88), 0x3e473e47u);
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_157),
                                    select_half_operand<2, false, false>(packed_word_2));
    packed_word_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_90), 0x3e473e47u);
    packed_word_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3e473e47u);
    packed_word_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_87), 0x3e473e47u);
    packed_word_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_89), 0x3e473e47u);
    packed_word_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_91), 0x3e473e47u);
    packed_word_68 = (packed_word_80 << 0x5u) + 0x7ff88000u;
    packed_word_69 = (packed_word_82 << 0x5u) + 0x7ff88000u;
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_131 = (packed_word_98 << 0x5u) + 0x7ff88000u;
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                   select_half_operand<2, false, false>(packed_word_73));
    packed_word_132 = (packed_word_86 << 0x5u) + 0x7ff88000u;
    packed_word_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_78),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_103 = (packed_word_88 << 0x5u) + 0x7ff88000u;
    packed_word_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_84),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_126 = (packed_word_90 << 0x5u) + 0x7ff88000u;
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                    select_half_operand<2, false, false>(packed_word_69));
    packed_word_130 = (packed_word_99 << 0x5u) + 0x7ff88000u;
    packed_word_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_81),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_129 = (packed_word_87 << 0x5u) + 0x7ff88000u;
    packed_word_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_83),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_127 = (packed_word_89 << 0x5u) + 0x7ff88000u;
    packed_word_128 = (packed_word_91 << 0x5u) + 0x7ff88000u;
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_79),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_85),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_78), 0x3c203c20u);
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                    select_half_operand<2, false, false>(packed_word_132));
    packed_word_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0x3c203c20u);
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                    select_half_operand<2, false, false>(packed_word_126));
    packed_word_72 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                    select_half_operand<2, false, false>(packed_word_72));
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_70),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_81), 0x3c203c20u);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                   select_half_operand<2, false, false>(packed_word_129));
    packed_word_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_83), 0x3c203c20u);
    packed_word_80 = add_half_pairs(select_half_operand<2, false, false>(packed_word_127),
                                    select_half_operand<2, false, false>(packed_word_128));
    packed_word_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_71),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_79), 0x3c203c20u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_101),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_85), 0x3c203c20u);
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_74));
    packed_word_150 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_78), 0x3e473e47u);
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                    select_half_operand<2, false, false>(packed_word_80));
    packed_word_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0x3e473e47u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3c203c20u);
    packed_word_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_70), 0x3c203c20u);
    packed_word_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_81), 0x3e473e47u);
    packed_word_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_83), 0x3e473e47u);
    packed_word_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_71), 0x3c203c20u);
    packed_word_141 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_79), 0x3e473e47u);
    packed_word_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_85), 0x3e473e47u);
    packed_word_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3c203c20u);
    packed_word_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3e473e47u);
    packed_word_136 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_70), 0x3e473e47u);
    packed_word_150 = (packed_word_150 << 0x5u) + 0x7ff88000u;
    packed_word_140 = (packed_word_84 << 0x5u) + 0x7ff88000u;
    packed_word_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_71), 0x3e473e47u);
    packed_word_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3e473e47u);
    packed_word_71 = (packed_word_5 << 0x5u) + 0x7ff88000u;
    packed_word_70 = (packed_word_83 << 0x5u) + 0x7ff88000u;
    packed_word_141 = (packed_word_141 << 0x5u) + 0x7ff88000u;
    packed_word_139 = (packed_word_85 << 0x5u) + 0x7ff88000u;
    packed_word_83 = add_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                    select_half_operand<2, false, false>(packed_word_140));
    packed_word_138 = (packed_word_100 << 0x5u) + 0x7ff88000u;
    packed_word_136 = (packed_word_136 << 0x5u) + 0x7ff88000u;
    packed_word_5 = (packed_word_82 << 0x5u) + 0x7ff88000u;
    packed_word_82 = add_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                    select_half_operand<2, false, false>(packed_word_70));
    packed_word_137 = (packed_word_101 << 0x5u) + 0x7ff88000u;
    packed_word_84 = add_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                    select_half_operand<2, false, false>(packed_word_139));
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_83));
    packed_word_83 = add_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                    select_half_operand<2, false, false>(packed_word_136));
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_82));
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_84));
    packed_word_82 = add_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                    select_half_operand<2, false, false>(packed_word_5));
    packed_word_73 = add_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                    select_half_operand<2, false, false>(packed_word_83));
    packed_word_83 = packed_word_164 * 0x4u;
    {
        uint32_t loaded_word;
        loaded_word = packed_word_164 & 0x1u;
        condition_4 = loaded_word != 0;
    }
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_82));
    {
        uint32_t loaded_word;
        loaded_word = packed_word_164 & 0x2u;
        condition_5 = loaded_word != 0;
    }
    packed_word_87 = packed_word_83 & 0x1cu;
    packed_word_82 = condition_4 ? packed_word_3 : packed_word_72;
    packed_word_83 = condition_4 ? packed_word_74 : packed_word_73;
    packed_word_72 = condition_4 ? packed_word_72 : packed_word_3;
    packed_word_73 = condition_4 ? packed_word_73 : packed_word_74;
    packed_word_3 =
        uint32_t(int64_t(int32_t(packed_word_164)) * (1ll << 0x1du) >> 32) + packed_word_87;
    packed_word_74 = !condition_5 ? packed_word_82 : packed_word_83;
    packed_word_87 = !condition_5 ? packed_word_83 : packed_word_82;
    packed_word_86 = !condition_5 ? packed_word_72 : packed_word_73;
    packed_word_72 = !condition_5 ? packed_word_73 : packed_word_72;
    packed_word_161 = 0x1u ^ packed_word_3;
    packed_word_160 = 0x2u ^ packed_word_3;
    packed_word_159 = 0x3u ^ packed_word_3;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_74), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_74 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_86), "r"(packed_word_161), "r"(0x1fu), "r"(__activemask()));
        packed_word_73 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_87), "r"(packed_word_160), "r"(0x1fu), "r"(__activemask()));
        packed_word_87 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_72), "r"(packed_word_159), "r"(0x1fu), "r"(__activemask()));
        packed_word_72 = loaded_word;
    }
    {
        uint32_t loaded_word;
        loaded_word = packed_word_164 & 0x8u;
        condition_6 = loaded_word != 0;
    }
    packed_word_88 = !condition_6 ? packed_word_74 : packed_word_73;
    packed_word_73 = !condition_6 ? packed_word_73 : packed_word_74;
    packed_word_89 = !condition_6 ? packed_word_87 : packed_word_72;
    packed_word_98 = !condition_6 ? packed_word_72 : packed_word_87;
    packed_word_74 = !condition_3 ? packed_word_88 : packed_word_89;
    packed_word_87 = !condition_3 ? packed_word_73 : packed_word_98;
    packed_word_86 = !condition_3 ? packed_word_89 : packed_word_88;
    packed_word_73 = !condition_3 ? packed_word_98 : packed_word_73;
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_87));
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_86));
    packed_word_74 = add_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                    select_half_operand<2, false, false>(packed_word_73));
    packed_word_74 = add_half_pairs(select_half_operand<0, false, false>(packed_word_74),
                                    select_half_operand<1, false, false>(packed_word_74));
    packed_word_74 = permute_signed(packed_word_74, packed_word_74, 0x5410u);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x4100410u);
    packed_word_162 = uint32_t(int32_t(packed_word_165) >> 0x2u);
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_50, packed_word_51, 0u,
                             0u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_52, packed_word_53, 0u,
                             0u);
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_54, packed_word_55, 0u,
                             0u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_44, packed_word_45, 0u,
                             0u);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_46, packed_word_47, 0u,
                             0u);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_56, packed_word_57, 0u,
                             0u);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_60 = half_to_float_bits((packed_word_74 & 65535u) * 65537u);
    packed_word_74 = half_to_float_bits((packed_word_74 >> 16) * 65537u);
    packed_word_62 = packed_word_162 + 0x8u;
    packed_word_60 = reciprocal_bits(packed_word_60);
    packed_word_73 = uint32_t(int32_t(packed_word_62) >> 0x1fu);
    packed_word_61 = reciprocal_bits(packed_word_74);
    packed_word_73 = uint32_t(combine_words(packed_word_73, 0u) << 0x5u >> 32) + packed_word_62;
    packed_word_73 = packed_word_73 & 0xffffffe0u;
    packed_word_63 = uint32_t(int32_t(packed_word_165) >> 0x1fu);
    packed_word_113 = transpose_half_tile(packed_word_113);
    packed_word_151 = packed_word_62 + (0u - packed_word_73);
    packed_word_63 = uint32_t(combine_words(packed_word_63, 0u) << 0x5u >> 32) + packed_word_162;
    packed_word_86 = transpose_half_tile(packed_word_112);
    packed_word_123 = transpose_half_tile(packed_word_123);
    packed_word_102 = pack_half(packed_word_61, packed_word_60);
    packed_word_73 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_12, packed_word_13) + 0x50e0ull, memory_ranges, error_counts);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_102), "r"(packed_word_151), "r"(0x1fu), "r"(__activemask()));
        packed_word_151 = loaded_word;
    }
    packed_word_72 = packed_word_63 & 0xffffffe0u;
    packed_word_87 = transpose_half_tile(packed_word_122);
    packed_word_61 = packed_word_162 + (0u - packed_word_72);
    packed_word_60 = uint32_t(int64_t(int32_t(packed_word_165)) * (1ll << 0x1eu) >> 32) + 0x10u;
    packed_word_74 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_14, packed_word_15) + 0x50e0ull, memory_ranges, error_counts);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_102), "r"(packed_word_61), "r"(0x1fu), "r"(__activemask()));
        packed_word_156 = loaded_word;
    }
    packed_word_63 = uint32_t(int32_t(packed_word_60) >> 0x1fu);
    packed_word_63 = uint32_t(combine_words(packed_word_63, 0u) << 0x5u >> 32) + packed_word_60;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x4ce0ull, packed_word_12, packed_word_13,
        packed_word_14, packed_word_15, memory_ranges, error_counts);
    packed_word_151 = permute_signed(packed_word_151, packed_word_151, 0x5410u);
    packed_word_63 = packed_word_63 & 0xffffffe0u;
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_112 = packed_word_60 + (0u - packed_word_63);
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(0u));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_75 = uint32_t(int64_t(int32_t(packed_word_165)) * (1ll << 0x1eu) >> 32) + 0x18u;
    packed_word_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                    select_half_operand<2, false, false>(0u));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(packed_word_96));
    packed_word_96 = uint32_t(int32_t(packed_word_75) >> 0x1fu);
    packed_word_94 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_156 = permute_signed(packed_word_156, packed_word_156, 0x5410u);
    packed_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(packed_word_113));
    packed_word_113 = uint32_t(combine_words(packed_word_96, 0u) << 0x5u >> 32) + packed_word_75;
    packed_word_95 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                         select_half_operand<2, false, false>(packed_word_156));
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(0u));
    packed_word_122 = packed_word_113 & 0xffffffe0u;
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                         select_half_operand<2, false, false>(packed_word_156));
    packed_word_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(0u));
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                         select_half_operand<2, false, false>(packed_word_156));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_123),
                                   select_half_operand<2, false, false>(0u));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_94));
    packed_word_119 = transpose_half_tile(packed_word_119);
    packed_word_113 = packed_word_75 + (0u - packed_word_122);
    packed_word_94 = transpose_half_tile(packed_word_117);
    packed_word_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(packed_word_60));
    packed_word_75 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_16, packed_word_17) + 0x50e0ull, memory_ranges, error_counts);
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(packed_word_95));
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                         select_half_operand<2, false, false>(packed_word_156));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_95 = transpose_half_tile(packed_word_116);
    packed_word_92 = transpose_half_tile(packed_word_115);
    packed_word_118 = transpose_half_tile(packed_word_118);
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_121 = transpose_half_tile(packed_word_121);
    packed_word_125 = transpose_half_tile(packed_word_125);
    packed_word_149 = transpose_half_tile(packed_word_149);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_102), "r"(packed_word_112), "r"(0x1fu), "r"(__activemask()));
        packed_word_112 = loaded_word;
    }
    packed_word_93 = transpose_half_tile(packed_word_114);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_102), "r"(packed_word_113), "r"(0x1fu), "r"(__activemask()));
        packed_word_134 = loaded_word;
    }
    packed_word_120 = transpose_half_tile(packed_word_120);
    packed_word_96 = transpose_half_tile(packed_word_124);
    packed_word_97 = transpose_half_tile(packed_word_148);
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(0u));
    packed_word_119 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_119),
                                    select_half_operand<2, false, false>(0u));
    packed_word_102 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_18, packed_word_19) + 0x50e0ull, memory_ranges, error_counts);
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(packed_word_94));
    packed_word_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(0u));
    packed_word_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(packed_word_119));
    packed_word_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                    select_half_operand<2, false, false>(0u));
    packed_word_125 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_125),
                                    select_half_operand<2, false, false>(0u));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x4ee0ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    packed_word_118 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_149),
                                    select_half_operand<2, false, false>(0u));
    packed_word_135 = permute_signed(packed_word_112, packed_word_112, 0x5410u);
    packed_word_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                   select_half_operand<2, false, false>(packed_word_92));
    load_two_boundary_words<CheckBounds>(combine_words(packed_word_10, packed_word_11) + 0x5130ull,
                                         packed_word_112, packed_word_113, memory_ranges,
                                         error_counts);
    packed_word_134 = permute_signed(packed_word_134, packed_word_134, 0x5410u);
    packed_word_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                   select_half_operand<2, false, false>(packed_word_121));
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_125));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(packed_word_118));
    packed_word_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_129),
                                          select_half_operand<2, false, false>(packed_word_134));
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_86, packed_word_87, 0u,
                             0u);
    packed_word_126 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_126),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_128),
                                          select_half_operand<2, false, false>(packed_word_134));
    packed_word_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                    select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_116, packed_word_117, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_92, packed_word_93, 0u,
                             0u);
    multiply_accumulate_half(packed_word_118, packed_word_119, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_94, packed_word_95, 0u,
                             0u);
    multiply_accumulate_half(packed_word_120, packed_word_121, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_96, packed_word_97, 0u,
                             0u);
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                         select_half_operand<2, false, false>(packed_word_135));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_132),
                                   select_half_operand<2, false, false>(0u));
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                         select_half_operand<2, false, false>(packed_word_134));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_129),
                                   select_half_operand<2, false, false>(0u));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                   select_half_operand<2, false, false>(packed_word_63));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_127),
                                         select_half_operand<2, false, false>(packed_word_134));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                         select_half_operand<2, false, false>(packed_word_135));
    packed_word_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_126),
                                    select_half_operand<2, false, false>(0u));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_128));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(packed_word_103));
    multiply_accumulate_half(packed_word_126, packed_word_127, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_86, packed_word_87, 0u,
                             0u);
    multiply_accumulate_half(packed_word_124, packed_word_125, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_92, packed_word_93, 0u,
                             0u);
    multiply_accumulate_half(packed_word_122, packed_word_123, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_94, packed_word_95, 0u,
                             0u);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_96, packed_word_97, 0u,
                             0u);
    load_two_boundary_words<CheckBounds>(combine_words(packed_word_10, packed_word_11) + 0x5330ull,
                                         packed_word_60, packed_word_61, memory_ranges,
                                         error_counts);
    packed_word_103 = transpose_half_tile(packed_word_101);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_91 = transpose_half_tile(packed_word_91);
    packed_word_90 = transpose_half_tile(packed_word_90);
    multiply_accumulate_half(packed_word_128, packed_word_129, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_56 = transpose_half_tile(packed_word_100);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_46, packed_word_47, 0u,
                             0u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_70 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_70),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_45 = transpose_half_tile(packed_word_45);
    packed_word_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_91 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(0u));
    packed_word_44 = transpose_half_tile(packed_word_44);
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_70),
                                   select_half_operand<2, false, false>(0u));
    packed_word_89 = transpose_half_tile(packed_word_89);
    packed_word_129 = transpose_half_tile(packed_word_129);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_48 = transpose_half_tile(packed_word_47);
    packed_word_99 = transpose_half_tile(packed_word_99);
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_71),
                                   select_half_operand<2, false, false>(packed_word_70));
    packed_word_101 = transpose_half_tile(packed_word_101);
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                   select_half_operand<2, false, false>(packed_word_91));
    packed_word_88 = transpose_half_tile(packed_word_88);
    packed_word_49 = transpose_half_tile(packed_word_46);
    packed_word_128 = transpose_half_tile(packed_word_128);
    packed_word_90 = transpose_half_tile(packed_word_98);
    packed_word_91 = transpose_half_tile(packed_word_100);
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                        select_half_operand<2, false, false>(packed_word_151));
    packed_word_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                   select_half_operand<2, false, false>(0u));
    packed_word_157 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_157),
                                          select_half_operand<2, false, false>(packed_word_151));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(0u));
    packed_word_158 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_158),
                                          select_half_operand<2, false, false>(packed_word_156));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_2),
                                  select_half_operand<2, false, false>(0u));
    packed_word_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                          select_half_operand<2, false, false>(packed_word_156));
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                          select_half_operand<2, false, false>(packed_word_156));
    packed_word_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                          select_half_operand<2, false, false>(packed_word_156));
    packed_word_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_56),
                                   select_half_operand<2, false, false>(packed_word_57));
    packed_word_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_45));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_157),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_89 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_89),
                                   select_half_operand<2, false, false>(0u));
    packed_word_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_133),
                                    select_half_operand<2, false, false>(0u));
    packed_word_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_156),
                                    select_half_operand<2, false, false>(0u));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_48),
                                   select_half_operand<2, false, false>(0u));
    packed_word_129 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_129),
                                    select_half_operand<2, false, false>(0u));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                  select_half_operand<2, false, false>(0u));
    packed_word_88 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_88),
                                   select_half_operand<2, false, false>(packed_word_89));
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_158),
                                   select_half_operand<2, false, false>(packed_word_133));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                   select_half_operand<2, false, false>(packed_word_156));
    packed_word_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_49),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_89 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                   select_half_operand<2, false, false>(packed_word_129));
    packed_word_90 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                   select_half_operand<2, false, false>(packed_word_99));
    packed_word_91 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_140 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<2, false, false>(packed_word_135));
    packed_word_139 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                          select_half_operand<2, false, false>(packed_word_134));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                        select_half_operand<2, false, false>(packed_word_134));
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_56, packed_word_57,
                             packed_word_114, packed_word_115);
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_140),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_139),
                                  select_half_operand<2, false, false>(0u));
    packed_word_135 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_135),
                                    select_half_operand<2, false, false>(0u));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_116, packed_word_117, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_70, packed_word_71,
                             packed_word_116, packed_word_117);
    multiply_accumulate_half(packed_word_118, packed_word_119, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_88, packed_word_89,
                             packed_word_118, packed_word_119);
    multiply_accumulate_half(packed_word_120, packed_word_121, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_90, packed_word_91,
                             packed_word_120, packed_word_121);
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                         select_half_operand<2, false, false>(packed_word_134));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                         select_half_operand<2, false, false>(packed_word_134));
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_150),
                                   select_half_operand<2, false, false>(packed_word_49));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_138),
                                   select_half_operand<2, false, false>(packed_word_135));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(packed_word_48));
    multiply_accumulate_half(packed_word_124, packed_word_125, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_70, packed_word_71,
                             packed_word_124, packed_word_125);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_56, packed_word_57,
                             packed_word_126, packed_word_127);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_90, packed_word_91,
                             packed_word_62, packed_word_63);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_50, packed_word_51, 0u,
                             0u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_88, packed_word_89,
                             packed_word_122, packed_word_123);
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_124),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_125),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_52, packed_word_53, 0u,
                             0u);
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_48),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                        select_half_operand<2, false, false>(packed_word_82));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_49),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                        select_half_operand<2, false, false>(packed_word_84));
    packed_word_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                      select_half_operand<2, false, false>(packed_word_80),
                                      select_half_operand<2, false, false>(packed_word_5));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                      select_half_operand<2, false, false>(packed_word_78),
                                      select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                   select_half_operand<2, false, false>(0u));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(0u));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(0u));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_51),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_62 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                    select_half_operand<2, false, false>(packed_word_2));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                   select_half_operand<2, false, false>(0u));
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(packed_word_65));
    packed_word_20 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_20),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(packed_word_47));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_62), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_65 = loaded_word;
    }
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_119),
                                   select_half_operand<2, false, false>(packed_word_64));
    packed_word_21 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_21),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_22 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_23 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                        select_half_operand<2, false, false>(packed_word_85));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                        select_half_operand<2, false, false>(packed_word_83));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                      select_half_operand<2, false, false>(packed_word_79),
                                      select_half_operand<2, false, false>(packed_word_2));
    packed_word_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                      select_half_operand<2, false, false>(packed_word_81),
                                      select_half_operand<2, false, false>(packed_word_5));
    multiply_accumulate_half(packed_word_20, packed_word_21, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_12, packed_word_13,
                             packed_word_20, packed_word_21);
    packed_word_63 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                    select_half_operand<2, false, false>(packed_word_2));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                        select_half_operand<2, false, false>(packed_word_54));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                      select_half_operand<2, false, false>(packed_word_68),
                                      select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_12, packed_word_13,
                             packed_word_22, packed_word_23);
    packed_word_23 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                         select_half_operand<2, false, false>(packed_word_52));
    packed_word_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                      select_half_operand<2, false, false>(packed_word_58),
                                      select_half_operand<2, false, false>(packed_word_29));
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                        select_half_operand<2, false, false>(packed_word_55));
    packed_word_66 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                    select_half_operand<2, false, false>(packed_word_29));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                        select_half_operand<2, false, false>(packed_word_53));
    packed_word_22 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_62 = add_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                    select_half_operand<2, false, false>(packed_word_65));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                      select_half_operand<2, false, false>(packed_word_59),
                                      select_half_operand<2, false, false>(packed_word_2));
    packed_word_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                      select_half_operand<2, false, false>(packed_word_69),
                                      select_half_operand<2, false, false>(packed_word_5));
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_2));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_62), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_37 = loaded_word;
    }
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_14, packed_word_15,
                             packed_word_22, packed_word_23);
    packed_word_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_75));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_5), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_36 = loaded_word;
    }
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_74));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_65 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_63), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_64 = loaded_word;
    }
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_14, packed_word_15,
                             packed_word_30, packed_word_31);
    packed_word_37 = add_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                    select_half_operand<2, false, false>(packed_word_37));
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_102));
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_102));
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_36));
    packed_word_36 = permute_signed(packed_word_37, packed_word_37, 0x5432u);
    packed_word_65 = add_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                    select_half_operand<2, false, false>(packed_word_65));
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_16, packed_word_17,
                             packed_word_28, packed_word_29);
    image_mma_half(packed_word_20, packed_word_21, packed_word_20, packed_word_21, packed_word_22,
                   packed_word_23, packed_word_112, packed_word_113, 0u, 0u);
    packed_word_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_75));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_22 = loaded_word;
    }
    packed_word_37 = add_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                    select_half_operand<2, false, false>(packed_word_36));
    packed_word_41 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_75));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_5), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_36 = loaded_word;
    }
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_102));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_102));
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_18, packed_word_19,
                             packed_word_30, packed_word_31);
    packed_word_63 = add_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                    select_half_operand<2, false, false>(packed_word_64));
    image_mma_half(packed_word_12, packed_word_13, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_112, packed_word_113, 0u, 0u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_63), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_2 = loaded_word;
    }
    multiply_accumulate_half(packed_word_16, packed_word_17, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_16, packed_word_17,
                             packed_word_40, packed_word_41);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_18, packed_word_19,
                             packed_word_42, packed_word_43);
    packed_word_22 = add_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                    select_half_operand<2, false, false>(packed_word_22));
    image_mma_half(packed_word_20, packed_word_21, packed_word_28, packed_word_29, packed_word_30,
                   packed_word_31, packed_word_60, packed_word_61, packed_word_20, packed_word_21);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_36));
    packed_word_23 = permute_signed(packed_word_22, packed_word_22, 0x5432u);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_28 = permute_signed(packed_word_5, packed_word_5, 0x5432u);
    packed_word_22 = add_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                    select_half_operand<2, false, false>(packed_word_23));
    packed_word_15 = permute_signed(packed_word_2, packed_word_2, 0x5432u);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(packed_word_28));
    packed_word_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_22), 0x4100410u);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x4100410u);
    packed_word_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_5), 0x4100410u);
    packed_word_23 = half_to_float_bits((packed_word_22 & 65535u) * 65537u);
    image_mma_half(packed_word_16, packed_word_17, packed_word_16, packed_word_17, packed_word_18,
                   packed_word_19, packed_word_60, packed_word_61, packed_word_12, packed_word_13);
    packed_word_14 = half_to_float_bits((packed_word_37 & 65535u) * 65537u);
    packed_word_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_2), 0x4100410u);
    packed_word_22 = half_to_float_bits((packed_word_22 >> 16) * 65537u);
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_37 = half_to_float_bits((packed_word_37 >> 16) * 65537u);
    packed_word_12 = half_to_float_bits((packed_word_5 & 65535u) * 65537u);
    packed_word_5 = half_to_float_bits((packed_word_5 >> 16) * 65537u);
    packed_word_24 = blockIdx.x;
    packed_word_15 = half_to_float_bits((packed_word_2 & 65535u) * 65537u);
    packed_word_23 = rsqrt_bits(packed_word_23);
    packed_word_2 = half_to_float_bits((packed_word_2 >> 16) * 65537u);
    packed_word_13 = condition_4 ? packed_word_21 : packed_word_20;
    packed_word_20 = condition_4 ? packed_word_20 : packed_word_21;
    packed_word_21 = parameters[10];
    packed_word_22 = rsqrt_bits(packed_word_22);
    packed_word_14 = rsqrt_bits(packed_word_14);
    packed_word_37 = rsqrt_bits(packed_word_37);
    packed_word_12 = rsqrt_bits(packed_word_12);
    packed_word_5 = rsqrt_bits(packed_word_5);
    packed_word_18 = condition_4 ? packed_word_17 : packed_word_16;
    packed_word_15 = rsqrt_bits(packed_word_15);
    packed_word_2 = rsqrt_bits(packed_word_2);
    packed_word_17 = condition_4 ? packed_word_16 : packed_word_17;
    packed_word_163 = packed_word_164;
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_26 = pack_half(packed_word_22, packed_word_23);
    packed_word_39 = pack_half(packed_word_37, packed_word_14);
    packed_word_22 = pack_half(packed_word_5, packed_word_12);
    packed_word_14 = !condition_5 ? packed_word_20 : packed_word_17;
    packed_word_5 = packed_word_163 & 0xfffffff0u;
    packed_word_12 = !condition_5 ? packed_word_13 : packed_word_18;
    packed_word_19 = 0u - packed_word_5 + packed_word_164;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_14), "r"(packed_word_161), "r"(0x1fu), "r"(__activemask()));
        packed_word_14 = loaded_word;
    }
    packed_word_16 = pack_half(packed_word_2, packed_word_15);
    packed_word_164 = packed_word_164;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_12), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_15 = loaded_word;
    }
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                        select_half_operand<2, false, false>(packed_word_39));
    packed_word_24 = packed_word_24 * 0x8u + packed_word_21;
    packed_word_21 = uint32_t(int32_t(packed_word_164) >> 0x5u);
    packed_word_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_2 = uint32_t(int32_t(packed_word_19) >> 0x1fu);
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                         select_half_operand<2, false, false>(packed_word_16));
    packed_word_81 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                         select_half_operand<2, false, false>(packed_word_16));
    packed_word_85 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                         select_half_operand<2, false, false>(packed_word_16));
    packed_word_83 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<2, false, false>(packed_word_16));
    packed_word_16 = uint32_t(int32_t(packed_word_163) >> 0x4u);
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_22));
    packed_word_53 = packed_word_21 * 0x4u + packed_word_0;
    packed_word_2 = uint32_t(combine_words(packed_word_2, 0u) << 0x2u >> 32) + packed_word_19;
    packed_word_21 = packed_word_21 * (0u - 0x8u) + packed_word_24;
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                        select_half_operand<2, false, false>(packed_word_26));
    packed_word_52 = packed_word_2 & 0xfffffffcu;
    packed_word_16 = packed_word_16 * 0x4u + packed_word_21;
    packed_word_20 = !condition_5 ? packed_word_17 : packed_word_20;
    packed_word_13 = !condition_5 ? packed_word_18 : packed_word_13;
    packed_word_52 = packed_word_16 + packed_word_19 + (0u - packed_word_52);
    packed_word_24 = !condition_6 ? packed_word_15 : packed_word_14;
    packed_word_0 = !condition_6 ? packed_word_14 : packed_word_15;
    packed_word_53 =
        uint32_t(int64_t(int32_t(packed_word_2)) * (1ll << 0x1eu) >> 32) + packed_word_53;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_160), "r"(0x1fu), "r"(__activemask()));
        packed_word_15 = loaded_word;
    }
    condition_0 = int32_t(packed_word_52) >= int32_t(uniform_word_9);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_20), "r"(packed_word_159), "r"(0x1fu), "r"(__activemask()));
        packed_word_20 = loaded_word;
    }
    packed_word_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_26));
    condition_1 = int32_t(packed_word_53) >= int32_t(uniform_word_8) || condition_0;
    packed_word_16 = packed_word_52 | packed_word_53;
    packed_word_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                         select_half_operand<2, false, false>(packed_word_22));
    packed_word_2 = __float_as_uint(__int2float_rn(int32_t(uniform_word_6)));
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_54 = __float_as_uint(__int2float_rn(int32_t(uniform_word_7)));
    condition_1 = int32_t(packed_word_16) < int32_t(0u) || condition_1;
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_5 = reciprocal_bits(packed_word_2);
    packed_word_58 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                         select_half_operand<2, false, false>(packed_word_26));
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_26));
    packed_word_80 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                         select_half_operand<2, false, false>(packed_word_39));
    packed_word_55 = reciprocal_bits(packed_word_54);
    packed_word_84 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_39));
    packed_word_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<2, false, false>(packed_word_39));
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                         select_half_operand<2, false, false>(packed_word_22));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_22));
    packed_word_58 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_80 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_80),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_84 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_82),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_81 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_81),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_85 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_83 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_83),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_66 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<0, false, false>(packed_word_4));
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                         select_half_operand<0, false, false>(packed_word_4));

    packed_word_49 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_58),
                                   select_half_operand<2, false, false>(0u));
    packed_word_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                         select_half_operand<2, false, false>(packed_word_73));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_68),
                                   select_half_operand<2, false, false>(0u));
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                         select_half_operand<2, false, false>(packed_word_74));
    if (condition_3) {
        packed_word_24 = !condition_6 ? packed_word_15 : packed_word_20;
    }
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                         select_half_operand<2, false, false>(packed_word_74));
    if (condition_3) {
        packed_word_0 = !condition_6 ? packed_word_20 : packed_word_15;
    }
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(0u));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_80),
                                   select_half_operand<2, false, false>(0u));
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<2, false, false>(packed_word_75));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_84),
                                   select_half_operand<2, false, false>(0u));
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                          select_half_operand<2, false, false>(packed_word_75));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_82),
                                   select_half_operand<2, false, false>(0u));
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_108),
                                          select_half_operand<2, false, false>(packed_word_75));
    packed_word_73 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(0u));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_109),
                                          select_half_operand<2, false, false>(packed_word_75));
    packed_word_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_81),
                                   select_half_operand<2, false, false>(0u));
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_74 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_85),
                                   select_half_operand<2, false, false>(0u));
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_75 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_83),
                                   select_half_operand<2, false, false>(0u));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_110),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_13),
                                   select_half_operand<2, false, false>(0u));
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_111),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(0u));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_66),
                                   select_half_operand<2, false, false>(0u));
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_69),
                                   select_half_operand<2, false, false>(0u));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_12),
                                   select_half_operand<2, false, false>(0u));
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(0u));
    if (!condition_1) {
        uniform_word_8 = parameters[14];
        uniform_word_9 = parameters[15];
        condition_1 = int32_t(packed_word_53) >= int32_t(uniform_word_7);
        packed_word_20 = parameters[16];
        packed_word_21 = parameters[17];
        packed_word_12 = __float_as_uint(__uint2float_rn(packed_word_52));
        packed_word_13 = __float_as_uint(__uint2float_rn(packed_word_53));
        condition_1 = int32_t(packed_word_52) >= int32_t(uniform_word_6) || condition_1;
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(0x3f000000u)));
        packed_word_16 = parameters[18];
        packed_word_17 = parameters[19];
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float(packed_word_13)), "f"(__uint_as_float(0x3f000000u)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_12)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_14)));
        condition_1 = 0ull == combine_words(uniform_word_8, uniform_word_9) || condition_1;
        if (!condition_1) {
            packed_word_22 = parameters[20];
            packed_word_23 = parameters[21];
        }
        uniform_condition_0 = vote_warp_predicate<true>(condition_1);
        if (!condition_1) {
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_16)),
                  "f"(__uint_as_float(packed_word_20)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_17)),
                  "f"(__uint_as_float(packed_word_21)));
        }
        if (!uniform_condition_0) {
            uniform_word_8 = parameters[14];
            uniform_word_9 = parameters[15];
        }
        if (!condition_1) {
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(packed_word_13)), "f"(__uint_as_float(packed_word_22)));

            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_15))
                : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_23)));

            {
                {
                    uint32_t sampled_channels[4];
                    // Sample the frame texture using its independent texture/sampler handle.
                    image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                              packed_word_14, packed_word_15);
                    packed_word_14 = sampled_channels[0];
                    packed_word_15 = sampled_channels[1];
                    packed_word_12 = sampled_channels[2];
                }
            }

            packed_word_17 = 0x3e000000u;
        }
        packed_word_20 = half_to_float_bits((packed_word_24 & 65535u) * 65537u);
        packed_word_21 = half_to_float_bits((packed_word_24 >> 16) * 65537u);
        packed_word_22 = half_to_float_bits((packed_word_0 & 65535u) * 65537u);
        uniform_word_9 = parameters[13];
        uniform_word_8 = parameters[12];
        uniform_condition_0 = uint32_t(uniform_word_9) != 0u;
        if (!condition_1) {
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(packed_word_17)),
                  "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(packed_word_17)),
                  "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_17)),
                  "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));
        }
        packed_word_15 = 0u;
        if (!condition_1) {
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(uniform_word_8)),
                  "f"(__uint_as_float(packed_word_13)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(uniform_word_8)),
                  "f"(__uint_as_float(packed_word_16)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(uniform_word_8)),
                  "f"(__uint_as_float(packed_word_17)));
        }
        if (condition_1) {
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(uniform_word_8)));

            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(uniform_word_8)));

            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(uniform_word_8)));
        }
        if (uniform_condition_0) {
            packed_word_15 = 0x41000000u;
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_15)),
                  "f"(__uint_as_float(0x3f000000u)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_13)), "f"(__uint_as_float(packed_word_15)),
                  "f"(__uint_as_float(0x3f000000u)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(packed_word_15)),
                  "f"(__uint_as_float(0x3f000000u)));
            packed_word_15 = 0x3f800000u;
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_12)));
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_13)));
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_14)));
        }
        uniform_word_10 = parameters[26];
        uniform_word_11 = parameters[27];
        packed_word_21 = 0x3f800000u;
        uniform_condition_0 = combine_words(uniform_word_10, uniform_word_11) != 0ull;
        if (uniform_condition_0) {
            packed_word_16 = parameters[26];
            packed_word_17 = parameters[27];
            packed_word_16 = read_boundary_half<CheckBounds>(
                combine_words(packed_word_16, packed_word_17), memory_ranges, error_counts);
            packed_word_21 = 0u;
            packed_word_20 = half_to_float_bits((packed_word_16 & 65535u) * 65537u);
            {
                uint32_t loaded_word;
                asm("{.reg .pred p; setp.ne.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                    : "=r"(loaded_word)
                    : "f"(__uint_as_float((packed_word_20 & 0x7fffffffu))),
                      "f"(__uint_as_float(0x7f800000u)));
                condition_1 = bool(loaded_word);
            }
            if (condition_1) {
                asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_21))
                    : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_20)));
            }
        }
        uniform_word_10 = parameters[22];
        uniform_word_11 = parameters[23];
        condition_1 = int32_t(packed_word_53) >= int32_t(uniform_word_7);
        packed_word_16 = uint32_t(condition_0);

        uniform_word_12 = parameters[24];
        uniform_word_13 = parameters[25];
        {
            uint32_t loaded_word;
            asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                : "=r"(loaded_word)
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(0u)));
            condition_0 = bool(loaded_word);
        }
        condition_1 = int32_t(packed_word_52) < int32_t(uniform_word_6) && !condition_1;
        packed_word_0 = half_to_float_bits((packed_word_0 >> 16) * 65537u);
        uniform_condition_0 = combine_words(uniform_word_10, uniform_word_11) != 0ull;
        uniform_condition_0 = uint32_t(uniform_word_9) != 0u && uniform_condition_0;
        condition_1 = 0ull != combine_words(uniform_word_12, uniform_word_13) && condition_1;
        condition_2 = uniform_condition_0;
        condition_1 = condition_1 & condition_0 & condition_2;
        condition_0 = int32_t(packed_word_16) != int32_t(0u);
        if (condition_1) {
            packed_word_22 = parameters[36];
            packed_word_23 = parameters[37];
            uniform_word_12 = parameters[32];
            uniform_word_13 = parameters[33];
            uniform_word_14 = parameters[34];
            uniform_word_15 = parameters[35];
            uniform_word_8 = parameters[40];
            uniform_word_9 = parameters[41];
            packed_word_24 = parameters[38];
            packed_word_25 = parameters[39];
            uniform_word_10 = parameters[24];
            uniform_word_11 = parameters[25];
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_23)),
                  "f"(__uint_as_float(uniform_word_15)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_24)),
                  "f"(__uint_as_float(packed_word_22)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_22))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(packed_word_25)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(uniform_word_8)));
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_10, uniform_word_11),
                          packed_word_22, packed_word_23);
                packed_word_22 = sampled_channels[0];
                packed_word_23 = sampled_channels[1];
            }
            packed_word_16 = parameters[28];
            packed_word_17 = parameters[29];
            uniform_word_8 = parameters[42];
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_33))
                : "f"(__uint_as_float(packed_word_2)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
            condition_1 = uint32_t(packed_word_16) != 0u;
            if (condition_1) {
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(uniform_word_9)),
                      "f"(__uint_as_float(packed_word_19)));

                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_18))
                    : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(uniform_word_8)),
                      "f"(__uint_as_float(packed_word_18)));
            }
            uniform_word_8 = parameters[22];
            uniform_word_9 = parameters[23];
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_20))
                : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(packed_word_18)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
            asm("cvt.rmi.ftz.f32.f32 %0,%1;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_16)));
            asm("cvt.rmi.ftz.f32.f32 %0,%1;"
                : "=f"(reinterpret_cast<float&>(packed_word_20))
                : "f"(__uint_as_float(packed_word_20)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_25))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(0x3f000000u)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float((packed_word_25 ^ 0x80000000u))));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(0x3f000000u)));
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_19)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_18))
                : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(packed_word_18)),
                  "f"(__uint_as_float((packed_word_23 ^ 0x80000000u))));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_22))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_19)));
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_18))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_18)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_22)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(0x40200000u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_29))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_18)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_16)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(0x3fc00000u)),
                  "f"(__uint_as_float((packed_word_27 ^ 0x80000000u))));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_28))
                : "f"(__uint_as_float((packed_word_22 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_16)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_24))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_29)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_20))
                : "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_22)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_29)), "f"(__uint_as_float(0x40200000u)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(0x3f800000u)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_18))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_24)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_30))
                : "f"(__uint_as_float((packed_word_29 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_24)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float((packed_word_20 ^ 0x80000000u))),
                  "f"(__uint_as_float(0x3f800000u)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_24)), "f"(__uint_as_float(0x3fc00000u)),
                  "f"(__uint_as_float((packed_word_31 ^ 0x80000000u))));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_29))
                : "f"(__uint_as_float(packed_word_18)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_29)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_28))
                : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(0x3f000000u)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float((packed_word_27 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_16)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(0x3f800000u)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_18))
                : "f"(__uint_as_float((packed_word_29 ^ 0x80000000u))),
                  "f"(__uint_as_float(0x3f800000u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_30))
                : "f"(__uint_as_float(packed_word_30)), "f"(__uint_as_float(0x3f000000u)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float((packed_word_28 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_19)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_22))
                : "f"(__uint_as_float(packed_word_23)),
                  "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float((packed_word_31 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_18)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_32))
                : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(packed_word_16)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_54)),
                  "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_18))
                : "f"(__uint_as_float((packed_word_30 ^ 0x80000000u))),
                  "f"(__uint_as_float(packed_word_19)));
            packed_word_22 = float_minmax(packed_word_22, 0x3f000000u, !true);
            packed_word_24 = reciprocal_bits(packed_word_32);
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(packed_word_18)));
            packed_word_22 = float_minmax(packed_word_22, packed_word_27, true);
            packed_word_26 = reciprocal_bits(packed_word_31);
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_24))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(packed_word_24)),
                  "f"(__uint_as_float(packed_word_25)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_25)),
                  "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_25))
                : "f"(__uint_as_float(packed_word_25)), "f"(__uint_as_float(0x40000000u)));
            packed_word_24 = float_minmax(packed_word_24, 0x3f000000u, !true);
            packed_word_16 = float_minmax(packed_word_16, 0x3f000000u, !true);
            packed_word_25 = float_minmax(packed_word_25, 0x3f000000u, !true);
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_26))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_26)),
                  "f"(__uint_as_float(packed_word_23)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(0x40000000u)));
            packed_word_18 = parameters[30];
            packed_word_19 = parameters[31];
            packed_word_16 = float_minmax(packed_word_16, packed_word_33, true);
            packed_word_34 = float_minmax(packed_word_25, packed_word_33, true);
            packed_word_26 = float_minmax(packed_word_26, 0x3f000000u, !true);
            packed_word_24 = float_minmax(packed_word_24, packed_word_33, true);
            packed_word_23 = float_minmax(packed_word_23, 0x3f000000u, !true);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_16)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_34))
                : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_34)));
            packed_word_26 = float_minmax(packed_word_26, packed_word_27, true);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_24))
                : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_24)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_25))
                : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_22)));
            packed_word_36 = float_minmax(packed_word_23, packed_word_27, true);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_26)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_36)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float(packed_word_17)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_34))
                : "f"(__uint_as_float(packed_word_34)), "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float(packed_word_17)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_24)), "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float(packed_word_17)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_25)), "f"(__uint_as_float(uniform_word_12)),
                  "f"(__uint_as_float(packed_word_18)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_24))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(uniform_word_13)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_23))
                : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(uniform_word_12)),
                  "f"(__uint_as_float(packed_word_18)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(uniform_word_13)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(uniform_word_14)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_27))
                : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(uniform_word_12)),
                  "f"(__uint_as_float(packed_word_18)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_25))
                : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(uniform_word_14)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_26))
                : "f"(__uint_as_float(packed_word_34)), "f"(__uint_as_float(uniform_word_13)));
            packed_word_22 = packed_word_16;
            packed_word_18 = packed_word_16;
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(uniform_word_14)));
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                          packed_word_16, packed_word_17);
                packed_word_16 = sampled_channels[0];
                packed_word_17 = sampled_channels[1];
                packed_word_34 = sampled_channels[2];
            }
            packed_word_23 = packed_word_25;
            packed_word_27 = packed_word_25;
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                          packed_word_24, packed_word_25);
                packed_word_24 = sampled_channels[0];
                packed_word_25 = sampled_channels[1];
                packed_word_33 = sampled_channels[2];
            }
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                          packed_word_22, packed_word_23);
                packed_word_22 = sampled_channels[0];
                packed_word_23 = sampled_channels[1];
                packed_word_35 = sampled_channels[2];
            }
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                          packed_word_18, packed_word_19);
                packed_word_18 = sampled_channels[0];
                packed_word_19 = sampled_channels[1];
                packed_word_36 = sampled_channels[2];
            }
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_8, uniform_word_9),
                          packed_word_26, packed_word_27);
                packed_word_26 = sampled_channels[0];
                packed_word_27 = sampled_channels[1];
                packed_word_37 = sampled_channels[2];
            }
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_40))
                : "f"(__uint_as_float(packed_word_0)),
                  "f"(__uint_as_float((0x3fb8aa3bu ^ 0x80000000u))));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_20))
                : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(packed_word_31)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_29))
                : "f"(__uint_as_float(packed_word_29)), "f"(__uint_as_float(packed_word_32)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_38))
                : "f"(__uint_as_float(packed_word_32)), "f"(__uint_as_float(packed_word_31)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_39))
                : "f"(__uint_as_float(packed_word_30)), "f"(__uint_as_float(packed_word_32)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_28))
                : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_31)));
            packed_word_40 = image_mufu_ex2(packed_word_40);
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(packed_word_29)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_30))
                : "f"(__uint_as_float(packed_word_38)), "f"(__uint_as_float(packed_word_41)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_39)), "f"(__uint_as_float(packed_word_30)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_31))
                : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_31)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_0))
                : "f"(__uint_as_float(packed_word_40)), "f"(__uint_as_float(0x3f800000u)));
            packed_word_31 = reciprocal_bits(packed_word_31);
            packed_word_0 = reciprocal_bits(packed_word_0);
            asm("fma.rn.ftz.sat.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_21))
                : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_21)),
                  "f"(__uint_as_float(0u)));

            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(packed_word_29)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(packed_word_29)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_34))
                : "f"(__uint_as_float(packed_word_34)), "f"(__uint_as_float(packed_word_29)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_24)), "f"(__uint_as_float(packed_word_20)),
                  "f"(__uint_as_float(packed_word_41)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_25)), "f"(__uint_as_float(packed_word_20)),
                  "f"(__uint_as_float(packed_word_16)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_34))
                : "f"(__uint_as_float(packed_word_33)), "f"(__uint_as_float(packed_word_20)),
                  "f"(__uint_as_float(packed_word_34)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(packed_word_38)),
                  "f"(__uint_as_float(packed_word_41)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(packed_word_38)),
                  "f"(__uint_as_float(packed_word_16)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_35))
                : "f"(__uint_as_float(packed_word_35)), "f"(__uint_as_float(packed_word_38)),
                  "f"(__uint_as_float(packed_word_34)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_39)),
                  "f"(__uint_as_float(packed_word_41)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_39)),
                  "f"(__uint_as_float(packed_word_16)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_36))
                : "f"(__uint_as_float(packed_word_36)), "f"(__uint_as_float(packed_word_39)),
                  "f"(__uint_as_float(packed_word_35)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_26))
                : "f"(__uint_as_float(packed_word_26)), "f"(__uint_as_float(packed_word_28)),
                  "f"(__uint_as_float(packed_word_41)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(packed_word_28)),
                  "f"(__uint_as_float(packed_word_16)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_37))
                : "f"(__uint_as_float(packed_word_37)), "f"(__uint_as_float(packed_word_28)),
                  "f"(__uint_as_float(packed_word_36)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_26))
                : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(packed_word_26)),
                  "f"(__uint_as_float((packed_word_12 ^ 0x80000000u))));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_16))
                : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(packed_word_16)),
                  "f"(__uint_as_float((packed_word_13 ^ 0x80000000u))));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_37))
                : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(packed_word_37)),
                  "f"(__uint_as_float((packed_word_14 ^ 0x80000000u))));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_12))
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(packed_word_26)),
                  "f"(__uint_as_float(packed_word_12)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_13))
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(packed_word_16)),
                  "f"(__uint_as_float(packed_word_13)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_14))
                : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(packed_word_37)),
                  "f"(__uint_as_float(packed_word_14)));
        }

        uniform_word_8 = parameters[4];
        // Write the reconstructed four-channel pixel to the bound output surface.
        image_surface(uniform_word_8, packed_word_52, packed_word_53, packed_word_12,
                      packed_word_13, packed_word_14, packed_word_15);
    }

    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x3cd0ull, packed_word_28, packed_word_29,
        packed_word_30, packed_word_31, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x3ed0ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x44d0ull, packed_word_12, packed_word_13,
        packed_word_14, packed_word_15, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x46d0ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x40d0ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x48d0ull, packed_word_20, packed_word_21,
        packed_word_22, packed_word_23, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x42d0ull, packed_word_40, packed_word_41,
        packed_word_42, packed_word_43, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x4ad0ull, packed_word_24, packed_word_25,
        packed_word_26, packed_word_27, memory_ranges, error_counts);
    packed_word_44 = permute_signed(packed_word_44, packed_word_45, 0x5410u);
    uniform_word_8 = parameters[8];
    packed_word_46 = permute_signed(packed_word_46, packed_word_47, 0x5410u);
    packed_word_45 = permute_signed(packed_word_73, packed_word_78, 0x5410u);
    packed_word_47 = permute_signed(packed_word_74, packed_word_75, 0x5410u);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_146, packed_word_147,
                             packed_word_28, packed_word_29);
    packed_word_28 = permute_signed(packed_word_59, packed_word_64, 0x5410u);
    packed_word_29 = permute_signed(packed_word_66, packed_word_69, 0x5410u);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_144, packed_word_145,
                             packed_word_30, packed_word_31);
    packed_word_30 = permute_signed(packed_word_58, packed_word_65, 0x5410u);
    packed_word_31 = permute_signed(packed_word_67, packed_word_68, 0x5410u);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_142, packed_word_143,
                             packed_word_32, packed_word_33);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_152, packed_word_153,
                             packed_word_34, packed_word_35);
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_146, packed_word_147,
                             packed_word_12, packed_word_13);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_144, packed_word_145,
                             packed_word_14, packed_word_15);
    multiply_accumulate_half(packed_word_16, packed_word_17, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_142, packed_word_143,
                             packed_word_16, packed_word_17);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_152, packed_word_153,
                             packed_word_18, packed_word_19);
    packed_word_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_78),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_74),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_32),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_34),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_154, packed_word_155,
                             packed_word_36, packed_word_37);
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_33),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_78), 0x3c203c20u);
    packed_word_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_35),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_6, packed_word_7,
                             packed_word_38, packed_word_39);
    packed_word_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x3c203c20u);
    packed_word_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_12),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x3c203c20u);
    multiply_accumulate_half(packed_word_20, packed_word_21, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_154, packed_word_155,
                             packed_word_20, packed_word_21);
    packed_word_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x3c203c20u);
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_14),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_6, packed_word_7,
                             packed_word_22, packed_word_23);
    packed_word_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_16),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_18),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_78), 0x3e473e47u);
    packed_word_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x3e473e47u);
    packed_word_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x3c203c20u);
    packed_word_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x3e473e47u);
    packed_word_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x3c203c20u);
    packed_word_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3c203c20u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_8, packed_word_9,
                             packed_word_40, packed_word_41);
    packed_word_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3c203c20u);
    packed_word_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0x3c203c20u);
    packed_word_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0x3c203c20u);
    packed_word_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x3e473e47u);
    packed_word_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x3e473e47u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_76, packed_word_77,
                             packed_word_42, packed_word_43);
    packed_word_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3e473e47u);
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_79),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3e473e47u);
    packed_word_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x3e473e47u);
    packed_word_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_75),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0x3e473e47u);
    packed_word_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0x3e473e47u);
    packed_word_45 = (packed_word_74 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_8, packed_word_9,
                             packed_word_24, packed_word_25);
    packed_word_44 = (packed_word_32 << 0x5u) + 0x7ff88000u;
    packed_word_47 = (packed_word_47 << 0x5u) + 0x7ff88000u;
    packed_word_8 = (packed_word_78 << 0x5u) + 0x7ff88000u;
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_47));
    packed_word_46 = (packed_word_33 << 0x5u) + 0x7ff88000u;
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_36),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_59 = (packed_word_35 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_28, packed_word_29,
                             packed_word_30, packed_word_31, packed_word_76, packed_word_77,
                             packed_word_26, packed_word_27);
    packed_word_0 = (packed_word_12 << 0x5u) + 0x7ff88000u;
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_45));
    packed_word_33 = (packed_word_14 << 0x5u) + 0x7ff88000u;
    packed_word_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_37),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_32 = (packed_word_16 << 0x5u) + 0x7ff88000u;
    packed_word_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_39),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_35 = (packed_word_18 << 0x5u) + 0x7ff88000u;
    packed_word_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_13),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_79), 0x3c203c20u);
    packed_word_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_15),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x3c203c20u);
    packed_word_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_17),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_19),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_20),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_22),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_79), 0x3e473e47u);
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_6),
                                   select_half_operand<2, false, false>(packed_word_7));
    packed_word_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x3e473e47u);
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_33));
    packed_word_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x3c203c20u);
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                    select_half_operand<2, false, false>(packed_word_35));
    packed_word_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x3c203c20u);
    packed_word_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_38),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x3c203c20u);
    packed_word_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3c203c20u);
    packed_word_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3c203c20u);
    packed_word_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0x3c203c20u);
    packed_word_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_19), 0x3c203c20u);
    packed_word_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0x3c203c20u);
    packed_word_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_22), 0x3c203c20u);
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_6),
                                    select_half_operand<2, false, false>(packed_word_16));
    packed_word_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x3e473e47u);
    packed_word_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x3e473e47u);
    packed_word_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x3e473e47u);
    packed_word_9 = (packed_word_79 << 0x5u) + 0x7ff88000u;
    packed_word_34 = (packed_word_75 << 0x5u) + 0x7ff88000u;
    packed_word_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3e473e47u);
    packed_word_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3e473e47u);
    packed_word_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0x3e473e47u);
    packed_word_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_19), 0x3e473e47u);
    packed_word_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0x3e473e47u);
    packed_word_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_22), 0x3e473e47u);
    packed_word_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x3c203c20u);
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                    select_half_operand<2, false, false>(packed_word_34));
    packed_word_22 = (packed_word_36 << 0x5u) + 0x7ff88000u;
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                    select_half_operand<2, false, false>(packed_word_59));
    packed_word_20 = (packed_word_37 << 0x5u) + 0x7ff88000u;
    packed_word_29 = (packed_word_39 << 0x5u) + 0x7ff88000u;
    packed_word_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_21),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x3e473e47u);
    packed_word_30 = (packed_word_13 << 0x5u) + 0x7ff88000u;
    packed_word_39 = (packed_word_15 << 0x5u) + 0x7ff88000u;
    packed_word_36 = (packed_word_17 << 0x5u) + 0x7ff88000u;
    packed_word_65 = (packed_word_19 << 0x5u) + 0x7ff88000u;
    packed_word_28 = (packed_word_28 << 0x5u) + 0x7ff88000u;
    packed_word_37 = (packed_word_6 << 0x5u) + 0x7ff88000u;
    packed_word_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_23),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_31 = (packed_word_38 << 0x5u) + 0x7ff88000u;
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_14));
    packed_word_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_21), 0x3c203c20u);
    packed_word_13 = add_half_pairs(select_half_operand<2, false, false>(packed_word_20),
                                    select_half_operand<2, false, false>(packed_word_29));
    packed_word_15 = add_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                    select_half_operand<2, false, false>(packed_word_39));
    packed_word_19 = add_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                    select_half_operand<2, false, false>(packed_word_65));
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                    select_half_operand<2, false, false>(packed_word_37));
    packed_word_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_23 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_23), 0x3c203c20u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_43),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_41),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_24),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_25),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_26),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_27),
        select_half_operand<1, false, false>(packed_word_4), 0x3d343d34u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x3c203c20u);
    packed_word_58 = add_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                    select_half_operand<2, false, false>(packed_word_19));
    packed_word_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_21), 0x3e473e47u);
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                   select_half_operand<2, false, false>(packed_word_31));
    packed_word_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x3c203c20u);
    packed_word_18 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_13));
    packed_word_23 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_23), 0x3e473e47u);
    packed_word_19 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                    select_half_operand<2, false, false>(packed_word_14));
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x3c203c20u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x4ce0ull, packed_word_12, packed_word_13,
        packed_word_14, packed_word_15, memory_ranges, error_counts);
    packed_word_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x3c203c20u);
    packed_word_17 = add_half_pairs(select_half_operand<2, false, false>(packed_word_7),
                                    select_half_operand<2, false, false>(packed_word_6));
    packed_word_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x3c203c20u);
    packed_word_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0x3c203c20u);
    packed_word_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0x3c203c20u);
    packed_word_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0x3c203c20u);
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x3e473e47u);
    packed_word_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x3e473e47u);
    packed_word_4 = (packed_word_21 << 0x5u) + 0x7ff88000u;
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x3e473e47u);
    packed_word_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x3e473e47u);
    packed_word_21 = (packed_word_23 << 0x5u) + 0x7ff88000u;
    packed_word_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x3e473e47u);
    packed_word_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0x3e473e47u);
    packed_word_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0x3e473e47u);
    packed_word_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0x3e473e47u);
    packed_word_43 = (packed_word_42 << 0x5u) + 0x7ff88000u;
    packed_word_42 = (packed_word_6 << 0x5u) + 0x7ff88000u;
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_4),
                                   select_half_operand<2, false, false>(packed_word_21));
    packed_word_26 = (packed_word_40 << 0x5u) + 0x7ff88000u;
    packed_word_27 = (packed_word_41 << 0x5u) + 0x7ff88000u;
    packed_word_23 = (packed_word_24 << 0x5u) + 0x7ff88000u;
    packed_word_38 = (packed_word_25 << 0x5u) + 0x7ff88000u;
    packed_word_40 = (packed_word_7 << 0x5u) + 0x7ff88000u;
    packed_word_41 = (packed_word_16 << 0x5u) + 0x7ff88000u;
    packed_word_58 = add_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                    select_half_operand<2, false, false>(packed_word_6));
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                   select_half_operand<2, false, false>(packed_word_42));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                    select_half_operand<2, false, false>(packed_word_40));
    packed_word_24 = add_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                    select_half_operand<2, false, false>(packed_word_41));
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                   select_half_operand<2, false, false>(packed_word_6));
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                   select_half_operand<2, false, false>(packed_word_7));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_19),
                                    select_half_operand<2, false, false>(packed_word_16));
    packed_word_17 = add_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                    select_half_operand<2, false, false>(packed_word_24));
    packed_word_18 = condition_4 ? packed_word_7 : packed_word_6;
    packed_word_6 = condition_4 ? packed_word_6 : packed_word_7;
    packed_word_19 = condition_4 ? packed_word_17 : packed_word_16;
    packed_word_17 = condition_4 ? packed_word_16 : packed_word_17;
    packed_word_58 = !condition_5 ? packed_word_18 : packed_word_19;
    packed_word_67 = !condition_5 ? packed_word_19 : packed_word_18;
    packed_word_64 = !condition_5 ? packed_word_6 : packed_word_17;
    packed_word_66 = !condition_5 ? packed_word_17 : packed_word_6;
    load_two_boundary_words<CheckBounds>(combine_words(packed_word_10, packed_word_11) + 0x5130ull,
                                         packed_word_6, packed_word_7, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_10, packed_word_11) + 0x4ee0ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_58), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_58 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_64), "r"(packed_word_161), "r"(0x1fu), "r"(__activemask()));
        packed_word_25 = loaded_word;
    }
    packed_word_68 = !condition_6 ? packed_word_58 : packed_word_25;
    packed_word_69 = !condition_6 ? packed_word_25 : packed_word_58;
    load_two_boundary_words<CheckBounds>(combine_words(packed_word_10, packed_word_11) + 0x5330ull,
                                         packed_word_24, packed_word_25, memory_ranges,
                                         error_counts);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_67), "r"(packed_word_160), "r"(0x1fu), "r"(__activemask()));
        packed_word_67 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_66), "r"(packed_word_159), "r"(0x1fu), "r"(__activemask()));
        packed_word_66 = loaded_word;
    }
    packed_word_73 = !condition_6 ? packed_word_67 : packed_word_66;
    packed_word_74 = !condition_6 ? packed_word_66 : packed_word_67;
    packed_word_75 = !condition_3 ? packed_word_68 : packed_word_73;
    packed_word_76 = !condition_3 ? packed_word_69 : packed_word_74;
    packed_word_68 = !condition_3 ? packed_word_73 : packed_word_68;
    packed_word_69 = !condition_3 ? packed_word_74 : packed_word_69;
    packed_word_75 = add_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                    select_half_operand<2, false, false>(packed_word_76));
    packed_word_75 = add_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                    select_half_operand<2, false, false>(packed_word_68));
    packed_word_75 = add_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                    select_half_operand<2, false, false>(packed_word_69));
    packed_word_75 = add_half_pairs(select_half_operand<0, false, false>(packed_word_75),
                                    select_half_operand<1, false, false>(packed_word_75));
    packed_word_75 = permute_signed(packed_word_75, packed_word_75, 0x5410u);
    packed_word_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x4100410u);
    packed_word_64 = packed_word_162 + 0x8u;
    packed_word_10 = half_to_float_bits((packed_word_75 & 65535u) * 65537u);
    packed_word_75 = half_to_float_bits((packed_word_75 >> 16) * 65537u);
    packed_word_11 = uint32_t(int32_t(packed_word_64) >> 0x1fu);
    packed_word_10 = reciprocal_bits(packed_word_10);
    packed_word_11 = uint32_t(combine_words(packed_word_11, 0u) << 0x5u >> 32) + packed_word_64;
    packed_word_75 = reciprocal_bits(packed_word_75);
    packed_word_67 = packed_word_11 & 0xffffffe0u;
    packed_word_66 = packed_word_64 + (0u - packed_word_67);
    packed_word_64 = uint32_t(int64_t(int32_t(packed_word_165)) * (1ll << 0x1eu) >> 32) + 0x10u;
    packed_word_165 = uint32_t(int64_t(int32_t(packed_word_165)) * (1ll << 0x1eu) >> 32) + 0x18u;
    packed_word_58 = packed_word_162 + (0u - packed_word_72);
    packed_word_67 = uint32_t(int32_t(packed_word_64) >> 0x1fu);
    packed_word_68 = uint32_t(int32_t(packed_word_165) >> 0x1fu);
    packed_word_11 = pack_half(packed_word_75, packed_word_10);
    packed_word_67 = uint32_t(combine_words(packed_word_67, 0u) << 0x5u >> 32) + packed_word_64;
    packed_word_68 = uint32_t(combine_words(packed_word_68, 0u) << 0x5u >> 32) + packed_word_165;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_11), "r"(packed_word_66), "r"(0x1fu), "r"(__activemask()));
        packed_word_10 = loaded_word;
    }
    packed_word_67 = packed_word_67 & 0xffffffe0u;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_11), "r"(packed_word_58), "r"(0x1fu), "r"(__activemask()));
        packed_word_58 = loaded_word;
    }
    packed_word_68 = packed_word_68 & 0xffffffe0u;
    packed_word_72 = packed_word_64 + (0u - packed_word_67);
    packed_word_74 = packed_word_165 + (0u - packed_word_68);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_11), "r"(packed_word_72), "r"(0x1fu), "r"(__activemask()));
        packed_word_66 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_11), "r"(packed_word_74), "r"(0x1fu), "r"(__activemask()));
        packed_word_69 = loaded_word;
    }
    packed_word_68 = permute_signed(packed_word_10, packed_word_10, 0x5410u);
    packed_word_64 = permute_signed(packed_word_58, packed_word_58, 0x5410u);
    packed_word_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                   select_half_operand<2, false, false>(0u));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                        select_half_operand<2, false, false>(packed_word_68));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(0u));
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                        select_half_operand<2, false, false>(packed_word_64));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(0u));
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_59),
                                   select_half_operand<2, false, false>(0u));
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                  select_half_operand<2, false, false>(packed_word_34));
    packed_word_34 = permute_signed(packed_word_66, packed_word_66, 0x5410u);
    packed_word_69 = permute_signed(packed_word_69, packed_word_69, 0x5410u);
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_8),
                                  select_half_operand<2, false, false>(packed_word_45));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_47));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_33 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_69));
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                         select_half_operand<2, false, false>(packed_word_69));
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_86, packed_word_87, 0u,
                             0u);
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                        select_half_operand<2, false, false>(packed_word_34));
    packed_word_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_33),
                                   select_half_operand<2, false, false>(0u));
    packed_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_64));
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_94, packed_word_95, 0u,
                             0u);
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_68));
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_96, packed_word_97, 0u,
                             0u);
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(0u));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                        select_half_operand<2, false, false>(packed_word_69));
    packed_word_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                   select_half_operand<2, false, false>(0u));
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_69));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_0),
                                  select_half_operand<2, false, false>(packed_word_33));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                  select_half_operand<2, false, false>(packed_word_30));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_35));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_32));
    packed_word_22 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                         select_half_operand<2, false, false>(packed_word_64));
    packed_word_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                   select_half_operand<2, false, false>(0u));
    packed_word_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_64));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                  select_half_operand<2, false, false>(0u));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(0u));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_96, packed_word_97, 0u,
                             0u);
    packed_word_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_23 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                         select_half_operand<2, false, false>(packed_word_34));
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_86, packed_word_87, 0u,
                             0u);
    packed_word_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_21 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_21),
                                         select_half_operand<2, false, false>(packed_word_69));
    packed_word_41 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_69));
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_94, packed_word_95, 0u,
                             0u);
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_20),
                                        select_half_operand<2, false, false>(packed_word_68));
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_22),
                                  select_half_operand<2, false, false>(packed_word_31));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                  select_half_operand<2, false, false>(packed_word_0));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_26),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_42));
    packed_word_4 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_4),
                                        select_half_operand<2, false, false>(packed_word_69));
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_37),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_41),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_56, packed_word_57,
                             packed_word_58, packed_word_59);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_70, packed_word_71,
                             packed_word_46, packed_word_47);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_88, packed_word_89,
                             packed_word_44, packed_word_45);
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_90, packed_word_91,
                             packed_word_66, packed_word_67);
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_69));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_21),
                                  select_half_operand<2, false, false>(0u));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                   select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_28),
                                  select_half_operand<2, false, false>(packed_word_37));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_4),
                                  select_half_operand<2, false, false>(packed_word_9));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_23),
                                   select_half_operand<2, false, false>(packed_word_10));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_70, packed_word_71,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_90, packed_word_91,
                             packed_word_96, packed_word_97);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_56, packed_word_57,
                             packed_word_86, packed_word_87);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_88, packed_word_89,
                             packed_word_94, packed_word_95);
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                  select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_67),
                                  select_half_operand<2, false, false>(0u));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                  select_half_operand<2, false, false>(0u));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_66),
                                   select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_58),
                                  select_half_operand<2, false, false>(packed_word_9));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_59),
                                  select_half_operand<2, false, false>(packed_word_0));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_4));
    packed_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                  select_half_operand<2, false, false>(0u));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                  select_half_operand<2, false, false>(0u));
    packed_word_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(packed_word_21));
    packed_word_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(packed_word_23));
    packed_word_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(packed_word_4));
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_12, packed_word_13,
                             packed_word_48, packed_word_49);
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_12, packed_word_13,
                             packed_word_50, packed_word_51);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_14, packed_word_15,
                             packed_word_60, packed_word_61);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_14, packed_word_15,
                             packed_word_62, packed_word_63);
    image_mma_half(packed_word_48, packed_word_49, packed_word_48, packed_word_49, packed_word_50,
                   packed_word_51, packed_word_6, packed_word_7, 0u, 0u);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_16, packed_word_17,
                             packed_word_104, packed_word_105);
    image_mma_half(packed_word_12, packed_word_13, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_6, packed_word_7, 0u, 0u);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_18, packed_word_19,
                             packed_word_106, packed_word_107);
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_16, packed_word_17,
                             packed_word_108, packed_word_109);
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_18, packed_word_19,
                             packed_word_110, packed_word_111);
    image_mma_half(packed_word_48, packed_word_49, packed_word_104, packed_word_105,
                   packed_word_106, packed_word_107, packed_word_24, packed_word_25, packed_word_48,
                   packed_word_49);
    packed_word_53 = packed_word_53 + 0x4u;
    image_mma_half(packed_word_12, packed_word_13, packed_word_108, packed_word_109,
                   packed_word_110, packed_word_111, packed_word_24, packed_word_25, packed_word_12,
                   packed_word_13);
    condition_0 = int32_t(packed_word_53) >= int32_t(uniform_word_8) || condition_0;
    packed_word_4 = packed_word_52 | packed_word_53;
    packed_word_0 = condition_4 ? packed_word_49 : packed_word_48;
    packed_word_48 = condition_4 ? packed_word_48 : packed_word_49;
    condition_0 = int32_t(packed_word_4) < int32_t(0u) || condition_0;
    packed_word_7 = condition_4 ? packed_word_13 : packed_word_12;
    packed_word_13 = condition_4 ? packed_word_12 : packed_word_13;
    packed_word_4 = !condition_5 ? packed_word_0 : packed_word_7;
    packed_word_6 = !condition_5 ? packed_word_48 : packed_word_13;
    packed_word_7 = !condition_5 ? packed_word_7 : packed_word_0;
    packed_word_48 = !condition_5 ? packed_word_13 : packed_word_48;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_4), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_4 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_6), "r"(packed_word_161), "r"(0x1fu), "r"(__activemask()));
        packed_word_161 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_7), "r"(packed_word_160), "r"(0x1fu), "r"(__activemask()));
        packed_word_7 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_48), "r"(packed_word_159), "r"(0x1fu), "r"(__activemask()));
        packed_word_48 = loaded_word;
    }
    if (condition_0) {
        return;
    }
    uniform_word_8 = parameters[14];
    uniform_word_9 = parameters[15];
    condition_0 = int32_t(packed_word_53) >= int32_t(uniform_word_7);
    packed_word_10 = parameters[16];
    packed_word_11 = parameters[17];
    packed_word_3 = __float_as_uint(__uint2float_rn(packed_word_52));
    packed_word_0 = __float_as_uint(__uint2float_rn(packed_word_53));
    condition_0 = int32_t(packed_word_52) < int32_t(uniform_word_6) && !condition_0;
    asm("add.rn.ftz.f32 %0,%1,%2;"
        : "=f"(reinterpret_cast<float&>(packed_word_6))
        : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(0x3f000000u)));
    packed_word_8 = parameters[18];
    packed_word_9 = parameters[19];
    asm("add.rn.ftz.f32 %0,%1,%2;"
        : "=f"(reinterpret_cast<float&>(packed_word_0))
        : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(0x3f000000u)));
    asm("mul.rn.ftz.f32 %0,%1,%2;"
        : "=f"(reinterpret_cast<float&>(packed_word_15))
        : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_6)));
    asm("mul.rn.ftz.f32 %0,%1,%2;"
        : "=f"(reinterpret_cast<float&>(packed_word_0))
        : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_0)));
    condition_1 = 0ull != combine_words(uniform_word_8, uniform_word_9) && condition_0;
    if (condition_1) {
        packed_word_12 = parameters[20];
        packed_word_13 = parameters[21];
    }
    uniform_condition_0 = vote_warp_predicate<false>(condition_1);
    if (condition_1) {
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_3))
            : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(packed_word_8)),
              "f"(__uint_as_float(packed_word_10)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_9)),
              "f"(__uint_as_float(packed_word_11)));
    }
    if (uniform_condition_0) {
        uniform_word_6 = parameters[14];
        uniform_word_7 = parameters[15];
    }
    if (condition_1) {
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(packed_word_12)));

        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(packed_word_13)));

        {
            {
                uint32_t sampled_channels[4];
                image_tex(sampled_channels, combine_words(uniform_word_6, uniform_word_7),
                          packed_word_8, packed_word_9);
                packed_word_8 = sampled_channels[0];
                packed_word_9 = sampled_channels[1];
                packed_word_3 = sampled_channels[2];
            }
        }
    }
    packed_word_6 = !condition_6 ? packed_word_4 : packed_word_161;
    packed_word_161 = !condition_6 ? packed_word_161 : packed_word_4;
    if (condition_3) {
        packed_word_6 = !condition_6 ? packed_word_7 : packed_word_48;

        packed_word_161 = !condition_6 ? packed_word_48 : packed_word_7;
    }
    if (condition_1) {
        packed_word_10 = 0x3e000000u;
    }
    packed_word_11 = half_to_float_bits((packed_word_6 >> 16) * 65537u);
    packed_word_7 = half_to_float_bits((packed_word_6 & 65535u) * 65537u);
    packed_word_12 = half_to_float_bits((packed_word_161 & 65535u) * 65537u);
    uniform_word_7 = parameters[13];
    uniform_word_6 = parameters[12];
    uniform_condition_0 = uint32_t(uniform_word_7) != 0u;
    if (condition_1) {
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_9)), "f"(__uint_as_float(packed_word_10)),
              "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_8)), "f"(__uint_as_float(packed_word_10)),
              "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_3))
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(packed_word_10)),
              "f"(__uint_as_float((0x3d800000u ^ 0x80000000u))));
    }
    if (!condition_1) {
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(packed_word_11)), "f"(__uint_as_float(uniform_word_6)));

        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(uniform_word_6)));

        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_10))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(uniform_word_6)));
    }
    if (condition_1) {
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(packed_word_11)), "f"(__uint_as_float(uniform_word_6)),
              "f"(__uint_as_float(packed_word_6)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(uniform_word_6)),
              "f"(__uint_as_float(packed_word_4)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_10))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(uniform_word_6)),
              "f"(__uint_as_float(packed_word_3)));
    }
    packed_word_11 = 0u;
    if (uniform_condition_0) {
        packed_word_3 = 0x41000000u;
        packed_word_11 = 0x3f800000u;
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(packed_word_8)), "f"(__uint_as_float(packed_word_3)),
              "f"(__uint_as_float(0x3f000000u)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(packed_word_9)), "f"(__uint_as_float(packed_word_3)),
              "f"(__uint_as_float(0x3f000000u)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_10))
            : "f"(__uint_as_float(packed_word_10)), "f"(__uint_as_float(packed_word_3)),
              "f"(__uint_as_float(0x3f000000u)));
        asm("add.rn.ftz.sat.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_8)));
        asm("add.rn.ftz.sat.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_9)));
        asm("add.rn.ftz.sat.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_10))
            : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_10)));
    }
    uniform_word_8 = parameters[26];
    uniform_word_9 = parameters[27];
    packed_word_3 = 0x3f800000u;
    uniform_condition_0 = combine_words(uniform_word_8, uniform_word_9) != 0ull;
    if (uniform_condition_0) {
        packed_word_6 = parameters[26];
        packed_word_7 = parameters[27];
        packed_word_6 = read_boundary_half<CheckBounds>(combine_words(packed_word_6, packed_word_7),
                                                        memory_ranges, error_counts);
        packed_word_3 = 0u;
        packed_word_4 = half_to_float_bits((packed_word_6 & 65535u) * 65537u);
        {
            uint32_t loaded_word;
            asm("{.reg .pred p; setp.ne.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                : "=r"(loaded_word)
                : "f"(__uint_as_float((packed_word_4 & 0x7fffffffu))),
                  "f"(__uint_as_float(0x7f800000u)));
            condition_1 = bool(loaded_word);
        }
        if (condition_1) {
            asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_3))
                : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_4)));
        }
    }
    uniform_word_4 = parameters[22];
    uniform_word_5 = parameters[23];
    {
        uint32_t loaded_word;
        asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
            : "=r"(loaded_word)
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(0u)));
        condition_1 = bool(loaded_word);
    }
    packed_word_161 = half_to_float_bits((packed_word_161 >> 16) * 65537u);
    uniform_word_8 = parameters[24];
    uniform_word_9 = parameters[25];
    uniform_condition_0 = combine_words(uniform_word_4, uniform_word_5) != 0ull;
    uniform_condition_0 = uint32_t(uniform_word_7) != 0u && uniform_condition_0;
    condition_0 = 0ull != combine_words(uniform_word_8, uniform_word_9) && condition_0;
    condition_2 = uniform_condition_0;
    condition_0 = condition_0 & condition_1 & condition_2;
    if (condition_0) {
        packed_word_16 = parameters[36];
        packed_word_17 = parameters[37];
        uniform_word_8 = parameters[32];
        uniform_word_9 = parameters[33];
        uniform_word_10 = parameters[34];
        uniform_word_11 = parameters[35];
        uniform_word_6 = parameters[40];
        uniform_word_7 = parameters[41];
        packed_word_12 = parameters[38];
        packed_word_13 = parameters[39];
        uniform_word_4 = parameters[24];
        uniform_word_5 = parameters[25];
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(packed_word_17)),
              "f"(__uint_as_float(uniform_word_11)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_12)),
              "f"(__uint_as_float(packed_word_16)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(packed_word_13)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_13))
            : "f"(__uint_as_float(packed_word_4)), "f"(__uint_as_float(uniform_word_6)));
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_12, packed_word_13);
            packed_word_6 = sampled_channels[0];
            packed_word_7 = sampled_channels[1];
        }
        packed_word_22 = parameters[28];
        packed_word_23 = parameters[29];
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_20))
            : "f"(__uint_as_float(packed_word_2)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_25))
            : "f"(__uint_as_float(packed_word_54)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        packed_word_26 = parameters[30];
        packed_word_27 = parameters[31];
        uniform_word_4 = parameters[42];
        condition_0 = uint32_t(packed_word_22) != 0u;
        if (condition_0) {
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_15))
                : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(uniform_word_7)),
                  "f"(__uint_as_float(packed_word_15)));

            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_0))
                : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(uniform_word_4)),
                  "f"(__uint_as_float(packed_word_0)));
        }
        uniform_word_4 = parameters[22];
        uniform_word_5 = parameters[23];
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(packed_word_15)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(packed_word_0)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        asm("cvt.rmi.ftz.f32.f32 %0,%1;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_4)));
        asm("cvt.rmi.ftz.f32.f32 %0,%1;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_6)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_7))
            : "f"(__uint_as_float(packed_word_4)), "f"(__uint_as_float(0x3f000000u)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(packed_word_15)),
              "f"(__uint_as_float((packed_word_7 ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_13))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(0x3f000000u)));
        asm("add.rn.ftz.sat.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_12)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(packed_word_0)),
              "f"(__uint_as_float((packed_word_13 ^ 0x80000000u))));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_12)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_2))
            : "f"(__uint_as_float(packed_word_13)),
              "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
        asm("add.rn.ftz.sat.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_0)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_15)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(0x40200000u)));
        packed_word_2 = float_minmax(packed_word_2, 0x3f000000u, !true);
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_0)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_4)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_4)), "f"(__uint_as_float(0x3fc00000u)),
              "f"(__uint_as_float((packed_word_19 ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float((packed_word_15 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_4)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_17)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_21))
            : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(0x40200000u)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_12)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_15)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(0x3f800000u)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_6)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_21))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(0x3fc00000u)),
              "f"(__uint_as_float((packed_word_21 ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float((packed_word_17 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_6)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float((packed_word_12 ^ 0x80000000u))),
              "f"(__uint_as_float(0x3f800000u)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_0)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_17)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_4)), "f"(__uint_as_float(0x3f000000u)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_21))
            : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(0x3f800000u)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float((packed_word_19 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_14)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float((packed_word_17 ^ 0x80000000u))),
              "f"(__uint_as_float(0x3f800000u)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(0x3f000000u)));
        packed_word_2 = float_minmax(packed_word_2, packed_word_25, true);
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float((packed_word_4 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_15)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float((packed_word_21 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_14)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_0)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float((packed_word_6 ^ 0x80000000u))),
              "f"(__uint_as_float(packed_word_15)));
        packed_word_15 = reciprocal_bits(packed_word_19);
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_21))
            : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(packed_word_14)));
        packed_word_16 = reciprocal_bits(packed_word_21);
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_15)),
              "f"(__uint_as_float(packed_word_7)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_7)),
              "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_7))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(0x40000000u)));
        packed_word_15 = float_minmax(packed_word_15, 0x3f000000u, !true);
        packed_word_0 = float_minmax(packed_word_0, 0x3f000000u, !true);
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(packed_word_16)),
              "f"(__uint_as_float(packed_word_13)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_16))
            : "f"(__uint_as_float(packed_word_13)), "f"(__uint_as_float(0x40000000u)));
        packed_word_13 = float_minmax(packed_word_7, 0x3f000000u, !true);
        packed_word_0 = float_minmax(packed_word_0, packed_word_20, true);
        packed_word_7 = float_minmax(packed_word_14, 0x3f000000u, !true);
        packed_word_16 = float_minmax(packed_word_16, 0x3f000000u, !true);
        packed_word_18 = float_minmax(packed_word_13, packed_word_20, true);
        packed_word_14 = float_minmax(packed_word_15, packed_word_20, true);
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_0)));
        packed_word_20 = float_minmax(packed_word_16, packed_word_25, true);
        packed_word_16 = float_minmax(packed_word_7, packed_word_25, true);
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_18)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_14)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_5))
            : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_2)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_7))
            : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_20)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_55))
            : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_16)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_2))
            : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(packed_word_27)),
              "f"(__uint_as_float(packed_word_23)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_5))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(uniform_word_8)),
              "f"(__uint_as_float(packed_word_26)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_27)),
              "f"(__uint_as_float(packed_word_23)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_27)),
              "f"(__uint_as_float(packed_word_23)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(uniform_word_8)),
              "f"(__uint_as_float(packed_word_26)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_22))
            : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(uniform_word_9)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_23))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(uniform_word_10)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_7))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(uniform_word_8)),
              "f"(__uint_as_float(packed_word_26)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_15))
            : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(uniform_word_10)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_14))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(uniform_word_9)));
        packed_word_26 = packed_word_22;
        packed_word_28 = packed_word_22;
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_29))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(uniform_word_10)));
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_22, packed_word_23);
            packed_word_22 = sampled_channels[0];
            packed_word_23 = sampled_channels[1];
            packed_word_5 = sampled_channels[2];
        }
        packed_word_27 = packed_word_15;
        packed_word_25 = packed_word_15;
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_24))
            : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(uniform_word_9)));
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_14, packed_word_15);
            packed_word_14 = sampled_channels[0];
            packed_word_15 = sampled_channels[1];
            packed_word_2 = sampled_channels[2];
        }
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_26, packed_word_27);
            packed_word_26 = sampled_channels[0];
            packed_word_27 = sampled_channels[1];
            packed_word_7 = sampled_channels[2];
        }
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_28, packed_word_29);
            packed_word_28 = sampled_channels[0];
            packed_word_29 = sampled_channels[1];
            packed_word_13 = sampled_channels[2];
        }
        {
            uint32_t sampled_channels[4];
            image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                      packed_word_24, packed_word_25);
            packed_word_24 = sampled_channels[0];
            packed_word_25 = sampled_channels[1];
            packed_word_16 = sampled_channels[2];
        }
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_20))
            : "f"(__uint_as_float(packed_word_161)),
              "f"(__uint_as_float((0x3fb8aa3bu ^ 0x80000000u))));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_31))
            : "f"(__uint_as_float(packed_word_12)), "f"(__uint_as_float(packed_word_21)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_12))
            : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(packed_word_19)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_21)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_6))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(packed_word_19)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_21))
            : "f"(__uint_as_float(packed_word_4)), "f"(__uint_as_float(packed_word_21)));
        packed_word_20 = image_mufu_ex2(packed_word_20);
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_31)), "f"(__uint_as_float(packed_word_12)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(packed_word_17)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(packed_word_17)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_4))
            : "f"(__uint_as_float(packed_word_21)), "f"(__uint_as_float(packed_word_4)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_20)), "f"(__uint_as_float(0x3f800000u)));
        packed_word_0 = reciprocal_bits(packed_word_0);
        asm("fma.rn.ftz.sat.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_19))
            : "f"(__uint_as_float(packed_word_0)), "f"(__uint_as_float(packed_word_3)),
              "f"(__uint_as_float(0u)));
        packed_word_3 = reciprocal_bits(packed_word_4);

        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(packed_word_12)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(packed_word_12)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_5))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(packed_word_12)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(packed_word_31)),
              "f"(__uint_as_float(packed_word_17)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(packed_word_31)),
              "f"(__uint_as_float(packed_word_0)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_2))
            : "f"(__uint_as_float(packed_word_2)), "f"(__uint_as_float(packed_word_31)),
              "f"(__uint_as_float(packed_word_5)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_26)), "f"(__uint_as_float(packed_word_18)),
              "f"(__uint_as_float(packed_word_17)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_27)), "f"(__uint_as_float(packed_word_18)),
              "f"(__uint_as_float(packed_word_0)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_2))
            : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(packed_word_18)),
              "f"(__uint_as_float(packed_word_2)));

        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_6)),
              "f"(__uint_as_float(packed_word_17)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_29)), "f"(__uint_as_float(packed_word_6)),
              "f"(__uint_as_float(packed_word_0)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_13))
            : "f"(__uint_as_float(packed_word_13)), "f"(__uint_as_float(packed_word_6)),
              "f"(__uint_as_float(packed_word_2)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_24)), "f"(__uint_as_float(packed_word_21)),
              "f"(__uint_as_float(packed_word_17)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_25)), "f"(__uint_as_float(packed_word_21)),
              "f"(__uint_as_float(packed_word_0)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_13))
            : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(packed_word_21)),
              "f"(__uint_as_float(packed_word_13)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_17))
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(packed_word_17)),
              "f"(__uint_as_float((packed_word_8 ^ 0x80000000u))));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_0))
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(packed_word_0)),
              "f"(__uint_as_float((packed_word_9 ^ 0x80000000u))));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_13))
            : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(packed_word_13)),
              "f"(__uint_as_float((packed_word_10 ^ 0x80000000u))));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_8))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_17)),
              "f"(__uint_as_float(packed_word_8)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_9))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_0)),
              "f"(__uint_as_float(packed_word_9)));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_10))
            : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_13)),
              "f"(__uint_as_float(packed_word_10)));
    }

    uniform_word_4 = parameters[4];
    image_surface(uniform_word_4, packed_word_52, packed_word_53, packed_word_8, packed_word_9,
                  packed_word_10, packed_word_11);
    return;
}

extern "C" __global__
__launch_bounds__(32) void phoenix_output_reconstruction(KernelArguments arguments) {

    run_output_reconstruction<false>(arguments.words, nullptr, nullptr);
}
