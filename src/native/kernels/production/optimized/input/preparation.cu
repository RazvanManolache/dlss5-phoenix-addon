// DLSS5 Phoenix input/preparation: native frame-boundary CUDA source.
// Sample frame inputs, condition features and publish the packed neural input tile.
// Word slots are reused across phases; changing their lifetime or arithmetic order can change GPU code.
// The 64-bit texture handles contain low-word texture and high-word sampler indices.
// Compile to PTX, select independent texture mode, then assemble with a 128-register limit.

#define DLSS5_D3D12_SAMPLER
#include "../../common/texture_sampling.cuh"
#include "../../common/frame_boundary_memory.cuh"
#include "../../common/boundary_clear.cuh"

// The by-value parameter block matches the runtime's existing binary launch interface.
struct alignas(8) KernelArguments {
    uint32_t words[66];
};
static_assert(sizeof(KernelArguments) == 264, "Frame-boundary argument layout");

template <bool CheckBounds>
__device__ __forceinline__ void run_input_preparation(const uint32_t* parameters,
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
    uint32_t uniform_word_16 = 0;
    uint32_t uniform_word_17 = 0;
    uint32_t uniform_word_18 = 0;
    uint32_t uniform_word_19 = 0;
    uint32_t uniform_word_20 = 0;
    uint32_t uniform_word_21 = 0;
    uint32_t uniform_word_22 = 0;
    uint32_t uniform_word_23 = 0;
    uint32_t uniform_word_24 = 0;
    uint32_t uniform_word_25 = 0;
    uint32_t uniform_word_26 = 0;
    uint32_t uniform_word_27 = 0;
    uint32_t uniform_word_28 = 0;
    uint32_t uniform_word_29 = 0;
    uint32_t uniform_word_30 = 0;
    uint32_t uniform_word_31 = 0;
    uint32_t uniform_word_32 = 0;
    uint32_t uniform_word_33 = 0;
    uint32_t uniform_word_34 = 0;
    uint32_t uniform_word_35 = 0;
    uint32_t uniform_word_36 = 0;
    uint32_t uniform_word_37 = 0;
    uint32_t uniform_word_38 = 0;
    uint32_t uniform_word_39 = 0;
    uint32_t uniform_word_40 = 0;
    uint32_t angle_bits_0 = 0;
    uint32_t angle_bits_1 = 0;
    uint64_t copy_tokens[2] = {0, 0};
    bool condition_0 = false, condition_1 = false, condition_2 = false, condition_3 = false,
         condition_4 = false, condition_5 = false, condition_6 = false, uniform_condition_0 = false,
         uniform_condition_1 = false, uniform_condition_2 = false, uniform_condition_3 = false,
         uniform_condition_4 = false, uniform_condition_5 = false, uniform_condition_6 = false;
    constexpr uint32_t shared_bytes = 3072;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];
    packed_word_1 = 0u;
    packed_word_29 = threadIdx.y;
    packed_word_0 = blockDim.y;
    uniform_word_10 = 0u;
    uniform_word_11 = 0u;

    packed_word_2 = threadIdx.x;
    uniform_word_8 = parameters[8];
    uniform_word_9 = parameters[9];
    uniform_word_36 = parameters[37];
    uniform_word_37 = parameters[39];
    uniform_word_20 = parameters[6];
    uniform_word_21 = parameters[7];
    uniform_word_38 = parameters[42];
    uniform_word_39 = parameters[43];
    uniform_word_40 = parameters[48];
    uniform_word_22 = parameters[18];
    uniform_word_23 = parameters[19];
    packed_word_29 = packed_word_29 * 0x20u + packed_word_2;
    uniform_word_24 = parameters[20];
    uniform_word_25 = parameters[21];
    condition_0 = uint32_t(packed_word_29) > uint32_t(0x3fu);
    uniform_word_26 = parameters[40];
    uniform_word_27 = parameters[41];
    uniform_word_12 = parameters[24];
    uniform_word_13 = parameters[25];
    uniform_word_14 = parameters[26];
    uniform_word_15 = parameters[27];
    uniform_word_16 = parameters[12];
    uniform_word_17 = parameters[13];
    uniform_word_18 = parameters[14];
    uniform_word_19 = parameters[15];
    if (!condition_0) {
        packed_word_5 = blockIdx.x;
        packed_word_6 = parameters[52];
        packed_word_7 = parameters[53];
        uniform_word_6 = parameters[24];
        uniform_word_7 = parameters[25];
        packed_word_2 = packed_word_2 & 0x7u;
        packed_word_30 = blockIdx.y;
        const uint64_t optional_input_handle = combine_words(parameters[8], parameters[9]);
        packed_word_8 = parameters[46];
        packed_word_9 = parameters[47];
        uniform_word_28 = parameters[48];
        uniform_word_29 = parameters[49];
        uniform_word_30 = parameters[36];
        packed_word_40 = parameters[44];
        packed_word_41 = parameters[45];
        uniform_word_32 = parameters[30];
        packed_word_20 = reciprocal_bits(uniform_word_6);
        uniform_word_34 = parameters[2];
        uniform_word_35 = parameters[3];
        condition_0 = optional_input_handle != 0;
        uniform_word_6 = 0u;
        packed_word_23 = __float_as_uint(__int2float_rn(int32_t(packed_word_7)));
        // This is a floating scalar, with a lifetime separate from texture handles.
        float doubled_parameter;
        const float scalar_parameter = __uint_as_float(uniform_word_29);
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(doubled_parameter) : "f"(scalar_parameter), "f"(scalar_parameter));
        uniform_word_4 = __float_as_uint(doubled_parameter);
        uniform_word_31 = parameters[38];
        condition_0 = 0u != uint32_t(uniform_word_28) && !condition_0;
        packed_word_28 = reciprocal_bits(packed_word_23);
        packed_word_32 = packed_word_5 * 0x8u + packed_word_2;
        packed_word_3 = float_minmax(packed_word_8, packed_word_9, !true);
        packed_word_16 = parameters[34];
        uniform_word_28 = parameters[4];
        uniform_word_29 = parameters[5];
        packed_word_4 = packed_word_7 * 0x2u + (0u - packed_word_32);
        condition_1 = int32_t(packed_word_32) >= int32_t(packed_word_7);
        {
            uint32_t loaded_word;
            asm("{.reg .pred p; setp.ge.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                : "=r"(loaded_word)
                : "f"(__uint_as_float(packed_word_3)), "f"(__uint_as_float(0u)));
            condition_5 = bool(loaded_word);
        }
        uniform_word_33 = parameters[32];
        packed_word_21 = reciprocal_bits(uniform_word_7);
        packed_word_5 = packed_word_4 + (0u - 0x2u);
        packed_word_18 = parameters[28];
        packed_word_22 = __float_as_uint(__int2float_rn(int32_t(packed_word_6)));
        uniform_word_5 = 0x400u;
        packed_word_40 = !condition_5 ? packed_word_40 : 0x3f800000u;
        packed_word_5 = !condition_1 ? packed_word_32 : packed_word_5;
        packed_word_30 = packed_word_30 * 0x8u;
        packed_word_24 =
            uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(packed_word_41))));
        packed_word_32 = packed_word_32 * (0u - 0x72594cbdu);
        packed_word_31 = parameters[50];
        packed_word_5 = __float_as_uint(__int2float_rn(int32_t(packed_word_5)));
        uniform_word_5 = (uniform_word_6 << 0x18u) + uniform_word_5;
        condition_1 = 0ull != combine_words(uniform_word_28, uniform_word_29);
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_34))
            : "f"(__uint_as_float((packed_word_20 ^ 0x80000000u))),
              "f"(__uint_as_float((0u ^ 0x80000000u))));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_5))
            : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(0x3f000000u)));
        packed_word_33 = reciprocal_bits(packed_word_22);
        packed_word_25 = parameters[52];
        packed_word_39 = (packed_word_29 << 0x4u) + uniform_word_5;
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_35))
            : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_5)));
        condition_1 = 0ull == combine_words(uniform_word_34, uniform_word_35) || !condition_1;
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_36))
            : "f"(__uint_as_float((packed_word_21 ^ 0x80000000u))),
              "f"(__uint_as_float((0u ^ 0x80000000u))));
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_16))
            : "f"(__uint_as_float(packed_word_35)), "f"(__uint_as_float(uniform_word_30)),
              "f"(__uint_as_float(packed_word_16)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_37))
            : "f"(__uint_as_float(packed_word_23)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        packed_word_26 = parameters[35];
        asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_35)), "f"(__uint_as_float(uniform_word_32)),
              "f"(__uint_as_float(packed_word_18)));
        asm("add.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_38))
            : "f"(__uint_as_float(packed_word_22)),
              "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
        packed_word_39 = packed_word_39 + 0x400u;
        packed_word_40 = pack_half(packed_word_40, uniform_word_4);
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_16))
            : "f"(__uint_as_float(packed_word_16)), "f"(__uint_as_float(uniform_word_31)));
        asm("mul.rn.ftz.f32 %0,%1,%2;"
            : "=f"(reinterpret_cast<float&>(packed_word_18))
            : "f"(__uint_as_float(packed_word_18)), "f"(__uint_as_float(uniform_word_33)));
        packed_word_27 = parameters[44];
        packed_word_31 = packed_word_31 * (0u - 0x61c88647u);
        packed_word_2 = parameters[22];
        packed_word_3 = parameters[23];
        packed_word_8 = parameters[16];
        packed_word_9 = parameters[17];
        packed_word_10 = parameters[18];
        packed_word_11 = parameters[19];
        packed_word_12 = parameters[10];
        packed_word_13 = parameters[11];
        packed_word_14 = parameters[46];
        packed_word_15 = parameters[47];
        do {
            packed_word_42 =
                uint32_t(combine_words(packed_word_29, 0u) << 0x1du >> 32) + packed_word_30;
            uniform_word_4 = parameters[0];
            uniform_word_5 = parameters[1];
            packed_word_4 = packed_word_25 * 0x2u + (0u - packed_word_42);
            condition_2 = int32_t(packed_word_42) >= int32_t(packed_word_25);
            packed_word_5 = packed_word_4 + (0u - 0x2u);
            packed_word_5 = !condition_2 ? packed_word_42 : packed_word_5;
            packed_word_5 = __float_as_uint(__int2float_rn(int32_t(packed_word_5)));
            asm("add.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_6))
                : "f"(__uint_as_float(packed_word_5)), "f"(__uint_as_float(0x3f000000u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_6))
                : "f"(__uint_as_float(packed_word_33)), "f"(__uint_as_float(packed_word_6)));
            asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(uniform_word_36)),
                  "f"(__uint_as_float(packed_word_26)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(uniform_word_37)));
            {
                uint32_t sampled_channels[4];
                // Sample the frame texture using its independent texture/sampler handle.
                image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                          packed_word_16, packed_word_17);
                packed_word_4 = sampled_channels[0];
                packed_word_5 = sampled_channels[1];
                packed_word_45 = sampled_channels[2];
            }
            packed_word_42 = packed_word_42 * (0u - 0x27e9c7bfu);
            packed_word_19 = 0xcaa5b80du;
            packed_word_41 = 0xfa6dc5f9u;
            packed_word_42 = packed_word_31 ^ packed_word_32 ^ packed_word_42;
            packed_word_42 = 0x243f6a88u ^ packed_word_42;
            packed_word_7 = uint32_t(combine_words(packed_word_42, 0u) << 0x4u >> 32) + 0x4u;
            packed_word_7 = uint32_t(uint32_t(packed_word_42) >> packed_word_7);
            packed_word_7 = packed_word_42 ^ packed_word_7;
            packed_word_7 = packed_word_7 * 0x108ef2d9u;
            packed_word_42 = uint32_t(uint32_t(packed_word_7) >> 0x16u);
            packed_word_42 = packed_word_42 ^ packed_word_7;
            packed_word_7 = 0x2c9277b5u;
            packed_word_17 = packed_word_42 * packed_word_19 + 0x21dd796bu;
            packed_word_19 = 0x83232c31u;
            packed_word_7 = packed_word_42 * packed_word_7 + (0u - 0x53a9b4fbu);
            packed_word_46 = uint32_t(combine_words(packed_word_17, 0u) << 0x4u >> 32) + 0x4u;
            packed_word_19 = packed_word_42 * packed_word_19 + 0x3463e0acu;
            packed_word_44 = uint32_t(combine_words(packed_word_7, 0u) << 0x4u >> 32) + 0x4u;
            packed_word_42 = packed_word_42 * packed_word_41 + 0x4712a88eu;
            packed_word_46 = uint32_t(uint32_t(packed_word_17) >> packed_word_46);
            packed_word_44 = uint32_t(uint32_t(packed_word_7) >> packed_word_44);
            packed_word_46 = packed_word_17 ^ packed_word_46;
            packed_word_44 = packed_word_44 ^ packed_word_7;
            packed_word_7 = uint32_t(combine_words(packed_word_42, 0u) << 0x4u >> 32) + 0x4u;
            packed_word_46 = packed_word_46 * 0x108ef2d9u;
            packed_word_48 = uint32_t(combine_words(packed_word_19, 0u) << 0x4u >> 32) + 0x4u;
            packed_word_44 = packed_word_44 * 0x108ef2d9u;
            packed_word_7 = uint32_t(uint32_t(packed_word_42) >> packed_word_7);
            packed_word_17 = uint32_t(uint32_t(packed_word_46) >> 0x1eu);
            packed_word_46 = uint32_t(uint32_t(packed_word_46) >> 0x8u);
            packed_word_42 = packed_word_42 ^ packed_word_7;
            packed_word_48 = uint32_t(uint32_t(packed_word_19) >> packed_word_48);
            packed_word_7 = uint32_t(uint32_t(packed_word_44) >> 0x1eu);
            packed_word_42 = packed_word_42 * 0x108ef2d9u;
            packed_word_44 = uint32_t(uint32_t(packed_word_44) >> 0x8u);
            packed_word_17 = packed_word_17 ^ packed_word_46;
            packed_word_48 = packed_word_19 ^ packed_word_48;
            packed_word_7 = packed_word_44 ^ packed_word_7;
            packed_word_19 = packed_word_17 + 0x1u;
            packed_word_48 = packed_word_48 * 0x108ef2d9u;
            packed_word_17 = uint32_t(uint32_t(packed_word_42) >> 0x1eu);
            packed_word_7 = packed_word_7 + 0x1u;
            packed_word_19 = __float_as_uint(__uint2float_rn(packed_word_19));
            packed_word_41 = uint32_t(uint32_t(packed_word_48) >> 0x1eu);
            packed_word_48 = uint32_t(uint32_t(packed_word_48) >> 0x8u);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(0x33800000u)));
            packed_word_7 = __float_as_uint(__uint2float_rn(packed_word_7));
            packed_word_42 = uint32_t(uint32_t(packed_word_42) >> 0x8u);
            packed_word_41 = packed_word_41 ^ packed_word_48;
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_7))
                : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(0x33800000u)));
            packed_word_17 = packed_word_17 ^ packed_word_42;
            packed_word_19 = image_mufu_lg2(packed_word_19);
            packed_word_41 = packed_word_41 + 0x1u;
            packed_word_17 = packed_word_17 + 0x1u;
            packed_word_41 = __float_as_uint(__uint2float_rn(packed_word_41));
            packed_word_7 = image_mufu_lg2(packed_word_7);
            packed_word_17 = __float_as_uint(__uint2float_rn(packed_word_17));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_41)), "f"(__uint_as_float(0x33800000u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(0x33800000u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_41)), "f"(__uint_as_float(0x40c90fdbu)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(0x40c90fdbu)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(0x3f317218u)));
            angle_bits_0 = packed_word_41;
            asm("mul.rz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_46))
                : "f"(__uint_as_float(packed_word_41)), "f"(__uint_as_float(0x3e22f983u)));
            angle_bits_1 = packed_word_17;
            asm("mul.rz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_41))
                : "f"(__uint_as_float(packed_word_17)), "f"(__uint_as_float(0x3e22f983u)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_19))
                : "f"(__uint_as_float(packed_word_19)),
                  "f"(__uint_as_float((0x40000000u ^ 0x80000000u))));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(0x3f317218u)));
            packed_word_46 = cosine_radians(angle_bits_0);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_7))
                : "f"(__uint_as_float(packed_word_17)),
                  "f"(__uint_as_float((0x40000000u ^ 0x80000000u))));
            packed_word_19 = image_mufu_sqrt(packed_word_19);
            packed_word_44 = cosine_radians(angle_bits_1);
            packed_word_7 = image_mufu_sqrt(packed_word_7);
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_17))
                : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_46)));
            packed_word_42 = sine_radians(angle_bits_1);
            packed_word_17 =
                uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(packed_word_17))));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_44))
                : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(packed_word_44)));
            asm("mul.rn.ftz.f32 %0,%1,%2;"
                : "=f"(reinterpret_cast<float&>(packed_word_43))
                : "f"(__uint_as_float(packed_word_7)), "f"(__uint_as_float(packed_word_42)));
            packed_word_5 = pack_half(packed_word_5, packed_word_4);
            packed_word_45 = pack_half(0u, packed_word_45);
            packed_word_4 =
                uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(packed_word_44))));
            packed_word_5 =
                add_half_pairs(select_half_operand<2, false, false>(packed_word_5), 0xb800b800u);
            packed_word_45 =
                add_half_pairs(select_half_operand<0, false, false>(packed_word_45), 0xb800b800u);
            packed_word_42 =
                multiply_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                    select_half_operand<0, false, false>(packed_word_40));
            packed_word_5 =
                uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(packed_word_43))));
            packed_word_7 =
                multiply_half_pairs(select_half_operand<0, false, false>(packed_word_45),
                                    select_half_operand<0, false, false>(packed_word_40));
            packed_word_41 = permute_signed(packed_word_42, packed_word_41, 0x7632u);
            packed_word_46 = permute_signed(packed_word_42, packed_word_46, 0x7610u);
            packed_word_47 = permute_signed(packed_word_41, packed_word_47, 0x7610u);
            packed_word_48 = permute_signed(packed_word_7, packed_word_48, 0x7610u);
            if (!condition_1) {
                uniform_condition_0 = combine_words(uniform_word_20, uniform_word_21) != 0ull;
                packed_word_44 = 0u;
                packed_word_19 = 0u;
                uniform_word_6 = parameters[4];
                uniform_word_7 = parameters[5];
                if (uniform_condition_0) {
                    uniform_word_4 = parameters[6];
                    uniform_word_5 = parameters[7];
                    asm("add.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_43))
                        : "f"(__uint_as_float((packed_word_20 ^ 0x80000000u))),
                          "f"(__uint_as_float(packed_word_35)));
                    asm("add.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_44))
                        : "f"(__uint_as_float((packed_word_21 ^ 0x80000000u))),
                          "f"(__uint_as_float(packed_word_6)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_19))
                        : "f"(__uint_as_float(packed_word_35)),
                          "f"(__uint_as_float(uniform_word_12)),
                          "f"(__uint_as_float(packed_word_2)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_45))
                        : "f"(__uint_as_float(packed_word_6)),
                          "f"(__uint_as_float(uniform_word_13)),
                          "f"(__uint_as_float(packed_word_3)));
                    asm("add.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_46))
                        : "f"(__uint_as_float(packed_word_21)),
                          "f"(__uint_as_float(packed_word_6)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_43))
                        : "f"(__uint_as_float(packed_word_43)),
                          "f"(__uint_as_float(uniform_word_12)),
                          "f"(__uint_as_float(packed_word_2)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_47))
                        : "f"(__uint_as_float(packed_word_44)),
                          "f"(__uint_as_float(uniform_word_13)),
                          "f"(__uint_as_float(packed_word_3)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_44))
                        : "f"(__uint_as_float(packed_word_19)),
                          "f"(__uint_as_float(uniform_word_14)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_50))
                        : "f"(__uint_as_float(packed_word_46)),
                          "f"(__uint_as_float(uniform_word_13)),
                          "f"(__uint_as_float(packed_word_3)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_45))
                        : "f"(__uint_as_float(packed_word_45)),
                          "f"(__uint_as_float(uniform_word_15)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_47))
                        : "f"(__uint_as_float(packed_word_47)),
                          "f"(__uint_as_float(uniform_word_15)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_46))
                        : "f"(__uint_as_float(packed_word_43)),
                          "f"(__uint_as_float(uniform_word_14)));
                    asm("add.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_49))
                        : "f"(__uint_as_float(packed_word_20)),
                          "f"(__uint_as_float(packed_word_35)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_53))
                        : "f"(__uint_as_float(packed_word_50)),
                          "f"(__uint_as_float(uniform_word_15)));
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_49))
                        : "f"(__uint_as_float(packed_word_49)),
                          "f"(__uint_as_float(uniform_word_12)),
                          "f"(__uint_as_float(packed_word_2)));
                    {
                        uint32_t sampled_channels[4];
                        image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                                  packed_word_44, packed_word_45);
                        packed_word_44 = sampled_channels[0];
                    }
                    {
                        uint32_t sampled_channels[4];
                        image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                                  packed_word_46, packed_word_47);
                        packed_word_19 = sampled_channels[0];
                    }
                    packed_word_52 = packed_word_46;
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_48))
                        : "f"(__uint_as_float(packed_word_49)),
                          "f"(__uint_as_float(uniform_word_14)));
                    packed_word_49 = packed_word_47;
                    packed_word_55 = packed_word_53;
                    packed_word_54 = packed_word_48;
                    {
                        uint32_t sampled_channels[4];
                        image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                                  packed_word_48, packed_word_49);
                        packed_word_48 = sampled_channels[0];
                    }
                    {
                        uint32_t sampled_channels[4];
                        image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                                  packed_word_52, packed_word_53);
                        packed_word_50 = sampled_channels[0];
                    }
                    {
                        uint32_t sampled_channels[4];
                        image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                                  packed_word_54, packed_word_55);
                        packed_word_51 = sampled_channels[0];
                    }
                    packed_word_46 = reciprocal_bits(uniform_word_22);
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_46))
                        : "f"(__uint_as_float(packed_word_46)),
                          "f"(__uint_as_float(uniform_word_12)));

                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_19)),
                              "f"(__uint_as_float(packed_word_44)));
                        condition_3 = bool(loaded_word);
                    }
                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.geu.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_19)),
                              "f"(__uint_as_float(packed_word_44)));
                        condition_2 = bool(loaded_word);
                    }
                    condition_3 = 0u != uint32_t(uniform_word_38) && !condition_3;
                    condition_2 = 0u == uint32_t(uniform_word_38) && condition_2;
                    condition_3 = condition_2 | condition_3;
                    packed_word_19 = condition_3 ? packed_word_44 : packed_word_19;
                    packed_word_43 = !condition_3 ? packed_word_36 : 0u;

                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_48)),
                              "f"(__uint_as_float(packed_word_19)));
                        condition_4 = bool(loaded_word);
                    }
                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.geu.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_48)),
                              "f"(__uint_as_float(packed_word_19)));
                        condition_2 = bool(loaded_word);
                    }
                    condition_4 = 0u != uint32_t(uniform_word_38) && !condition_4;
                    condition_2 = 0u == uint32_t(uniform_word_38) && condition_2;
                    condition_2 = condition_2 | condition_4;
                    packed_word_19 = condition_2 ? packed_word_19 : packed_word_48;
                    packed_word_44 = condition_2 ? packed_word_43 : packed_word_36;
                    packed_word_48 = reciprocal_bits(uniform_word_23);

                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_50)),
                              "f"(__uint_as_float(packed_word_19)));
                        condition_4 = bool(loaded_word);
                    }
                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.geu.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_50)),
                              "f"(__uint_as_float(packed_word_19)));
                        condition_6 = bool(loaded_word);
                    }
                    condition_4 = 0u != uint32_t(uniform_word_38) && !condition_4;
                    condition_6 = 0u == uint32_t(uniform_word_38) && condition_6;
                    condition_4 = condition_6 | condition_4;
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_48))
                        : "f"(__uint_as_float(packed_word_48)),
                          "f"(__uint_as_float(uniform_word_13)));
                    packed_word_50 = condition_4 ? packed_word_19 : packed_word_50;
                    packed_word_19 = !condition_3 ? packed_word_34 : 0u;
                    packed_word_44 = condition_4 ? packed_word_44 : packed_word_21;
                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.gt.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_51)),
                              "f"(__uint_as_float(packed_word_50)));
                        condition_3 = bool(loaded_word);
                    }
                    {
                        uint32_t loaded_word;
                        asm("{.reg .pred p; setp.geu.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                            : "=r"(loaded_word)
                            : "f"(__uint_as_float(packed_word_51)),
                              "f"(__uint_as_float(packed_word_50)));
                        condition_6 = bool(loaded_word);
                    }
                    packed_word_19 = condition_2 ? packed_word_19 : packed_word_20;
                    condition_3 = 0u != uint32_t(uniform_word_38) && !condition_3;
                    condition_6 = 0u == uint32_t(uniform_word_38) && condition_6;
                    packed_word_19 = condition_4 ? packed_word_19 : packed_word_34;
                    condition_3 = condition_6 | condition_3;
                    packed_word_19 = condition_3 ? packed_word_19 : packed_word_20;
                    packed_word_43 = condition_3 ? packed_word_44 : packed_word_21;
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_44))
                        : "f"(__uint_as_float(packed_word_19)),
                          "f"(__uint_as_float(packed_word_46)));
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_19))
                        : "f"(__uint_as_float(packed_word_43)),
                          "f"(__uint_as_float(packed_word_48)));
                }
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_35)), "f"(__uint_as_float(packed_word_44)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(packed_word_19)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(packed_word_10)),
                      "f"(__uint_as_float(packed_word_8)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(packed_word_11)),
                      "f"(__uint_as_float(packed_word_9)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(uniform_word_24)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(uniform_word_25)));
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_6, uniform_word_7),
                              packed_word_44, packed_word_45);
                    packed_word_44 = sampled_channels[0];
                    packed_word_45 = sampled_channels[1];
                }
                uniform_word_4 = parameters[2];
                uniform_word_5 = parameters[3];
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(uniform_word_26)),
                      "f"(__uint_as_float(packed_word_35)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(uniform_word_27)),
                      "f"(__uint_as_float(packed_word_6)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(packed_word_46)),
                      "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))));
                asm("cvt.rmi.ftz.f32.f32 %0,%1;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_43)));
                asm("cvt.rmi.ftz.f32.f32 %0,%1;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_47)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(0x3f000000u)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(packed_word_23)), "f"(__uint_as_float(packed_word_46)),
                      "f"(__uint_as_float((packed_word_48 ^ 0x80000000u))));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_49))
                    : "f"(__uint_as_float(packed_word_47)), "f"(__uint_as_float(0x3f000000u)));
                asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_46)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_22)), "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float((packed_word_49 ^ 0x80000000u))));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_46)), "f"(__uint_as_float(packed_word_46)));
                asm("add.rn.ftz.sat.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(0u)), "f"(__uint_as_float(packed_word_19)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_46)), "f"(__uint_as_float(packed_word_45)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_50))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(0x40200000u)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_19)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(packed_word_46)), "f"(__uint_as_float(packed_word_43)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_50))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(0x3fc00000u)),
                      "f"(__uint_as_float((packed_word_50 ^ 0x80000000u))));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float((packed_word_45 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_43)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_44)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(packed_word_46)),
                      "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_45)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_52))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(0x40200000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_50))
                    : "f"(__uint_as_float(packed_word_50)), "f"(__uint_as_float(0x3f800000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_47)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float((packed_word_46 ^ 0x80000000u))),
                      "f"(__uint_as_float(0x3f800000u)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_52))
                    : "f"(__uint_as_float(packed_word_47)), "f"(__uint_as_float(0x3fc00000u)),
                      "f"(__uint_as_float((packed_word_52 ^ 0x80000000u))));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float((packed_word_44 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_47)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float((0x3f000000u ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_44)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(0x3f000000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float((packed_word_50 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_45)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_52))
                    : "f"(__uint_as_float(packed_word_52)), "f"(__uint_as_float(0x3f800000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_51))
                    : "f"(__uint_as_float((packed_word_19 ^ 0x80000000u))),
                      "f"(__uint_as_float(0x3f800000u)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_47)), "f"(__uint_as_float(0x3f000000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float((packed_word_43 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_44)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float((packed_word_52 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_51)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_50))
                    : "f"(__uint_as_float(packed_word_50)), "f"(__uint_as_float(packed_word_45)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float((packed_word_47 ^ 0x80000000u))),
                      "f"(__uint_as_float(packed_word_44)));
                packed_word_53 = reciprocal_bits(packed_word_50);
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_51))
                    : "f"(__uint_as_float(packed_word_52)), "f"(__uint_as_float(packed_word_44)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_52))
                    : "f"(__uint_as_float(packed_word_48)),
                      "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
                packed_word_54 = reciprocal_bits(packed_word_51);
                packed_word_52 = float_minmax(packed_word_52, 0x3f000000u, !true);
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_53))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(packed_word_53)),
                      "f"(__uint_as_float(packed_word_48)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_48)), "f"(__uint_as_float(0x40000000u)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_49)),
                      "f"(__uint_as_float((0x3f800000u ^ 0x80000000u))));
                packed_word_48 = float_minmax(packed_word_48, 0x3f000000u, !true);
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_54))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(packed_word_54)),
                      "f"(__uint_as_float(packed_word_49)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_49)), "f"(__uint_as_float(0x40000000u)));
                packed_word_49 = float_minmax(packed_word_45, 0x3f000000u, !true);
                packed_word_45 = float_minmax(packed_word_37, packed_word_52, true);
                packed_word_55 = float_minmax(packed_word_37, packed_word_48, true);
                packed_word_48 = float_minmax(packed_word_53, 0x3f000000u, !true);
                packed_word_57 = float_minmax(packed_word_44, 0x3f000000u, !true);
                packed_word_44 = float_minmax(packed_word_38, packed_word_49, true);
                packed_word_49 = float_minmax(packed_word_54, 0x3f000000u, !true);
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_53))
                    : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_45)));
                packed_word_45 = float_minmax(packed_word_37, packed_word_48, true);
                packed_word_52 = float_minmax(packed_word_38, packed_word_57, true);
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_33)), "f"(__uint_as_float(packed_word_44)));
                packed_word_48 = float_minmax(packed_word_38, packed_word_49, true);
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_55))
                    : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_55)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_28)), "f"(__uint_as_float(packed_word_45)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_54))
                    : "f"(__uint_as_float(packed_word_33)), "f"(__uint_as_float(packed_word_52)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_52))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(uniform_word_17)),
                      "f"(__uint_as_float(packed_word_13)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_33)), "f"(__uint_as_float(packed_word_48)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_49))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(uniform_word_16)),
                      "f"(__uint_as_float(packed_word_12)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_53))
                    : "f"(__uint_as_float(packed_word_53)), "f"(__uint_as_float(uniform_word_16)),
                      "f"(__uint_as_float(packed_word_12)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_54))
                    : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(uniform_word_17)),
                      "f"(__uint_as_float(packed_word_13)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_48)), "f"(__uint_as_float(uniform_word_17)),
                      "f"(__uint_as_float(packed_word_13)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_49)), "f"(__uint_as_float(uniform_word_18)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_49))
                    : "f"(__uint_as_float(packed_word_52)), "f"(__uint_as_float(uniform_word_19)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_55))
                    : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(uniform_word_16)),
                      "f"(__uint_as_float(packed_word_12)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_45))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(uniform_word_19)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_44))
                    : "f"(__uint_as_float(packed_word_53)), "f"(__uint_as_float(uniform_word_18)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_61))
                    : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(uniform_word_19)));
                packed_word_54 = packed_word_48;
                packed_word_60 = packed_word_48;
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_56))
                    : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(uniform_word_18)));
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                              packed_word_48, packed_word_49);
                    packed_word_48 = sampled_channels[0];
                    packed_word_49 = sampled_channels[1];
                    packed_word_53 = sampled_channels[2];
                }
                packed_word_55 = packed_word_45;
                packed_word_57 = packed_word_45;
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                              packed_word_44, packed_word_45);
                    packed_word_44 = sampled_channels[0];
                    packed_word_45 = sampled_channels[1];
                    packed_word_52 = sampled_channels[2];
                }
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                              packed_word_54, packed_word_55);
                    packed_word_54 = sampled_channels[0];
                    packed_word_55 = sampled_channels[1];
                    packed_word_58 = sampled_channels[2];
                }
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                              packed_word_60, packed_word_61);
                    packed_word_60 = sampled_channels[0];
                    packed_word_61 = sampled_channels[1];
                    packed_word_59 = sampled_channels[2];
                }
                {
                    uint32_t sampled_channels[4];
                    image_tex(sampled_channels, combine_words(uniform_word_4, uniform_word_5),
                              packed_word_56, packed_word_57);
                    packed_word_56 = sampled_channels[0];
                    packed_word_57 = sampled_channels[1];
                    packed_word_62 = sampled_channels[2];
                }
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_63))
                    : "f"(__uint_as_float(packed_word_46)), "f"(__uint_as_float(packed_word_51)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_46))
                    : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_50)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_19))
                    : "f"(__uint_as_float(packed_word_50)), "f"(__uint_as_float(packed_word_51)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_50))
                    : "f"(__uint_as_float(packed_word_47)), "f"(__uint_as_float(packed_word_50)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_43))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(packed_word_51)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_64))
                    : "f"(__uint_as_float(packed_word_63)), "f"(__uint_as_float(packed_word_46)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_19)), "f"(__uint_as_float(packed_word_64)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_64))
                    : "f"(__uint_as_float(packed_word_50)), "f"(__uint_as_float(packed_word_47)));
                asm("add.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_64))
                    : "f"(__uint_as_float(packed_word_43)), "f"(__uint_as_float(packed_word_64)));
                packed_word_64 = reciprocal_bits(packed_word_64);

                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_48)), "f"(__uint_as_float(packed_word_46)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_53))
                    : "f"(__uint_as_float(packed_word_53)), "f"(__uint_as_float(packed_word_46)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_49)), "f"(__uint_as_float(packed_word_46)));

                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_44)), "f"(__uint_as_float(packed_word_63)),
                      "f"(__uint_as_float(packed_word_47)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_53))
                    : "f"(__uint_as_float(packed_word_52)), "f"(__uint_as_float(packed_word_63)),
                      "f"(__uint_as_float(packed_word_53)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_45)), "f"(__uint_as_float(packed_word_63)),
                      "f"(__uint_as_float(packed_word_48)));

                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_54)), "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float(packed_word_47)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_58))
                    : "f"(__uint_as_float(packed_word_58)), "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float(packed_word_53)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_55)), "f"(__uint_as_float(packed_word_19)),
                      "f"(__uint_as_float(packed_word_48)));

                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_60)), "f"(__uint_as_float(packed_word_50)),
                      "f"(__uint_as_float(packed_word_47)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_59))
                    : "f"(__uint_as_float(packed_word_59)), "f"(__uint_as_float(packed_word_50)),
                      "f"(__uint_as_float(packed_word_58)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_48))
                    : "f"(__uint_as_float(packed_word_61)), "f"(__uint_as_float(packed_word_50)),
                      "f"(__uint_as_float(packed_word_48)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_56)), "f"(__uint_as_float(packed_word_43)),
                      "f"(__uint_as_float(packed_word_47)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_59))
                    : "f"(__uint_as_float(packed_word_62)), "f"(__uint_as_float(packed_word_43)),
                      "f"(__uint_as_float(packed_word_59)));
                asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                    : "=f"(reinterpret_cast<float&>(packed_word_57))
                    : "f"(__uint_as_float(packed_word_57)), "f"(__uint_as_float(packed_word_43)),
                      "f"(__uint_as_float(packed_word_48)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_47))
                    : "f"(__uint_as_float(packed_word_64)), "f"(__uint_as_float(packed_word_47)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_59))
                    : "f"(__uint_as_float(packed_word_64)), "f"(__uint_as_float(packed_word_59)));
                asm("mul.rn.ftz.f32 %0,%1,%2;"
                    : "=f"(reinterpret_cast<float&>(packed_word_64))
                    : "f"(__uint_as_float(packed_word_64)), "f"(__uint_as_float(packed_word_57)));
                packed_word_59 = pack_half(0u, packed_word_59);
                packed_word_47 = pack_half(packed_word_64, packed_word_47);
                packed_word_59 = add_half_pairs(
                    select_half_operand<0, false, false>(packed_word_59), 0xb800b800u);
                packed_word_47 = add_half_pairs(
                    select_half_operand<2, false, false>(packed_word_47), 0xb800b800u);
                packed_word_59 =
                    multiply_half_pairs(select_half_operand<0, false, false>(packed_word_59),
                                        select_half_operand<0, false, false>(packed_word_40));
                packed_word_46 =
                    multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                        select_half_operand<0, false, false>(packed_word_40));
                packed_word_48 = permute_signed(packed_word_59, packed_word_48, 0x7610u);
                packed_word_47 = permute_signed(packed_word_46, packed_word_47, 0x7632u);
            }
            if (!condition_0) {
                uniform_condition_0 = combine_words(uniform_word_8, uniform_word_9) != 0ull;
                condition_3 = uniform_condition_0;
                if (uniform_condition_0) {
                    uniform_word_5 = parameters[29];

                    uniform_word_4 = parameters[31];

                    uniform_word_6 = parameters[33];

                    uniform_word_28 = parameters[8];
                    uniform_word_29 = parameters[9];
                }
                packed_word_19 = uniform_word_5;
                if (condition_3) {
                    asm("fma.rn.ftz.f32 %0,%1,%2,%3;"
                        : "=f"(reinterpret_cast<float&>(packed_word_19))
                        : "f"(__uint_as_float(packed_word_6)), "f"(__uint_as_float(uniform_word_4)),
                          "f"(__uint_as_float(packed_word_19)));

                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_19))
                        : "f"(__uint_as_float(packed_word_19)),
                          "f"(__uint_as_float(uniform_word_6)));

                    {
                        {
                            uint32_t sampled_channels[4];
                            image_tex(sampled_channels,
                                      combine_words(uniform_word_28, uniform_word_29),
                                      packed_word_18, packed_word_19);
                            packed_word_50 = sampled_channels[1];
                            packed_word_51 = sampled_channels[2];
                        }
                    }
                }
                uniform_word_4 = parameters[43];
                condition_2 = 0u != uint32_t(uniform_word_40);
                packed_word_44 = permute_signed(0u, packed_word_44, 0x7610u);
                uniform_word_5 = parameters[44];
                if (uniform_condition_0) {
                    uniform_word_6 = parameters[43];

                    uniform_word_7 = parameters[44];
                }
                if (condition_2) {
                    packed_word_45 =
                        add_half_pairs(select_half_operand<0, false, true>(0u), 0xbc00bc00u);
                }
                packed_word_6 = uniform_word_4;
                if (condition_2) {
                    packed_word_44 = permute_signed(packed_word_45, packed_word_44, 0x7610u);
                }
                packed_word_19 = 0xbc00u;
                packed_word_43 = uniform_word_5;
                packed_word_45 = permute_signed(0u, packed_word_45, 0x7610u);
                if (condition_2) {
                    packed_word_45 = permute_signed(packed_word_19, packed_word_45, 0x7610u);
                }
                if (condition_3) {
                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_6))
                        : "f"(__uint_as_float(packed_word_50)),
                          "f"(__uint_as_float(uniform_word_6)));

                    asm("mul.rn.ftz.f32 %0,%1,%2;"
                        : "=f"(reinterpret_cast<float&>(packed_word_43))
                        : "f"(__uint_as_float(packed_word_51)),
                          "f"(__uint_as_float(uniform_word_7)));
                }
                packed_word_6 = pack_half(packed_word_43, packed_word_6);
                packed_word_43 = permute_signed(packed_word_6, packed_word_43, 0x7632u);
                packed_word_19 = permute_signed(packed_word_6, packed_word_19, 0x7610u);
            } else {
                {
                    uint32_t loaded_word;
                    asm("{.reg .pred p; setp.ge.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                        : "=r"(loaded_word)
                        : "f"(__uint_as_float(packed_word_15)), "f"(__uint_as_float(0u)));
                    condition_2 = bool(loaded_word);
                }
                {
                    uint32_t loaded_word;
                    asm("{.reg .pred p; setp.ge.ftz.f32 p,%1,%2; selp.u32 %0,1,0,p;}"
                        : "=r"(loaded_word)
                        : "f"(__uint_as_float(packed_word_14)), "f"(__uint_as_float(0u)));
                    condition_3 = bool(loaded_word);
                }
                packed_word_43 = permute_signed(packed_word_40, packed_word_43, 0x7632u);
                packed_word_19 = condition_2 ? packed_word_15 : packed_word_27;
                packed_word_6 = condition_3 ? packed_word_14 : packed_word_27;
                packed_word_19 = condition_5 ? packed_word_19 : 0x3f800000u ^ 0x80000000u;
                packed_word_6 = condition_5 ? packed_word_6 : 0x3f800000u ^ 0x80000000u;
                packed_word_19 = pack_half(0u, packed_word_19);
                packed_word_6 = pack_half(packed_word_6, uniform_word_39);
                packed_word_45 = permute_signed(packed_word_19, packed_word_45, 0x7610u);
                packed_word_44 = permute_signed(packed_word_6, packed_word_44, 0x7632u);
                packed_word_19 = permute_signed(packed_word_6, packed_word_19, 0x7610u);
            }
            packed_word_29 = packed_word_0 * 0x20u + packed_word_29;
            packed_word_6 = permute_signed(packed_word_42, packed_word_41, 0x5410u);
            packed_word_5 = packed_word_5 * 0x10000u;
            packed_word_7 = permute_signed(packed_word_7, packed_word_46, 0x5410u);
            condition_2 = uint32_t(packed_word_29) >= uint32_t(0x40u);
            packed_word_48 = permute_signed(packed_word_47, packed_word_48, 0x5410u);
            packed_word_4 = packed_word_4 | packed_word_5;
            packed_word_50 = permute_signed(packed_word_43, packed_word_44, 0x5410u);
            packed_word_49 = permute_signed(packed_word_24, packed_word_19, 0x5410u);
            packed_word_51 = packed_word_45 & 0xffffu;
            packed_word_5 = packed_word_17 + 0x3c000000u;
            store_four_shared_words<CheckBounds>(shared_tile, packed_word_39, packed_word_48,
                                                 packed_word_49, packed_word_50, packed_word_51,
                                                 shared_bytes, memory_ranges, error_counts);
            store_four_shared_words<CheckBounds>(
                shared_tile, packed_word_39 + (0u - 0x400u), packed_word_4, packed_word_5,
                packed_word_6, packed_word_7, shared_bytes, memory_ranges, error_counts);
            packed_word_39 = packed_word_0 * 0x200u + packed_word_39;
        } while (!condition_2);
    }

    packed_word_33 = threadIdx.x & 31u;
    packed_word_82 = parameters[56];
    packed_word_83 = parameters[57];
    uniform_word_5 = 0u;
    uniform_word_4 = 0x400u;
    __syncthreads();
    packed_word_158 = parameters[56];
    packed_word_159 = parameters[57];
    uniform_word_8 = parameters[60];
    uniform_word_9 = parameters[61];
    packed_word_0 = packed_word_33;
    split_address_words(packed_word_56, packed_word_57,
                        int64_t(int32_t(packed_word_33)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_word_82, packed_word_83));
    packed_word_2 = uint32_t(int32_t(packed_word_0) >> 0x2u);
    packed_word_12 = packed_word_33;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2010ull, packed_word_8, packed_word_9,
        packed_word_10, packed_word_11, memory_ranges, error_counts);
    packed_word_3 = uint32_t(combine_words(packed_word_2, 0u) << 0x2u >> 32) + packed_word_2;
    packed_word_12 = packed_word_12 * 0x2u;
    packed_word_3 = packed_word_3 & 0xffffffcu;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2210ull, packed_word_4, packed_word_5,
        packed_word_6, packed_word_7, memory_ranges, error_counts);
    uniform_word_4 = (uniform_word_5 << 0x18u) + uniform_word_4;
    packed_word_12 = packed_word_12 & 0x3fffffe0u;
    packed_word_3 = packed_word_2 + (0u - packed_word_3);
    packed_word_2 = packed_word_0 & 0x3ffffffcu;
    packed_word_3 = packed_word_3 * 0x4u + packed_word_12;
    packed_word_2 = packed_word_3 + packed_word_33 + (0u - packed_word_2);
    packed_word_20 = (packed_word_2 << 0x2u) + uniform_word_4;
    packed_word_28 =
        shared_read<CheckBounds>(shared_tile, packed_word_20, shared_bytes, error_counts);
    packed_word_29 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x100u, shared_bytes, error_counts);
    packed_word_30 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x400u, shared_bytes, error_counts);
    packed_word_31 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x500u, shared_bytes, error_counts);
    packed_word_68 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x40u, shared_bytes, error_counts);
    packed_word_69 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x140u, shared_bytes, error_counts);
    packed_word_70 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x440u, shared_bytes, error_counts);
    packed_word_71 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x540u, shared_bytes, error_counts);
    packed_word_60 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x200u, shared_bytes, error_counts);
    packed_word_61 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x300u, shared_bytes, error_counts);
    packed_word_62 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x600u, shared_bytes, error_counts);
    packed_word_63 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x700u, shared_bytes, error_counts);
    packed_word_12 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x240u, shared_bytes, error_counts);
    packed_word_13 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x340u, shared_bytes, error_counts);
    packed_word_14 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x640u, shared_bytes, error_counts);
    packed_word_15 =
        shared_read<CheckBounds>(shared_tile, packed_word_20 + 0x740u, shared_bytes, error_counts);
    __syncthreads();
    load_four_boundary_words<CheckBounds, 0>(combine_words(packed_word_56, packed_word_57),
                                             packed_word_16, packed_word_17, packed_word_18,
                                             packed_word_19, memory_ranges, error_counts);
    packed_word_0 = packed_word_0 & 0xfffffffcu;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x200ull, packed_word_24, packed_word_25,
        packed_word_26, packed_word_27, memory_ranges, error_counts);
    packed_word_123 = 0u - packed_word_0 + packed_word_33;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1000ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    split_address_words(packed_word_122, packed_word_123,
                        int64_t(int32_t(packed_word_123)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_word_82, packed_word_83));
    packed_word_158 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_158, packed_word_159) + 0x5060ull, memory_ranges, error_counts);
    packed_word_54 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_122, packed_word_123) + 0x2410ull, memory_ranges, error_counts);
    packed_word_125 = packed_word_33 + 0x4u + (0u - packed_word_0);
    split_address_words(packed_word_124, packed_word_125,
                        uint64_t(uint32_t(packed_word_125)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_82, packed_word_83));
    packed_word_81 = packed_word_33 + 0x8u + (0u - packed_word_0);
    packed_word_165 = threadIdx.x & 31u;
    packed_word_86 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_124, packed_word_125) + 0x2410ull, memory_ranges, error_counts);
    split_address_words(packed_word_80, packed_word_81,
                        uint64_t(uint32_t(packed_word_81)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_82, packed_word_83));
    packed_word_122 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_122, packed_word_123) + 0x5470ull, memory_ranges, error_counts);
    packed_word_74 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_80, packed_word_81) + 0x2410ull, memory_ranges, error_counts);
    packed_word_124 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_124, packed_word_125) + 0x5470ull, memory_ranges, error_counts);
    image_mma_half(packed_word_2, packed_word_3, packed_word_28, packed_word_29, packed_word_30,
                   packed_word_31, packed_word_10, packed_word_11, 0u, 0u);
    image_mma_half(packed_word_76, packed_word_77, packed_word_28, packed_word_29, packed_word_30,
                   packed_word_31, packed_word_6, packed_word_7, 0u, 0u);
    image_mma_half(packed_word_92, packed_word_93, packed_word_28, packed_word_29, packed_word_30,
                   packed_word_31, packed_word_8, packed_word_9, 0u, 0u);
    image_mma_half(packed_word_78, packed_word_79, packed_word_28, packed_word_29, packed_word_30,
                   packed_word_31, packed_word_4, packed_word_5, 0u, 0u);
    image_mma_half(packed_word_64, packed_word_65, packed_word_68, packed_word_69, packed_word_70,
                   packed_word_71, packed_word_10, packed_word_11, 0u, 0u);
    image_mma_half(packed_word_88, packed_word_89, packed_word_68, packed_word_69, packed_word_70,
                   packed_word_71, packed_word_8, packed_word_9, 0u, 0u);
    image_mma_half(packed_word_72, packed_word_73, packed_word_68, packed_word_69, packed_word_70,
                   packed_word_71, packed_word_4, packed_word_5, 0u, 0u);
    image_mma_half(packed_word_70, packed_word_71, packed_word_68, packed_word_69, packed_word_70,
                   packed_word_71, packed_word_6, packed_word_7, 0u, 0u);
    image_mma_half(packed_word_58, packed_word_59, packed_word_60, packed_word_61, packed_word_62,
                   packed_word_63, packed_word_10, packed_word_11, 0u, 0u);
    image_mma_half(packed_word_90, packed_word_91, packed_word_60, packed_word_61, packed_word_62,
                   packed_word_63, packed_word_8, packed_word_9, 0u, 0u);
    image_mma_half(packed_word_66, packed_word_67, packed_word_60, packed_word_61, packed_word_62,
                   packed_word_63, packed_word_4, packed_word_5, 0u, 0u);
    image_mma_half(packed_word_62, packed_word_63, packed_word_60, packed_word_61, packed_word_62,
                   packed_word_63, packed_word_6, packed_word_7, 0u, 0u);
    image_mma_half(packed_word_84, packed_word_85, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_10, packed_word_11, 0u, 0u);
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(0u));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                   select_half_operand<2, false, false>(0u));
    image_mma_half(packed_word_60, packed_word_61, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_6, packed_word_7, 0u, 0u);
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(packed_word_10));
    image_mma_half(packed_word_94, packed_word_95, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_8, packed_word_9, 0u, 0u);
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_2),
                                  select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                  select_half_operand<2, false, false>(0u));
    packed_word_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                   select_half_operand<2, false, false>(packed_word_8));
    image_mma_half(packed_word_68, packed_word_69, packed_word_12, packed_word_13, packed_word_14,
                   packed_word_15, packed_word_4, packed_word_5, 0u, 0u);
    // Accumulate packed FP8 activation/weight fragments in warp-wide Tensor Core operations.
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                  select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_64),
                                  select_half_operand<2, false, false>(0u));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_70),
                                   select_half_operand<2, false, false>(0u));
    packed_word_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_71),
                                   select_half_operand<2, false, false>(0u));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_89),
                                  select_half_operand<2, false, false>(packed_word_6));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_58),
                                  select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_88),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_59),
                                  select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                  select_half_operand<2, false, false>(0u));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_72),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(0u));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_73),
                                   select_half_operand<2, false, false>(packed_word_28));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                  select_half_operand<2, false, false>(packed_word_5));
    packed_word_29 = packed_word_33 + 0xcu + (0u - packed_word_0);
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_84),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                   select_half_operand<2, false, false>(0u));
    packed_word_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                  select_half_operand<2, false, false>(packed_word_6));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1200ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_85),
                                  select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_66),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_67),
                                  select_half_operand<2, false, false>(packed_word_12));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_68),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_69),
                                   select_half_operand<2, false, false>(packed_word_28));
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_18, packed_word_19, 0u, 0u);
    packed_word_0 = 0x2b28u;
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_105), 0x44004400u);
    packed_word_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_19), 0xc400c400u);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_19),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    split_address_words(packed_word_82, packed_word_83,
                        uint64_t(uint32_t(packed_word_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_word_82, packed_word_83));
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_19),
        select_half_operand<2, false, false>(packed_word_28), 0x3b283b28u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x400ull, packed_word_28, packed_word_29,
        packed_word_30, packed_word_31, memory_ranges, error_counts);
    packed_word_75 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_82, packed_word_83) + 0x2410ull, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_104), 0x44004400u);
    packed_word_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0xc400c400u);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_16, packed_word_17, 0u, 0u);
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_18),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x44004400u);
    packed_word_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x44004400u);
    packed_word_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0xc400c400u);
    packed_word_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0xc400c400u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_16),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_17),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_18),
        select_half_operand<2, false, false>(packed_word_106), 0x3b283b28u);
    packed_word_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x44004400u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_16),
        select_half_operand<2, false, false>(packed_word_87), 0x3b283b28u);
    packed_word_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_46), 0x44004400u);
    packed_word_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_17),
        select_half_operand<2, false, false>(packed_word_107), 0x3b283b28u);
    packed_word_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0x44004400u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0xc400c400u);
    packed_word_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0xc400c400u);
    packed_word_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0xc400c400u);
    packed_word_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x44004400u);
    packed_word_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_16),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_18),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_102), 0xc400c400u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_16),
        select_half_operand<2, false, false>(packed_word_17), 0x3b283b28u);
    packed_word_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x44004400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_18),
        select_half_operand<2, false, false>(packed_word_19), 0x3b283b28u);
    packed_word_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x44004400u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_100),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x44004400u);
    packed_word_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_102),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0xc400c400u);
    packed_word_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_19), 0xc400c400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<2, false, false>(packed_word_101), 0x3b283b28u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_102),
        select_half_operand<2, false, false>(packed_word_16), 0x3b283b28u);
    packed_word_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_17),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_19),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_17),
        select_half_operand<2, false, false>(packed_word_16), 0x3b283b28u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_19),
        select_half_operand<2, false, false>(packed_word_18), 0x3b283b28u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x600ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    packed_word_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x44004400u);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_26, packed_word_27, 0u,
                             0u);
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0xc400c400u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_100),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_24, packed_word_25, 0u,
                             0u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<2, false, false>(packed_word_101), 0x3b283b28u);
    packed_word_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x44004400u);
    packed_word_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x44004400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x44004400u);
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0xc400c400u);
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0xc400c400u);
    packed_word_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_118), 0xc400c400u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_24, packed_word_25, 0u,
                             0u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0x44004400u);
    packed_word_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_121),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_118), 0x3b283b28u);
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                          select_half_operand<2, false, false>(packed_word_108));
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_26, packed_word_27, 0u,
                             0u);
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                          select_half_operand<2, false, false>(packed_word_109));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_127 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_120), 0xc400c400u);
    packed_word_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_121),
        select_half_operand<2, false, false>(packed_word_126), 0x3b283b28u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_118));
    packed_word_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_102), 0x44004400u);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<2, false, false>(packed_word_106));
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                         select_half_operand<2, false, false>(packed_word_114));
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_92 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_92),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_93 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_93),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_88 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_88),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_89 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_89),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_90 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_90),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_91),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_94 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_94),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_95 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_95),
                                         select_half_operand<2, false, false>(packed_word_54));
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_24, packed_word_25, 0u, 0u);
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                          select_half_operand<2, false, false>(packed_word_107));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                         select_half_operand<2, false, false>(packed_word_113));
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                         select_half_operand<2, false, false>(packed_word_126));
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0xc400c400u);
    packed_word_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_127),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_87));
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_26, packed_word_27, 0u, 0u);
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_127),
        select_half_operand<2, false, false>(packed_word_128), 0x3b283b28u);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_120));
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_50),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_48),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_128));
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_24, packed_word_25, 0u,
                             0u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_50),
        select_half_operand<2, false, false>(packed_word_51), 0x3b283b28u);
    packed_word_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x44004400u);
    packed_word_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<2, false, false>(packed_word_49), 0x3b283b28u);
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_103), 0x44004400u);
    packed_word_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x44004400u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x44004400u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0xc400c400u);
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x44004400u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_114), 0xc400c400u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_48),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_26, packed_word_27, 0u,
                             0u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<2, false, false>(packed_word_50), 0x3b283b28u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x44004400u);
    packed_word_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_48), 0x3b283b28u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_115), 0xc400c400u);
    packed_word_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                         select_half_operand<2, false, false>(packed_word_53));
    packed_word_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_119 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x44004400u);
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                         select_half_operand<2, false, false>(packed_word_50));
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_49));
    packed_word_121 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x44004400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_114), 0xc400c400u);
    packed_word_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_119), 0xc400c400u);
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_53),
                                   select_half_operand<2, false, false>(0u));
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_52));
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(0u));
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_126 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_121), 0xc400c400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_40),
                                   select_half_operand<2, false, false>(packed_word_53));
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_120),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(0u));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                   select_half_operand<2, false, false>(packed_word_40));
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                   select_half_operand<2, false, false>(0u));
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_120),
        select_half_operand<2, false, false>(packed_word_121), 0x3b283b28u);
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_48));
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    packed_word_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_51));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(0u));
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                   select_half_operand<2, false, false>(0u));
    packed_word_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                          select_half_operand<2, false, false>(packed_word_121));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(0u));
    packed_word_119 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                          select_half_operand<2, false, false>(packed_word_119));
    packed_word_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(0u));
    packed_word_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(packed_word_40));
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0x44004400u);
    packed_word_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0x44004400u);
    packed_word_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(packed_word_41));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_45));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(packed_word_112));
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0x44004400u);
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_119),
                                   select_half_operand<2, false, false>(0u));
    packed_word_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0xc400c400u);
    packed_word_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                   select_half_operand<2, false, false>(packed_word_98));
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_87),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_99),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_127 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_126),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_44),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_87),
        select_half_operand<2, false, false>(packed_word_98), 0x3b283b28u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_99),
        select_half_operand<2, false, false>(packed_word_100), 0x3b283b28u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_126),
        select_half_operand<2, false, false>(packed_word_127), 0x3b283b28u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_44),
        select_half_operand<2, false, false>(packed_word_45), 0x3b283b28u);
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                          select_half_operand<2, false, false>(packed_word_100));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_51),
                                   select_half_operand<2, false, false>(packed_word_52));
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(0u));
    packed_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                         select_half_operand<2, false, false>(packed_word_45));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(0u));
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(0u));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                   select_half_operand<2, false, false>(0u));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(0u));
    packed_word_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                    select_half_operand<2, false, false>(0u));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                        select_half_operand<2, false, false>(packed_word_86));
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                   select_half_operand<2, false, false>(packed_word_47));
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                        select_half_operand<2, false, false>(packed_word_86));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(packed_word_49));
    packed_word_64 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(packed_word_52));
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_58 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_45));
    packed_word_59 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(packed_word_98));
    packed_word_84 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_25),
                                   select_half_operand<2, false, false>(packed_word_100));
    packed_word_85 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                         select_half_operand<2, false, false>(packed_word_86));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_36, packed_word_37,
                             packed_word_92, packed_word_93);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1400ull, packed_word_24, packed_word_25,
        packed_word_26, packed_word_27, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_36, packed_word_37,
                             packed_word_88, packed_word_89);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_36, packed_word_37,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_36, packed_word_37,
                             packed_word_94, packed_word_95);
    multiply_accumulate_half(packed_word_2, packed_word_3, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_38, packed_word_39,
                             packed_word_2, packed_word_3);
    multiply_accumulate_half(packed_word_64, packed_word_65, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_38, packed_word_39,
                             packed_word_64, packed_word_65);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_38, packed_word_39,
                             packed_word_58, packed_word_59);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_38, packed_word_39,
                             packed_word_84, packed_word_85);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1600ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    packed_word_66 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_66),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_75));
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_28, packed_word_29, 0u,
                             0u);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_30, packed_word_31, 0u,
                             0u);
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_75));
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_32, packed_word_33,
                             packed_word_66, packed_word_67);
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_70 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_70),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                         select_half_operand<2, false, false>(packed_word_75));
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_34, packed_word_35,
                             packed_word_62, packed_word_63);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x800ull, packed_word_40, packed_word_41,
        packed_word_42, packed_word_43, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_30, packed_word_31, 0u,
                             0u);
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_74));
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_28, packed_word_29, 0u,
                             0u);
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_74));
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x44004400u);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_32, packed_word_33,
                             packed_word_78, packed_word_79);
    packed_word_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0xc400c400u);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_34, packed_word_35,
                             packed_word_76, packed_word_77);
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x44004400u);
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x44004400u);
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0xc400c400u);
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_32, packed_word_33,
                             packed_word_72, packed_word_73);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_106),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_34, packed_word_35,
                             packed_word_70, packed_word_71);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x44004400u);
    packed_word_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_28, packed_word_29, 0u, 0u);
    packed_word_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0xc400c400u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_54),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_30, packed_word_31, 0u, 0u);
    packed_word_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x44004400u);
    packed_word_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    packed_word_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_106),
        select_half_operand<2, false, false>(packed_word_107), 0x3b283b28u);
    packed_word_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x44004400u);
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_32, packed_word_33,
                             packed_word_68, packed_word_69);
    packed_word_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0xc400c400u);
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0xc400c400u);
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_103),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0xc400c400u);
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_106),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_34, packed_word_35,
                             packed_word_74, packed_word_75);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_54),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0xa00ull, packed_word_44, packed_word_45,
        packed_word_46, packed_word_47, memory_ranges, error_counts);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_28, packed_word_29, 0u,
                             0u);
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_103),
        select_half_operand<2, false, false>(packed_word_104), 0x3b283b28u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_106),
        select_half_operand<2, false, false>(packed_word_108), 0x3b283b28u);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_30, packed_word_31, 0u,
                             0u);
    packed_word_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    packed_word_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x44004400u);
    packed_word_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x44004400u);
    packed_word_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0x44004400u);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x44004400u);
    packed_word_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0xc400c400u);
    packed_word_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0xc400c400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0xc400c400u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_54),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_113),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_111), 0x3b283b28u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_108),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<2, false, false>(packed_word_55), 0x3b283b28u);
    packed_word_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x44004400u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_113),
        select_half_operand<2, false, false>(packed_word_114), 0x3b283b28u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x44004400u);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x44004400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<2, false, false>(packed_word_112), 0x3b283b28u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0xc400c400u);
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_108),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_117), 0xc400c400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                          select_half_operand<2, false, false>(packed_word_104));
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                          select_half_operand<2, false, false>(packed_word_107));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_103));
    packed_word_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                          select_half_operand<2, false, false>(packed_word_109));
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                         select_half_operand<2, false, false>(packed_word_119));
    packed_word_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_29), 0x44004400u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_120),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_16, packed_word_17, 0u, 0u);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                         select_half_operand<2, false, false>(packed_word_113));
    packed_word_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_28), 0x44004400u);
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                          select_half_operand<2, false, false>(packed_word_105));
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                          select_half_operand<2, false, false>(packed_word_106));
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0x44004400u);
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_117));
    packed_word_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0xc400c400u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_120),
        select_half_operand<2, false, false>(packed_word_121), 0x3b283b28u);
    packed_word_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0xc400c400u);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_18, packed_word_19, 0u, 0u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                          select_half_operand<2, false, false>(packed_word_121));
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_35),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_34),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_35),
        select_half_operand<2, false, false>(packed_word_33), 0x3b283b28u);
    packed_word_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_30), 0x44004400u);
    packed_word_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_34),
        select_half_operand<2, false, false>(packed_word_32), 0x3b283b28u);
    packed_word_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_61), 0x44004400u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_110));
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_31), 0x44004400u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0xc400c400u);
    packed_word_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_112));
    packed_word_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0xc400c400u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_34),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_16, packed_word_17, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_112),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_34),
        select_half_operand<2, false, false>(packed_word_35), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_112),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x44004400u);
    packed_word_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_34), 0x3b283b28u);
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    packed_word_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_112), 0xc400c400u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x44004400u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_112),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_112),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_115),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_53));
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_115),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_53),
                                   select_half_operand<2, false, false>(0u));
    packed_word_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                         select_half_operand<2, false, false>(packed_word_32));
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                   select_half_operand<2, false, false>(0u));
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(0u));
    packed_word_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_35));
    packed_word_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0x44004400u);
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(0u));
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                         select_half_operand<2, false, false>(packed_word_52));
    packed_word_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                    select_half_operand<2, false, false>(0u));
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_34));
    packed_word_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_28),
                                   select_half_operand<2, false, false>(packed_word_53));
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(packed_word_49));
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_118), 0xc400c400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_51),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                    select_half_operand<2, false, false>(0u));
    packed_word_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                   select_half_operand<2, false, false>(packed_word_113));
    packed_word_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_18), 0x44004400u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_19), 0x44004400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_118), 0x3b283b28u);
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_103));
    packed_word_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                    select_half_operand<2, false, false>(0u));
    packed_word_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0xc400c400u);
    packed_word_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_102),
                                   select_half_operand<2, false, false>(0u));
    packed_word_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_17), 0x44004400u);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_60), 0xc400c400u);
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(packed_word_114));
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_62),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                   select_half_operand<2, false, false>(packed_word_31));
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_98),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(0u));
    packed_word_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                   select_half_operand<2, false, false>(packed_word_61));
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_62),
        select_half_operand<2, false, false>(packed_word_63), 0x3b283b28u);
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_60),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_98),
        select_half_operand<2, false, false>(packed_word_100), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                         select_half_operand<2, false, false>(packed_word_63));
    packed_word_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_60),
        select_half_operand<2, false, false>(packed_word_61), 0x3b283b28u);
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_19),
                                          select_half_operand<2, false, false>(packed_word_100));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(0u));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                         select_half_operand<2, false, false>(packed_word_61));
    packed_word_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                    select_half_operand<2, false, false>(0u));
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                         select_half_operand<2, false, false>(packed_word_33));
    packed_word_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                    select_half_operand<2, false, false>(0u));
    packed_word_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                    select_half_operand<2, false, false>(0u));
    packed_word_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                   select_half_operand<2, false, false>(0u));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(packed_word_63));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(packed_word_104));
    packed_word_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_17),
                                   select_half_operand<2, false, false>(packed_word_100));
    packed_word_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(packed_word_108));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1a00ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    packed_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                   select_half_operand<2, false, false>(packed_word_30));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1800ull, packed_word_28, packed_word_29,
        packed_word_30, packed_word_31, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_26, packed_word_27,
                             packed_word_84, packed_word_85);
    packed_word_118 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                    select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_32, packed_word_33,
                             packed_word_34, packed_word_35, packed_word_24, packed_word_25,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_2, packed_word_3, packed_word_32, packed_word_33,
                             packed_word_34, packed_word_35, packed_word_26, packed_word_27,
                             packed_word_2, packed_word_3);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_32, packed_word_33,
                             packed_word_34, packed_word_35, packed_word_36, packed_word_37,
                             packed_word_78, packed_word_79);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_32, packed_word_33,
                             packed_word_34, packed_word_35, packed_word_38, packed_word_39,
                             packed_word_76, packed_word_77);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0xe00ull, packed_word_32, packed_word_33,
        packed_word_34, packed_word_35, memory_ranges, error_counts);
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(packed_word_118));
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_24, packed_word_25,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_26, packed_word_27,
                             packed_word_58, packed_word_59);
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_36, packed_word_37,
                             packed_word_66, packed_word_67);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_38, packed_word_39,
                             packed_word_86, packed_word_87);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0xc00ull, packed_word_48, packed_word_49,
        packed_word_50, packed_word_51, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_24, packed_word_25,
                             packed_word_94, packed_word_95);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_36, packed_word_37,
                             packed_word_68, packed_word_69);
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_60, packed_word_61,
                             packed_word_62, packed_word_63, packed_word_38, packed_word_39,
                             packed_word_74, packed_word_75);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_40, packed_word_41, 0u,
                             0u);
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_42, packed_word_43, 0u,
                             0u);
    multiply_accumulate_half(packed_word_64, packed_word_65, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_26, packed_word_27,
                             packed_word_64, packed_word_65);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_24, packed_word_25,
                             packed_word_88, packed_word_89);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_36, packed_word_37,
                             packed_word_72, packed_word_73);
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_42, packed_word_43, 0u, 0u);
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_40, packed_word_41, 0u, 0u);
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_40, packed_word_41, 0u,
                             0u);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_38, packed_word_39,
                             packed_word_70, packed_word_71);
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_69), 0x44004400u);
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_40, packed_word_41, 0u,
                             0u);
    packed_word_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_74), 0x44004400u);
    packed_word_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_75), 0x44004400u);
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0xc400c400u);
    packed_word_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0xc400c400u);
    packed_word_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_52),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_42, packed_word_43, 0u,
                             0u);
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_68), 0x44004400u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0xc400c400u);
    packed_word_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_76),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0x44004400u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_79),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x44004400u);
    packed_word_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<2, false, false>(packed_word_53), 0x3b283b28u);
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_72), 0x44004400u);
    packed_word_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_76),
        select_half_operand<2, false, false>(packed_word_77), 0x3b283b28u);
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_73), 0x44004400u);
    packed_word_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_79),
        select_half_operand<2, false, false>(packed_word_87), 0x3b283b28u);
    packed_word_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    packed_word_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0xc400c400u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_108),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0x44004400u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<2, false, false>(packed_word_109), 0x3b283b28u);
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_44, packed_word_45, 0u, 0u);
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_44, packed_word_45, 0u,
                             0u);
    packed_word_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_76),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_79),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_76),
        select_half_operand<2, false, false>(packed_word_44), 0x3b283b28u);
    packed_word_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_70), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_79),
        select_half_operand<2, false, false>(packed_word_45), 0x3b283b28u);
    packed_word_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_71), 0x44004400u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0xc400c400u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0xc400c400u);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_108),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<2, false, false>(packed_word_109), 0x3b283b28u);
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_76),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0x44004400u);
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_111), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_76),
        select_half_operand<2, false, false>(packed_word_79), 0x3b283b28u);
    packed_word_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0xc400c400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0x44004400u);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0x44004400u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_79), 0xc400c400u);
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_76),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_74),
                                         select_half_operand<2, false, false>(packed_word_78));
    packed_word_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                         select_half_operand<2, false, false>(packed_word_113));
    packed_word_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x44004400u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0xc400c400u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_76),
        select_half_operand<2, false, false>(packed_word_79), 0x3b283b28u);
    packed_word_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_70),
                                         select_half_operand<2, false, false>(packed_word_121));
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_86 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_75),
                                         select_half_operand<2, false, false>(packed_word_86));
    packed_word_70 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_110));
    packed_word_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x44004400u);
    packed_word_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x44004400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_117),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_77));
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_112), 0x3b283b28u);
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_71),
                                         select_half_operand<2, false, false>(packed_word_126));
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0xc400c400u);
    packed_word_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                         select_half_operand<2, false, false>(packed_word_115));
    packed_word_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_26), 0xc400c400u);
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_87));
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_24),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_46, packed_word_47, 0u, 0u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_117),
        select_half_operand<2, false, false>(packed_word_118), 0x3b283b28u);
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_108));
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_109));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_26),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_46, packed_word_47, 0u,
                             0u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_111), 0x3b283b28u);
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x44004400u);
    packed_word_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_24),
        select_half_operand<2, false, false>(packed_word_25), 0x3b283b28u);
    packed_word_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x44004400u);
    packed_word_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_26),
        select_half_operand<2, false, false>(packed_word_27), 0x3b283b28u);
    packed_word_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x44004400u);
    packed_word_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0xc400c400u);
    packed_word_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x44004400u);
    packed_word_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x44004400u);
    packed_word_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_25), 0xc400c400u);
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_24),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_27), 0xc400c400u);
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_112), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_24),
        select_half_operand<2, false, false>(packed_word_25), 0x3b283b28u);
    packed_word_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_25), 0x3b283b28u);
    packed_word_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_24), 0x3b283b28u);
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_26),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_68), 0x44004400u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0xc400c400u);
    packed_word_72 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_72),
                                         select_half_operand<2, false, false>(packed_word_119));
    packed_word_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_73 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_73),
                                         select_half_operand<2, false, false>(packed_word_120));
    packed_word_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_117), 0xc400c400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_26),
        select_half_operand<2, false, false>(packed_word_27), 0x3b283b28u);
    packed_word_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_69), 0x44004400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                          select_half_operand<2, false, false>(packed_word_114));
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_70),
                                   select_half_operand<2, false, false>(0u));
    packed_word_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                    select_half_operand<2, false, false>(0u));
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_118), 0xc400c400u);
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_126 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x44004400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_26), 0x3b283b28u);
    packed_word_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_120), 0xc400c400u);
    packed_word_36 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(packed_word_37));
    packed_word_75 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_75),
                                   select_half_operand<2, false, false>(0u));
    packed_word_79 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(0u));
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_108));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_119),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_127 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_126), 0xc400c400u);
    packed_word_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_121),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_74),
                                   select_half_operand<2, false, false>(packed_word_75));
    packed_word_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                          select_half_operand<2, false, false>(packed_word_116));
    packed_word_44 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_79));
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0x44004400u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1c00ull, packed_word_24, packed_word_25,
        packed_word_26, packed_word_27, memory_ranges, error_counts);
    packed_word_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_62), 0xc400c400u);
    packed_word_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_63), 0xc400c400u);
    packed_word_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_119),
        select_half_operand<2, false, false>(packed_word_120), 0x3b283b28u);
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(0u));
    packed_word_126 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_121),
        select_half_operand<2, false, false>(packed_word_126), 0x3b283b28u);
    packed_word_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0xc400c400u);
    packed_word_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_74),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                          select_half_operand<2, false, false>(packed_word_118));
    packed_word_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_71),
                                   select_half_operand<2, false, false>(packed_word_42));
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_78),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                          select_half_operand<2, false, false>(packed_word_120));
    packed_word_126 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                          select_half_operand<2, false, false>(packed_word_126));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x1e00ull, packed_word_68, packed_word_69,
        packed_word_70, packed_word_71, memory_ranges, error_counts);
    packed_word_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_127),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_62),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_74),
        select_half_operand<2, false, false>(packed_word_75), 0x3b283b28u);
    packed_word_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_78),
        select_half_operand<2, false, false>(packed_word_79), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_127),
        select_half_operand<2, false, false>(packed_word_128), 0x3b283b28u);
    packed_word_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_62),
        select_half_operand<2, false, false>(packed_word_63), 0x3b283b28u);
    packed_word_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_75));
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                         select_half_operand<2, false, false>(packed_word_79));
    packed_word_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_54),
                                   select_half_operand<2, false, false>(0u));
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_46));
    packed_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                   select_half_operand<2, false, false>(0u));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_47));
    packed_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_116),
                                   select_half_operand<2, false, false>(0u));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_63));
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(0u));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                   select_half_operand<2, false, false>(0u));
    packed_word_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                    select_half_operand<2, false, false>(0u));
    packed_word_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(0u));
    packed_word_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                    select_half_operand<2, false, false>(0u));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_126),
                                   select_half_operand<2, false, false>(0u));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                   select_half_operand<2, false, false>(0u));
    packed_word_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(0u));
    packed_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(packed_word_38));
    packed_word_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                   select_half_operand<2, false, false>(packed_word_39));
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_72),
                                   select_half_operand<2, false, false>(packed_word_53));
    packed_word_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(packed_word_55));
    packed_word_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(packed_word_45));
    packed_word_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                   select_half_operand<2, false, false>(packed_word_111));
    packed_word_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(packed_word_112));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                   select_half_operand<2, false, false>(packed_word_120));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_73),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(packed_word_76));
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_28, packed_word_29,
                             packed_word_88, packed_word_89);
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_28, packed_word_29,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_28, packed_word_29,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_28, packed_word_29,
                             packed_word_94, packed_word_95);
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_30, packed_word_31,
                             packed_word_64, packed_word_65);
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_16, packed_word_17,
                             packed_word_100, packed_word_101);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_36, packed_word_37,
                             packed_word_38, packed_word_39, packed_word_18, packed_word_19,
                             packed_word_102, packed_word_103);
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_30, packed_word_31,
                             packed_word_58, packed_word_59);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_30, packed_word_31,
                             packed_word_96, packed_word_97);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_34, packed_word_35, 0u,
                             0u);
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_32, packed_word_33, 0u,
                             0u);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_34, packed_word_35, 0u,
                             0u);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_16, packed_word_17,
                             packed_word_66, packed_word_67);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2660ull, packed_word_64, packed_word_65,
        packed_word_66, packed_word_67, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_40, packed_word_41,
                             packed_word_42, packed_word_43, packed_word_18, packed_word_19,
                             packed_word_104, packed_word_105);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_16, packed_word_17,
                             packed_word_106, packed_word_107);
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_52, packed_word_53,
                             packed_word_54, packed_word_55, packed_word_18, packed_word_19,
                             packed_word_60, packed_word_61);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2460ull, packed_word_60, packed_word_61,
        packed_word_62, packed_word_63, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_32, packed_word_33, 0u,
                             0u);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_34, packed_word_35, 0u,
                             0u);
    multiply_accumulate_half(packed_word_2, packed_word_3, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_30, packed_word_31,
                             packed_word_2, packed_word_3);
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_48, packed_word_49, 0u,
                             0u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_50, packed_word_51, 0u,
                             0u);
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_48, packed_word_49, 0u,
                             0u);
    multiply_accumulate_half(packed_word_20, packed_word_21, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_50, packed_word_51, 0u,
                             0u);
    packed_word_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x44004400u);
    packed_word_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x44004400u);
    packed_word_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_58), 0x44004400u);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x44004400u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0xc400c400u);
    packed_word_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0xc400c400u);
    packed_word_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0xc400c400u);
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0xc400c400u);
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_16, packed_word_17,
                             packed_word_84, packed_word_85);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_8),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_9),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_8),
        select_half_operand<2, false, false>(packed_word_43), 0x3b283b28u);
    packed_word_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_59), 0x44004400u);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_44, packed_word_45,
                             packed_word_46, packed_word_47, packed_word_18, packed_word_19,
                             packed_word_98, packed_word_99);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2860ull, packed_word_16, packed_word_17,
        packed_word_18, packed_word_19, memory_ranges, error_counts);
    packed_word_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_11),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_42),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_11),
        select_half_operand<2, false, false>(packed_word_44), 0x3b283b28u);
    packed_word_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_9),
        select_half_operand<2, false, false>(packed_word_10), 0x3b283b28u);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_54), 0x44004400u);
    packed_word_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x44004400u);
    packed_word_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<2, false, false>(packed_word_45), 0x3b283b28u);
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_32, packed_word_33, 0u, 0u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0xc400c400u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0xc400c400u);
    packed_word_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0xc400c400u);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_32, packed_word_33, 0u,
                             0u);
    packed_word_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0xc400c400u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2a60ull, packed_word_8, packed_word_9,
        packed_word_10, packed_word_11, memory_ranges, error_counts);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_34, packed_word_35, 0u, 0u);
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_32),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_100),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_109),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_32),
        select_half_operand<2, false, false>(packed_word_33), 0x3b283b28u);
    packed_word_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<2, false, false>(packed_word_101), 0x3b283b28u);
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_109),
        select_half_operand<2, false, false>(packed_word_110), 0x3b283b28u);
    packed_word_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_53), 0x44004400u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0xc400c400u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0xc400c400u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_100),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x44004400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<2, false, false>(packed_word_109), 0x3b283b28u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_34), 0x44004400u);
    packed_word_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_35), 0x44004400u);
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_112), 0x3b283b28u);
    packed_word_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0xc400c400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0xc400c400u);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_48, packed_word_49, 0u, 0u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x44004400u);
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                          select_half_operand<2, false, false>(packed_word_100));
    packed_word_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0xc400c400u);
    multiply_accumulate_half(packed_word_4, packed_word_5, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_50, packed_word_51, 0u, 0u);
    packed_word_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_109),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_109),
        select_half_operand<2, false, false>(packed_word_6), 0x3b283b28u);
    packed_word_6 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                        select_half_operand<2, false, false>(packed_word_43));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_58),
                                         select_half_operand<2, false, false>(packed_word_99));
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_112));
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_48, packed_word_49, 0u,
                             0u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_118),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_30), 0x44004400u);
    packed_word_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_118),
        select_half_operand<2, false, false>(packed_word_119), 0x3b283b28u);
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_119));
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_50, packed_word_51, 0u,
                             0u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_48),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0x44004400u);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_113), 0x3b283b28u);
    packed_word_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x44004400u);
    packed_word_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<2, false, false>(packed_word_38), 0x3b283b28u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x44004400u);
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_59),
                                          select_half_operand<2, false, false>(packed_word_108));
    packed_word_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_22), 0x44004400u);
    packed_word_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_109));
    packed_word_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0xc400c400u);
    packed_word_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_23), 0x44004400u);
    packed_word_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_21), 0x44004400u);
    packed_word_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_58), 0xc400c400u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                         select_half_operand<2, false, false>(packed_word_110));
    packed_word_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0xc400c400u);
    packed_word_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_14),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0xc400c400u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0xc400c400u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_59),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x44004400u);
    packed_word_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_111),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_14),
        select_half_operand<2, false, false>(packed_word_48), 0x3b283b28u);
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_58),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_59),
        select_half_operand<2, false, false>(packed_word_109), 0x3b283b28u);
    packed_word_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_32), 0x44004400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<2, false, false>(packed_word_7), 0x3b283b28u);
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_31), 0x44004400u);
    packed_word_7 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                        select_half_operand<2, false, false>(packed_word_113));
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_114), 0xc400c400u);
    packed_word_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_58),
        select_half_operand<2, false, false>(packed_word_14), 0x3b283b28u);
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x44004400u);
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_42));
    packed_word_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x44004400u);
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_101));
    packed_word_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_59), 0xc400c400u);
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                          select_half_operand<2, false, false>(packed_word_117));
    packed_word_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0xc400c400u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0xc400c400u);
    packed_word_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                         select_half_operand<2, false, false>(packed_word_115));
    packed_word_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0xc400c400u);
    packed_word_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_111));
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_5), 0x44004400u);
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_59),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_49),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_116), 0x3b283b28u);
    packed_word_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0xc400c400u);
    packed_word_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_53),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x44004400u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_59),
        select_half_operand<2, false, false>(packed_word_14), 0x3b283b28u);
    packed_word_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_49),
        select_half_operand<2, false, false>(packed_word_52), 0x3b283b28u);
    packed_word_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_112),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_116));
    packed_word_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_113), 0xc400c400u);
    packed_word_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_53),
        select_half_operand<2, false, false>(packed_word_54), 0x3b283b28u);
    packed_word_49 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_49));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2c60ull, packed_word_52, packed_word_53,
        packed_word_54, packed_word_55, memory_ranges, error_counts);
    packed_word_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_112),
        select_half_operand<2, false, false>(packed_word_59), 0x3b283b28u);
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                         select_half_operand<2, false, false>(packed_word_58));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_56, packed_word_57) + 0x2e60ull, packed_word_56, packed_word_57,
        packed_word_58, packed_word_59, memory_ranges, error_counts);
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_50),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_114),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_50),
        select_half_operand<2, false, false>(packed_word_51), 0x3b283b28u);
    packed_word_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_110),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_48));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_20),
                                          select_half_operand<2, false, false>(packed_word_109));
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<2, false, false>(packed_word_50), 0x3b283b28u);
    packed_word_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_114),
        select_half_operand<2, false, false>(packed_word_115), 0x3b283b28u);
    packed_word_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_48),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                         select_half_operand<2, false, false>(packed_word_15));
    packed_word_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_33), 0x44004400u);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_38));
    packed_word_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                    select_half_operand<2, false, false>(0u));
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                         select_half_operand<2, false, false>(packed_word_51));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_21),
                                          select_half_operand<2, false, false>(packed_word_110));
    packed_word_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                          select_half_operand<2, false, false>(packed_word_115));
    packed_word_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x44004400u);
    packed_word_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_49),
                                   select_half_operand<2, false, false>(packed_word_20));
    packed_word_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0xc400c400u);
    packed_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                   select_half_operand<2, false, false>(0u));
    packed_word_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                   select_half_operand<2, false, false>(packed_word_109));
    packed_word_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                    select_half_operand<2, false, false>(0u));
    packed_word_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_5), 0xc400c400u);
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_50),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_38));
    packed_word_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_4),
                                          select_half_operand<2, false, false>(packed_word_113));
    packed_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                   select_half_operand<2, false, false>(packed_word_110));
    packed_word_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_30),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x44004400u);
    packed_word_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x44004400u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                          select_half_operand<2, false, false>(packed_word_111));
    packed_word_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0xc400c400u);
    packed_word_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_50),
        select_half_operand<2, false, false>(packed_word_14), 0x3b283b28u);
    packed_word_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0xc400c400u);
    packed_word_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_116),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_30),
        select_half_operand<2, false, false>(packed_word_31), 0x3b283b28u);
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(0u));
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                          select_half_operand<2, false, false>(packed_word_112));
    packed_word_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                    select_half_operand<2, false, false>(0u));
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_116),
        select_half_operand<2, false, false>(packed_word_117), 0x3b283b28u);
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_32),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                         select_half_operand<2, false, false>(packed_word_31));
    packed_word_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_word_15),
        select_half_operand<0, false, true>(packed_word_0), 0x37283728u);
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                   select_half_operand<2, false, false>(packed_word_43));
    packed_word_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_14));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(packed_word_108));
    packed_word_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_15),
        select_half_operand<2, false, false>(packed_word_0), 0x3b283b28u);
    packed_word_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                        select_half_operand<2, false, false>(packed_word_0));
    packed_word_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_70, packed_word_71,
                             packed_word_46, packed_word_47);
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_40),
                                   select_half_operand<2, false, false>(packed_word_31));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_42));
    packed_word_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_32),
        select_half_operand<2, false, false>(packed_word_33), 0x3b283b28u);
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_26, packed_word_27,
                             packed_word_2, packed_word_3);
    packed_word_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                   select_half_operand<2, false, false>(0u));
    packed_word_33 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_13),
                                         select_half_operand<2, false, false>(packed_word_33));
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(0u));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                   select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                  select_half_operand<2, false, false>(packed_word_6));
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_24, packed_word_25,
                             packed_word_92, packed_word_93);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                  select_half_operand<2, false, false>(packed_word_35));
    packed_word_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                   select_half_operand<2, false, false>(packed_word_22));
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_23));
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_48, packed_word_49,
                             packed_word_50, packed_word_51, packed_word_68, packed_word_69,
                             packed_word_84, packed_word_85);
    packed_word_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                   select_half_operand<2, false, false>(0u));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                  select_half_operand<2, false, false>(0u));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_33),
                                   select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                  select_half_operand<2, false, false>(packed_word_34));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                  select_half_operand<2, false, false>(packed_word_5));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_13));
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_70, packed_word_71,
                             packed_word_102, packed_word_103);
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_40),
                                  select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_41),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_26, packed_word_27,
                             packed_word_72, packed_word_73);
    packed_word_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_37),
                                   select_half_operand<2, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_26, packed_word_27,
                             packed_word_28, packed_word_29);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_68, packed_word_69,
                             packed_word_88, packed_word_89);
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_20, packed_word_21,
                             packed_word_22, packed_word_23, packed_word_24, packed_word_25,
                             packed_word_86, packed_word_87);
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_24, packed_word_25,
                             packed_word_74, packed_word_75);
    multiply_accumulate_half(packed_word_32, packed_word_33, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_24, packed_word_25,
                             packed_word_76, packed_word_77);
    multiply_accumulate_half(packed_word_28, packed_word_29, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_26, packed_word_27,
                             packed_word_78, packed_word_79);
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_68, packed_word_69,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_24, packed_word_25, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_70, packed_word_71,
                             packed_word_104, packed_word_105);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_46),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_20, packed_word_21, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_70, packed_word_71,
                             packed_word_94, packed_word_95);
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_22, packed_word_23, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_68, packed_word_69,
                             packed_word_106, packed_word_107);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                  select_half_operand<2, false, false>(0u));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                  select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_146, packed_word_147, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_64, packed_word_65, 0u,
                             0u);
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_51),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_66, packed_word_67, 0u,
                             0u);
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_48),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                  select_half_operand<2, false, false>(packed_word_3));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                  select_half_operand<2, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_60, packed_word_61, 0u,
                             0u);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_49),
                                  select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_62, packed_word_63, 0u,
                             0u);
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                  select_half_operand<2, false, false>(0u));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_24),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_25),
                                  select_half_operand<2, false, false>(0u));
    packed_word_72 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_73 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_74 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_26),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_75 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_27),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_28),
                                  select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                  select_half_operand<2, false, false>(0u));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_20),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_21),
                                  select_half_operand<2, false, false>(0u));
    packed_word_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_32),
                                   select_half_operand<2, false, false>(packed_word_3));
    packed_word_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_33),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_22),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_23),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_146),
                                        select_half_operand<2, false, false>(packed_word_146));
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                        select_half_operand<2, false, false>(packed_word_106));
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_18, packed_word_19, 0u,
                             0u);
    packed_word_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                      select_half_operand<2, false, false>(packed_word_84),
                                      select_half_operand<2, false, false>(packed_word_0));
    packed_word_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_108),
                                      select_half_operand<2, false, false>(packed_word_108),
                                      select_half_operand<2, false, false>(packed_word_3));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_147),
                                        select_half_operand<2, false, false>(packed_word_147));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                      select_half_operand<2, false, false>(packed_word_85),
                                      select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_152, packed_word_153, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_18, packed_word_19, 0u, 0u);
    multiply_accumulate_half(packed_word_120, packed_word_121, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_18, packed_word_19, 0u,
                             0u);
    multiply_accumulate_half(packed_word_140, packed_word_141, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_18, packed_word_19, 0u,
                             0u);
    multiply_accumulate_half(packed_word_132, packed_word_133, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_64, packed_word_65, 0u, 0u);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_8, packed_word_9, 0u, 0u);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_8, packed_word_9, 0u, 0u);
    multiply_accumulate_half(packed_word_148, packed_word_149, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_8, packed_word_9, 0u, 0u);
    multiply_accumulate_half(packed_word_130, packed_word_131, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_8, packed_word_9, 0u, 0u);
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                        select_half_operand<2, false, false>(packed_word_107));
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_136, packed_word_137, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_66, packed_word_67, 0u, 0u);
    packed_word_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_109),
                                      select_half_operand<2, false, false>(packed_word_109),
                                      select_half_operand<2, false, false>(packed_word_8));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_9), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_86 = loaded_word;
    }
    multiply_accumulate_half(packed_word_134, packed_word_135, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_60, packed_word_61, 0u, 0u);
    multiply_accumulate_half(packed_word_138, packed_word_139, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_62, packed_word_63, 0u, 0u);
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_16, packed_word_17, 0u,
                             0u);
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_16, packed_word_17, 0u, 0u);
    multiply_accumulate_half(packed_word_118, packed_word_119, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_16, packed_word_17, 0u,
                             0u);
    multiply_accumulate_half(packed_word_142, packed_word_143, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_16, packed_word_17, 0u,
                             0u);
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                    select_half_operand<2, false, false>(packed_word_2));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_16), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_17 = loaded_word;
    }
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                        select_half_operand<2, false, false>(packed_word_132));
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_10, packed_word_11, 0u,
                             0u);
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                        select_half_operand<2, false, false>(packed_word_136));
    packed_word_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                      select_half_operand<2, false, false>(packed_word_134),
                                      select_half_operand<2, false, false>(packed_word_0));
    packed_word_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                      select_half_operand<2, false, false>(packed_word_138),
                                      select_half_operand<2, false, false>(packed_word_3));
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                   select_half_operand<2, false, false>(packed_word_86));
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                        select_half_operand<2, false, false>(packed_word_137));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                        select_half_operand<2, false, false>(packed_word_133));
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_0));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_0 = loaded_word;
    }
    packed_word_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                      select_half_operand<2, false, false>(packed_word_139),
                                      select_half_operand<2, false, false>(packed_word_8));
    packed_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                      select_half_operand<2, false, false>(packed_word_135),
                                      select_half_operand<2, false, false>(packed_word_2));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                    select_half_operand<2, false, false>(packed_word_17));
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_2));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_16), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_17 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_8 = loaded_word;
    }
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_10, packed_word_11, 0u, 0u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_87 = loaded_word;
    }
    packed_word_0 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                   select_half_operand<2, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_150, packed_word_151, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_10, packed_word_11, 0u,
                             0u);
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                    select_half_operand<2, false, false>(packed_word_17));
    multiply_accumulate_half(packed_word_144, packed_word_145, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_10, packed_word_11, 0u,
                             0u);
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                         select_half_operand<2, false, false>(packed_word_14));
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_110),
                                         select_half_operand<2, false, false>(packed_word_110));
    packed_word_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                      select_half_operand<2, false, false>(packed_word_12),
                                      select_half_operand<2, false, false>(packed_word_10));
    packed_word_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_96),
                                      select_half_operand<2, false, false>(packed_word_11));
    packed_word_98 = add_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                    select_half_operand<2, false, false>(packed_word_10));
    packed_word_11 = permute_signed(packed_word_0, packed_word_0, 0x5432u);
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_8));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_99 = loaded_word;
    }
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                        select_half_operand<2, false, false>(packed_word_15));
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_111),
                                         select_half_operand<2, false, false>(packed_word_111));
    packed_word_0 = add_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_11 = permute_signed(packed_word_16, packed_word_16, 0x5432u);
    packed_word_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_13),
                                      select_half_operand<2, false, false>(packed_word_13),
                                      select_half_operand<2, false, false>(packed_word_8));
    packed_word_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                      select_half_operand<2, false, false>(packed_word_97),
                                      select_half_operand<2, false, false>(packed_word_10));
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                    select_half_operand<2, false, false>(packed_word_9));
    packed_word_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                        select_half_operand<2, false, false>(packed_word_18));
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                         select_half_operand<2, false, false>(packed_word_76));
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_87));
    packed_word_87 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                         select_half_operand<2, false, false>(packed_word_150));
    packed_word_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                      select_half_operand<2, false, false>(packed_word_78),
                                      select_half_operand<2, false, false>(packed_word_9));
    packed_word_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_152),
                                      select_half_operand<2, false, false>(packed_word_152),
                                      select_half_operand<2, false, false>(packed_word_11));
    packed_word_91 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_120),
                                      select_half_operand<2, false, false>(packed_word_120),
                                      select_half_operand<2, false, false>(packed_word_87));
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_0), 0x4100410u);
    packed_word_86 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_148),
                                         select_half_operand<2, false, false>(packed_word_148));
    packed_word_87 = add_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                    select_half_operand<2, false, false>(packed_word_90));
    packed_word_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x4100410u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_100 = loaded_word;
    }
    packed_word_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_118),
                                      select_half_operand<2, false, false>(packed_word_118),
                                      select_half_operand<2, false, false>(packed_word_86));
    packed_word_101 = half_to_float_bits((packed_word_9 & 65535u) * 65537u);
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(packed_word_99));
    packed_word_98 = half_to_float_bits((packed_word_11 & 65535u) * 65537u);
    packed_word_105 = half_to_float_bits((packed_word_11 >> 16) * 65537u);
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_149),
                                         select_half_operand<2, false, false>(packed_word_149));
    packed_word_9 = half_to_float_bits((packed_word_9 >> 16) * 65537u);
    packed_word_0 = add_half_pairs(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(packed_word_86));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_2), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_99 = loaded_word;
    }
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_19),
                                         select_half_operand<2, false, false>(packed_word_19));
    packed_word_86 = rsqrt_bits(packed_word_9);
    packed_word_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_77));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_16), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_103 = loaded_word;
    }
    packed_word_113 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_119),
                                      select_half_operand<2, false, false>(packed_word_119),
                                      select_half_operand<2, false, false>(packed_word_11));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_87), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_102 = loaded_word;
    }
    packed_word_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                      select_half_operand<2, false, false>(packed_word_79),
                                      select_half_operand<2, false, false>(packed_word_10));
    packed_word_101 = rsqrt_bits(packed_word_101);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_9 = loaded_word;
    }
    packed_word_17 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_153),
                                      select_half_operand<2, false, false>(packed_word_153),
                                      select_half_operand<2, false, false>(packed_word_17));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_8), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_11 = loaded_word;
    }
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                    select_half_operand<2, false, false>(packed_word_10));
    packed_word_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_151),
                                         select_half_operand<2, false, false>(packed_word_151));
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                          select_half_operand<2, false, false>(packed_word_130));
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_144),
                                          select_half_operand<2, false, false>(packed_word_144));
    packed_word_104 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_142),
                                      select_half_operand<2, false, false>(packed_word_142),
                                      select_half_operand<2, false, false>(packed_word_104));
    packed_word_112 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                      select_half_operand<2, false, false>(packed_word_140),
                                      select_half_operand<2, false, false>(packed_word_112));
    packed_word_114 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_121),
                                      select_half_operand<2, false, false>(packed_word_121),
                                      select_half_operand<2, false, false>(packed_word_17));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_10), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_17 = loaded_word;
    }
    packed_word_3 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_100));
    packed_word_100 = add_half_pairs(select_half_operand<2, false, false>(packed_word_112),
                                     select_half_operand<2, false, false>(packed_word_104));
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                          select_half_operand<2, false, false>(packed_word_131));
    packed_word_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_145),
                                          select_half_operand<2, false, false>(packed_word_145));
    packed_word_101 = pack_half(packed_word_86, packed_word_101);
    packed_word_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_143),
                                      select_half_operand<2, false, false>(packed_word_143),
                                      select_half_operand<2, false, false>(packed_word_104));
    packed_word_112 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                      select_half_operand<2, false, false>(packed_word_141),
                                      select_half_operand<2, false, false>(packed_word_112));
    packed_word_98 = rsqrt_bits(packed_word_98);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_99));
    packed_word_99 = add_half_pairs(select_half_operand<2, false, false>(packed_word_114),
                                    select_half_operand<2, false, false>(packed_word_113));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                    select_half_operand<2, false, false>(packed_word_103));
    packed_word_87 = add_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                    select_half_operand<2, false, false>(packed_word_102));
    packed_word_105 = rsqrt_bits(packed_word_105);
    packed_word_86 = add_half_pairs(select_half_operand<2, false, false>(packed_word_112),
                                    select_half_operand<2, false, false>(packed_word_86));
    packed_word_0 = add_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_102 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                     select_half_operand<2, false, false>(packed_word_11));
    packed_word_8 = parameters[56];
    packed_word_9 = parameters[57];
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_100), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_115 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_99), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_114 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_16), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_113 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_86), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_117 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_87), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_112 = loaded_word;
    }
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                    select_half_operand<2, false, false>(packed_word_17));
    packed_word_98 = pack_half(packed_word_105, packed_word_98);
    packed_word_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_84),
                                         select_half_operand<2, false, false>(packed_word_101));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_0), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_103 = loaded_word;
    }
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_108),
                                          select_half_operand<2, false, false>(packed_word_101));
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_146),
                                          select_half_operand<2, false, false>(packed_word_101));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_11 = loaded_word;
    }
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                          select_half_operand<2, false, false>(packed_word_101));
    packed_word_101 = permute_signed(packed_word_102, packed_word_102, 0x5432u);
    split_address_words(packed_word_160, packed_word_161,
                        int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_word_8, packed_word_9));
    packed_word_100 = add_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                     select_half_operand<2, false, false>(packed_word_115));
    packed_word_84 = add_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                    select_half_operand<2, false, false>(packed_word_101));
    packed_word_99 = add_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                    select_half_operand<2, false, false>(packed_word_114));
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_85),
                                          select_half_operand<2, false, false>(packed_word_98));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_109),
                                          select_half_operand<2, false, false>(packed_word_98));
    packed_word_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_147),
                                          select_half_operand<2, false, false>(packed_word_98));
    packed_word_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                          select_half_operand<2, false, false>(packed_word_98));
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_84), 0x4100410u);
    packed_word_113 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                     select_half_operand<2, false, false>(packed_word_113));
    packed_word_116 = add_half_pairs(select_half_operand<2, false, false>(packed_word_86),
                                     select_half_operand<2, false, false>(packed_word_117));
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_87),
                                    select_half_operand<2, false, false>(packed_word_112));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3060ull, packed_word_84, packed_word_85,
        packed_word_86, packed_word_87, memory_ranges, error_counts);
    packed_word_8 = permute_signed(packed_word_113, packed_word_113, 0x5432u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_100), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_101 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_99), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_102 = loaded_word;
    }
    multiply_accumulate_half(packed_word_88, packed_word_89, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_52, packed_word_53, 0u,
                             0u);
    packed_word_107 = add_half_pairs(select_half_operand<2, false, false>(packed_word_113),
                                     select_half_operand<2, false, false>(packed_word_8));
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                    select_half_operand<2, false, false>(packed_word_11));
    packed_word_9 = half_to_float_bits((packed_word_98 & 65535u) * 65537u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_116), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_11 = loaded_word;
    }
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_103));
    packed_word_0 = half_to_float_bits((packed_word_98 >> 16) * 65537u);
    multiply_accumulate_half(packed_word_126, packed_word_127, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_9 = rsqrt_bits(packed_word_9);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_107), 0x4100410u);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_0 = rsqrt_bits(packed_word_0);
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_92, packed_word_93,
                             packed_word_94, packed_word_95, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_107 = add_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                     select_half_operand<2, false, false>(packed_word_101));
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_52, packed_word_53, 0u, 0u);
    multiply_accumulate_half(packed_word_128, packed_word_129, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_54, packed_word_55, 0u, 0u);
    multiply_accumulate_half(packed_word_112, packed_word_113, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_56, packed_word_57, 0u, 0u);
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_58, packed_word_59, 0u, 0u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3260ull, packed_word_4, packed_word_5,
        packed_word_6, packed_word_7, memory_ranges, error_counts);
    packed_word_99 = add_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                    select_half_operand<2, false, false>(packed_word_102));
    packed_word_101 = permute_signed(packed_word_16, packed_word_16, 0x5432u);
    packed_word_102 = half_to_float_bits((packed_word_98 & 65535u) * 65537u);
    packed_word_98 = half_to_float_bits((packed_word_98 >> 16) * 65537u);
    packed_word_16 = add_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                    select_half_operand<2, false, false>(packed_word_101));
    packed_word_102 = rsqrt_bits(packed_word_102);
    packed_word_116 = add_half_pairs(select_half_operand<2, false, false>(packed_word_116),
                                     select_half_operand<2, false, false>(packed_word_11));
    packed_word_11 = permute_signed(packed_word_10, packed_word_10, 0x5432u);
    packed_word_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_16), 0x4100410u);
    packed_word_103 = rsqrt_bits(packed_word_98);
    packed_word_9 = pack_half(packed_word_0, packed_word_9);
    packed_word_100 = half_to_float_bits((packed_word_16 & 65535u) * 65537u);
    packed_word_117 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                     select_half_operand<2, false, false>(packed_word_11));
    packed_word_101 = half_to_float_bits((packed_word_16 >> 16) * 65537u);
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                        select_half_operand<2, false, false>(packed_word_9));
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_110),
                                         select_half_operand<2, false, false>(packed_word_9));
    packed_word_9 = permute_signed(packed_word_8, packed_word_8, 0x5432u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_117), 0x4100410u);
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                    select_half_operand<2, false, false>(packed_word_9));
    packed_word_12 = pack_half(packed_word_103, packed_word_102);
    packed_word_103 = half_to_float_bits((packed_word_117 & 65535u) * 65537u);
    packed_word_100 = rsqrt_bits(packed_word_100);
    packed_word_154 = half_to_float_bits((packed_word_117 >> 16) * 65537u);
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x4100410u);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3460ull, packed_word_8, packed_word_9,
        packed_word_10, packed_word_11, memory_ranges, error_counts);
    packed_word_102 = half_to_float_bits((packed_word_117 & 65535u) * 65537u);
    packed_word_101 = rsqrt_bits(packed_word_101);
    packed_word_117 = half_to_float_bits((packed_word_117 >> 16) * 65537u);
    packed_word_14 = permute_signed(packed_word_99, packed_word_99, 0x5432u);
    packed_word_103 = rsqrt_bits(packed_word_103);
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                    select_half_operand<2, false, false>(packed_word_14));
    packed_word_154 = rsqrt_bits(packed_word_154);
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_13),
                                         select_half_operand<2, false, false>(packed_word_12));
    packed_word_102 = rsqrt_bits(packed_word_102);
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                         select_half_operand<2, false, false>(packed_word_12));
    packed_word_117 = rsqrt_bits(packed_word_117);
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                          select_half_operand<2, false, false>(packed_word_12));
    packed_word_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x4100410u);
    packed_word_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_111),
                                          select_half_operand<2, false, false>(packed_word_12));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3660ull, packed_word_12, packed_word_13,
        packed_word_14, packed_word_15, memory_ranges, error_counts);
    packed_word_157 = pack_half(packed_word_101, packed_word_100);
    packed_word_154 = pack_half(packed_word_154, packed_word_103);
    packed_word_163 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_152),
                                          select_half_operand<2, false, false>(packed_word_157));
    packed_word_164 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                          select_half_operand<2, false, false>(packed_word_157));
    packed_word_162 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                          select_half_operand<2, false, false>(packed_word_157));
    packed_word_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                          select_half_operand<2, false, false>(packed_word_154));
    packed_word_102 = pack_half(packed_word_117, packed_word_102);
    packed_word_153 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_153),
                                          select_half_operand<2, false, false>(packed_word_154));
    packed_word_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_19),
                                          select_half_operand<2, false, false>(packed_word_154));
    packed_word_157 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                          select_half_operand<2, false, false>(packed_word_157));
    packed_word_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                          select_half_operand<2, false, false>(packed_word_154));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3860ull, packed_word_76, packed_word_77,
        packed_word_78, packed_word_79, memory_ranges, error_counts);
    packed_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(0u));
    packed_word_100 = half_to_float_bits((packed_word_155 & 65535u) * 65537u);
    packed_word_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_19));
    packed_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(0u));
    packed_word_101 = half_to_float_bits((packed_word_155 >> 16) * 65537u);
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                  select_half_operand<2, false, false>(0u));
    packed_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_16),
                                   select_half_operand<2, false, false>(packed_word_19));
    packed_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_0 = pack_half(0u, packed_word_158);
    packed_word_100 = rsqrt_bits(packed_word_100);
    packed_word_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_150),
                                          select_half_operand<2, false, false>(packed_word_102));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3a60ull, packed_word_96, packed_word_97,
        packed_word_98, packed_word_99, memory_ranges, error_counts);
    packed_word_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_101 = rsqrt_bits(packed_word_101);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(0u));
    packed_word_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_109),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(packed_word_17));
    packed_word_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_108),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                    select_half_operand<2, false, false>(0u));
    packed_word_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_120),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_150),
                                    select_half_operand<2, false, false>(0u));
    packed_word_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_148),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                    select_half_operand<2, false, false>(packed_word_105));
    packed_word_118 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_118),
                                          select_half_operand<2, false, false>(packed_word_102));
    packed_word_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                    select_half_operand<2, false, false>(0u));
    packed_word_108 = permute_signed(packed_word_107, packed_word_107, 0x5432u);
    packed_word_148 = pack_half(packed_word_101, packed_word_100);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3c60ull, packed_word_100,
        packed_word_101, packed_word_102, packed_word_103, memory_ranges, error_counts);
    packed_word_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                    select_half_operand<2, false, false>(packed_word_111));
    packed_word_158 = add_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                     select_half_operand<2, false, false>(packed_word_108));
    packed_word_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                    select_half_operand<2, false, false>(packed_word_109));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_160, packed_word_161) + 0x3e60ull, packed_word_108,
        packed_word_109, packed_word_110, packed_word_111, memory_ranges, error_counts);
    packed_word_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_146),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_158 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_158), 0x4100410u);
    packed_word_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_147),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_146),
                                    select_half_operand<2, false, false>(0u));
    packed_word_150 = half_to_float_bits((packed_word_158 & 65535u) * 65537u);
    packed_word_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_147),
                                    select_half_operand<2, false, false>(packed_word_146));
    packed_word_146 = half_to_float_bits((packed_word_158 >> 16) * 65537u);
    packed_word_150 = rsqrt_bits(packed_word_150);
    packed_word_158 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_119),
                                          select_half_operand<2, false, false>(packed_word_148));
    packed_word_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_154),
                                    select_half_operand<2, false, false>(0u));
    packed_word_160 = rsqrt_bits(packed_word_146);
    packed_word_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_121),
                                          select_half_operand<2, false, false>(packed_word_148));
    packed_word_159 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_149),
                                          select_half_operand<2, false, false>(packed_word_148));
    packed_word_161 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_151),
                                          select_half_operand<2, false, false>(packed_word_148));
    packed_word_163 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_163),
                                    select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_148, packed_word_149, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_16, packed_word_17,
                             packed_word_86, packed_word_87);
    packed_word_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_153),
                                    select_half_operand<2, false, false>(0u));
    packed_word_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_157),
                                   select_half_operand<2, false, false>(0u));
    packed_word_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_164),
                                   select_half_operand<2, false, false>(packed_word_163));
    packed_word_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_162),
                                   select_half_operand<2, false, false>(packed_word_87));
    multiply_accumulate_half(packed_word_146, packed_word_147, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_18, packed_word_19,
                             packed_word_84, packed_word_85);
    packed_word_160 = pack_half(packed_word_160, packed_word_150);
    packed_word_162 = parameters[56];
    packed_word_163 = parameters[57];
    packed_word_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_117),
                                   select_half_operand<2, false, false>(packed_word_154));
    packed_word_117 = permute_signed(packed_word_116, packed_word_116, 0x5432u);
    packed_word_84 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_152),
                                   select_half_operand<2, false, false>(packed_word_153));
    packed_word_116 = add_half_pairs(select_half_operand<2, false, false>(packed_word_116),
                                     select_half_operand<2, false, false>(packed_word_117));
    packed_word_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_116), 0x4100410u);
    packed_word_119 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_156),
                                    select_half_operand<2, false, false>(0u));
    packed_word_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                    select_half_operand<2, false, false>(0u));
    packed_word_119 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                    select_half_operand<2, false, false>(packed_word_119));
    packed_word_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_161),
                                    select_half_operand<2, false, false>(0u));
    packed_word_116 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_158),
                                    select_half_operand<2, false, false>(packed_word_121));
    packed_word_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_130),
                                          select_half_operand<2, false, false>(packed_word_160));
    packed_word_140 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                          select_half_operand<2, false, false>(packed_word_160));
    packed_word_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_155),
                                    select_half_operand<2, false, false>(0u));
    packed_word_118 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_118),
                                    select_half_operand<2, false, false>(packed_word_155));
    multiply_accumulate_half(packed_word_150, packed_word_151, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_86, packed_word_87,
                             packed_word_4, packed_word_5);
    packed_word_4 = permute_signed(packed_word_3, packed_word_3, 0x5432u);
    packed_word_4 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_4));
    packed_word_3 = half_to_float_bits((packed_word_117 & 65535u) * 65537u);
    multiply_accumulate_half(packed_word_152, packed_word_153, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_84, packed_word_85,
                             packed_word_6, packed_word_7);
    packed_word_6 = half_to_float_bits((packed_word_117 >> 16) * 65537u);
    packed_word_7 = permute_signed(packed_word_2, packed_word_2, 0x5432u);
    packed_word_3 = rsqrt_bits(packed_word_3);
    packed_word_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x4100410u);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(packed_word_7));
    packed_word_6 = rsqrt_bits(packed_word_6);
    packed_word_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_2), 0x4100410u);
    packed_word_4 = half_to_float_bits((packed_word_5 & 65535u) * 65537u);
    packed_word_5 = half_to_float_bits((packed_word_5 >> 16) * 65537u);
    packed_word_7 = half_to_float_bits((packed_word_2 >> 16) * 65537u);
    packed_word_117 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_159),
                                    select_half_operand<2, false, false>(packed_word_120));
    packed_word_130 = half_to_float_bits((packed_word_2 & 65535u) * 65537u);
    packed_word_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_142),
                                          select_half_operand<2, false, false>(packed_word_160));
    packed_word_4 = rsqrt_bits(packed_word_4);
    packed_word_5 = rsqrt_bits(packed_word_5);
    packed_word_142 = pack_half(packed_word_6, packed_word_3);
    packed_word_6 = rsqrt_bits(packed_word_130);
    packed_word_7 = rsqrt_bits(packed_word_7);
    packed_word_160 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_144),
                                          select_half_operand<2, false, false>(packed_word_160));
    packed_word_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_145),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_5 = pack_half(packed_word_5, packed_word_4);
    packed_word_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_143),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_140),
                                  select_half_operand<2, false, false>(0u));
    packed_word_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_131),
                                          select_half_operand<2, false, false>(packed_word_142));
    packed_word_160 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_160),
                                    select_half_operand<2, false, false>(0u));
    packed_word_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_141),
                                    select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_145),
                                  select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_half(packed_word_7, packed_word_6);
    packed_word_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_120),
                                    select_half_operand<2, false, false>(packed_word_3));
    packed_word_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_121),
                                    select_half_operand<2, false, false>(packed_word_160));
    packed_word_139 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                          select_half_operand<2, false, false>(packed_word_6));
    packed_word_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_143),
                                    select_half_operand<2, false, false>(packed_word_130));
    packed_word_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                          select_half_operand<2, false, false>(packed_word_6));
    packed_word_131 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_131),
                                    select_half_operand<2, false, false>(packed_word_4));
    packed_word_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                          select_half_operand<2, false, false>(packed_word_5));
    packed_word_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                          select_half_operand<2, false, false>(packed_word_6));
    packed_word_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                          select_half_operand<2, false, false>(packed_word_6));
    packed_word_139 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                          select_half_operand<0, false, false>(packed_word_0));
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_118, packed_word_119,
                             packed_word_8, packed_word_9);
    packed_word_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_138),
                                  select_half_operand<2, false, false>(0u));
    packed_word_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_132),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_136),
                                  select_half_operand<2, false, false>(0u));
    packed_word_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_133),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_134),
                                  select_half_operand<2, false, false>(packed_word_5));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_132),
                                  select_half_operand<2, false, false>(packed_word_7));
    multiply_accumulate_half(packed_word_10, packed_word_11, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_116, packed_word_117,
                             packed_word_10, packed_word_11);
    multiply_accumulate_half(packed_word_2, packed_word_3, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_120, packed_word_121,
                             packed_word_12, packed_word_13);
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_139),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_104, packed_word_105,
                             packed_word_106, packed_word_107, packed_word_130, packed_word_131,
                             packed_word_14, packed_word_15);
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_135),
                                  select_half_operand<2, false, false>(packed_word_12));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_137),
                                   select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_133),
                                  select_half_operand<2, false, false>(packed_word_14));
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_18, packed_word_19,
                             packed_word_76, packed_word_77);
    packed_word_76 = 0x29c0u;
    packed_word_0 = permute_signed(packed_word_0, packed_word_76, 0x5410u);
    packed_word_146 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_146),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_148 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_148),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_150 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_150),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_152 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_152),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_139 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_146), 0x3c203c20u);
    packed_word_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_8),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_148 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_148), 0x3c203c20u);
    packed_word_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_9),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_150 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_150), 0x3c203c20u);
    packed_word_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_10),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_152), 0x3c203c20u);
    packed_word_139 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_139), 0x3e473e47u);
    packed_word_148 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_148), 0x3e473e47u);
    packed_word_150 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_150), 0x3e473e47u);
    packed_word_152 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_152), 0x3e473e47u);
    packed_word_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3c203c20u);
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3c203c20u);
    packed_word_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0x3c203c20u);
    packed_word_147 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_147),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_16, packed_word_17,
                             packed_word_78, packed_word_79);
    packed_word_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3e473e47u);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3e473e47u);
    packed_word_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0x3e473e47u);
    packed_word_139 = (packed_word_139 << 0x5u) + 0x7ff88000u;
    packed_word_76 = (packed_word_148 << 0x5u) + 0x7ff88000u;
    packed_word_78 = (packed_word_150 << 0x5u) + 0x7ff88000u;
    packed_word_140 = (packed_word_152 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_96, packed_word_97, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_86, packed_word_87,
                             packed_word_96, packed_word_97);
    packed_word_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_147), 0x3c203c20u);
    packed_word_147 = (packed_word_8 << 0x5u) + 0x7ff88000u;
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                   select_half_operand<2, false, false>(packed_word_76));
    packed_word_136 = (packed_word_9 << 0x5u) + 0x7ff88000u;
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_140));
    packed_word_135 = (packed_word_10 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_84, packed_word_85,
                             packed_word_98, packed_word_99);
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_147),
                                   select_half_operand<2, false, false>(packed_word_135));
    packed_word_149 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_149),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_151 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_151),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_153 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_153),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_149 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_149), 0x3c203c20u);
    packed_word_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_11),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_118, packed_word_119,
                             packed_word_100, packed_word_101);
    packed_word_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_2),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_151 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_151), 0x3c203c20u);
    packed_word_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_104),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_153), 0x3c203c20u);
    packed_word_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_77), 0x3e473e47u);
    packed_word_149 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_149), 0x3e473e47u);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_116, packed_word_117,
                             packed_word_102, packed_word_103);
    packed_word_151 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_151), 0x3e473e47u);
    packed_word_153 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_153), 0x3e473e47u);
    packed_word_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0x3c203c20u);
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3c203c20u);
    packed_word_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_104), 0x3c203c20u);
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_120, packed_word_121,
                             packed_word_108, packed_word_109);
    packed_word_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0x3e473e47u);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3e473e47u);
    packed_word_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_104), 0x3e473e47u);
    packed_word_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_13),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_77 = (packed_word_77 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_130, packed_word_131,
                             packed_word_110, packed_word_111);
    packed_word_146 = (packed_word_149 << 0x5u) + 0x7ff88000u;
    packed_word_79 = (packed_word_151 << 0x5u) + 0x7ff88000u;
    packed_word_141 = (packed_word_153 << 0x5u) + 0x7ff88000u;
    packed_word_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_15),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_134 = (packed_word_11 << 0x5u) + 0x7ff88000u;
    packed_word_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_97),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_9 = (packed_word_9 << 0x5u) + 0x7ff88000u;
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_99),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_6 = (packed_word_104 << 0x5u) + 0x7ff88000u;
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                    select_half_operand<2, false, false>(packed_word_146));
    packed_word_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3c203c20u);
    packed_word_11 = add_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                    select_half_operand<2, false, false>(packed_word_141));
    packed_word_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3c203c20u);
    packed_word_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_12),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x3c203c20u);
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_14),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3c203c20u);
    packed_word_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_96),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3e473e47u);
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                    select_half_operand<2, false, false>(packed_word_11));
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_98),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_4 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                   select_half_operand<2, false, false>(packed_word_6));
    packed_word_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3c203c20u);
    packed_word_11 = add_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                    select_half_operand<2, false, false>(packed_word_134));
    packed_word_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3e473e47u);
    packed_word_104 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                     select_half_operand<2, false, false>(packed_word_4));
    packed_word_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3c203c20u);
    packed_word_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_97), 0x3e473e47u);
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x3c203c20u);
    packed_word_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_3),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3e473e47u);
    packed_word_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_105),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3c203c20u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3e473e47u);
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_101),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_11 = (packed_word_13 << 0x5u) + 0x7ff88000u;
    packed_word_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_102),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3e473e47u);
    packed_word_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_103),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_13 = (packed_word_15 << 0x5u) + 0x7ff88000u;
    packed_word_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_96), 0x3e473e47u);
    packed_word_15 = (packed_word_97 << 0x5u) + 0x7ff88000u;
    packed_word_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_108),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3e473e47u);
    packed_word_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_109),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_97 = (packed_word_99 << 0x5u) + 0x7ff88000u;
    packed_word_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_110),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0x3c203c20u);
    packed_word_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_111),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x3c203c20u);
    packed_word_10 = (packed_word_12 << 0x5u) + 0x7ff88000u;
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3c203c20u);
    packed_word_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3c203c20u);
    packed_word_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_102), 0x3c203c20u);
    packed_word_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_103), 0x3c203c20u);
    packed_word_12 = (packed_word_14 << 0x5u) + 0x7ff88000u;
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_14 = (packed_word_96 << 0x5u) + 0x7ff88000u;
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_97));
    packed_word_96 = (packed_word_98 << 0x5u) + 0x7ff88000u;
    packed_word_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_3), 0x3e473e47u);
    packed_word_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x3e473e47u);
    packed_word_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3e473e47u);
    packed_word_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3e473e47u);
    packed_word_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_102), 0x3e473e47u);
    packed_word_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_103), 0x3e473e47u);
    packed_word_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_108), 0x3c203c20u);
    packed_word_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0x3c203c20u);
    packed_word_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0x3c203c20u);
    packed_word_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0x3c203c20u);
    packed_word_105 = add_half_pairs(select_half_operand<2, false, false>(packed_word_7),
                                     select_half_operand<2, false, false>(packed_word_8));
    packed_word_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_108), 0x3e473e47u);
    packed_word_106 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                     select_half_operand<2, false, false>(packed_word_12));
    packed_word_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_109), 0x3e473e47u);
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(packed_word_96));
    packed_word_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_110), 0x3e473e47u);
    packed_word_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_111), 0x3e473e47u);
    packed_word_8 = (packed_word_3 << 0x5u) + 0x7ff88000u;
    packed_word_7 = (packed_word_4 << 0x5u) + 0x7ff88000u;
    packed_word_145 = (packed_word_100 << 0x5u) + 0x7ff88000u;
    packed_word_144 = (packed_word_101 << 0x5u) + 0x7ff88000u;
    packed_word_143 = (packed_word_102 << 0x5u) + 0x7ff88000u;
    packed_word_142 = (packed_word_103 << 0x5u) + 0x7ff88000u;
    packed_word_106 = add_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                     select_half_operand<2, false, false>(packed_word_5));
    packed_word_137 = (packed_word_108 << 0x5u) + 0x7ff88000u;
    packed_word_100 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                     select_half_operand<2, false, false>(packed_word_7));
    packed_word_3 = (packed_word_109 << 0x5u) + 0x7ff88000u;
    packed_word_101 = add_half_pairs(select_half_operand<2, false, false>(packed_word_145),
                                     select_half_operand<2, false, false>(packed_word_143));
    packed_word_4 = (packed_word_110 << 0x5u) + 0x7ff88000u;
    packed_word_107 = add_half_pairs(select_half_operand<2, false, false>(packed_word_144),
                                     select_half_operand<2, false, false>(packed_word_142));
    packed_word_5 = (packed_word_111 << 0x5u) + 0x7ff88000u;
    packed_word_103 = add_half_pairs(select_half_operand<2, false, false>(packed_word_2),
                                     select_half_operand<2, false, false>(packed_word_100));
    packed_word_101 = add_half_pairs(select_half_operand<2, false, false>(packed_word_106),
                                     select_half_operand<2, false, false>(packed_word_101));
    packed_word_102 = add_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                     select_half_operand<2, false, false>(packed_word_4));
    packed_word_105 = add_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                     select_half_operand<2, false, false>(packed_word_107));
    packed_word_2 = add_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                   select_half_operand<2, false, false>(packed_word_5));
    packed_word_100 = packed_word_165 * 0x4u;
    packed_word_102 = add_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                     select_half_operand<2, false, false>(packed_word_102));
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x1u;
        condition_1 = loaded_word != 0;
    }
    packed_word_105 = add_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                     select_half_operand<2, false, false>(packed_word_2));
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x2u;
        condition_0 = loaded_word != 0;
    }
    packed_word_2 = packed_word_100 & 0x1cu;
    packed_word_106 = condition_1 ? packed_word_103 : packed_word_104;
    packed_word_107 = condition_1 ? packed_word_105 : packed_word_102;
    packed_word_103 = condition_1 ? packed_word_104 : packed_word_103;
    packed_word_102 = condition_1 ? packed_word_102 : packed_word_105;
    packed_word_2 =
        uint32_t(int64_t(int32_t(packed_word_165)) * (1ll << 0x1du) >> 32) + packed_word_2;
    packed_word_111 = !condition_0 ? packed_word_106 : packed_word_107;
    packed_word_107 = !condition_0 ? packed_word_107 : packed_word_106;
    packed_word_106 = !condition_0 ? packed_word_103 : packed_word_102;
    packed_word_153 = 0x1u ^ packed_word_2;
    packed_word_110 = !condition_0 ? packed_word_102 : packed_word_103;
    packed_word_152 = 0x2u ^ packed_word_2;
    packed_word_151 = 0x3u ^ packed_word_2;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_111), "r"(packed_word_2), "r"(0x1fu), "r"(__activemask()));
        packed_word_111 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_106), "r"(packed_word_153), "r"(0x1fu), "r"(__activemask()));
        packed_word_106 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_107), "r"(packed_word_152), "r"(0x1fu), "r"(__activemask()));
        packed_word_107 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_110), "r"(packed_word_151), "r"(0x1fu), "r"(__activemask()));
        packed_word_110 = loaded_word;
    }
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x8u;
        condition_1 = loaded_word != 0;
    }
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x10u;
        condition_2 = loaded_word != 0;
    }
    packed_word_132 = !condition_1 ? packed_word_111 : packed_word_106;
    packed_word_133 = !condition_1 ? packed_word_106 : packed_word_111;
    packed_word_111 = !condition_1 ? packed_word_107 : packed_word_110;
    packed_word_106 = !condition_1 ? packed_word_110 : packed_word_107;
    packed_word_138 = !condition_2 ? packed_word_132 : packed_word_111;
    packed_word_107 = !condition_2 ? packed_word_133 : packed_word_106;
    packed_word_132 = !condition_2 ? packed_word_111 : packed_word_132;
    packed_word_106 = !condition_2 ? packed_word_106 : packed_word_133;
    packed_word_107 = add_half_pairs(select_half_operand<2, false, false>(packed_word_138),
                                     select_half_operand<2, false, false>(packed_word_107));
    packed_word_107 = add_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                     select_half_operand<2, false, false>(packed_word_132));
    packed_word_132 = add_half_pairs(select_half_operand<2, false, false>(packed_word_107),
                                     select_half_operand<2, false, false>(packed_word_106));
    packed_word_132 = add_half_pairs(select_half_operand<0, false, false>(packed_word_132),
                                     select_half_operand<1, false, false>(packed_word_132));
    packed_word_132 = permute_signed(packed_word_132, packed_word_132, 0x5410u);
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_60, packed_word_61, 0u,
                             0u);
    packed_word_138 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_132), 0x4100410u);
    packed_word_154 = packed_word_165;
    packed_word_150 = uint32_t(int32_t(packed_word_154) >> 0x2u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_62, packed_word_63, 0u,
                             0u);
    multiply_accumulate_half(packed_word_102, packed_word_103, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_64, packed_word_65, 0u,
                             0u);
    multiply_accumulate_half(packed_word_104, packed_word_105, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_66, packed_word_67, 0u,
                             0u);
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_52, packed_word_53, 0u,
                             0u);
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_54, packed_word_55, 0u,
                             0u);
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_56, packed_word_57, 0u,
                             0u);
    multiply_accumulate_half(packed_word_132, packed_word_133, packed_word_72, packed_word_73,
                             packed_word_74, packed_word_75, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_72 = half_to_float_bits((packed_word_138 & 65535u) * 65537u);
    packed_word_73 = half_to_float_bits((packed_word_138 >> 16) * 65537u);
    packed_word_74 = packed_word_150 + 0x8u;
    packed_word_72 = reciprocal_bits(packed_word_72);
    packed_word_73 = reciprocal_bits(packed_word_73);
    packed_word_75 = uint32_t(int32_t(packed_word_74) >> 0x1fu);
    packed_word_75 = uint32_t(combine_words(packed_word_75, 0u) << 0x5u >> 32) + packed_word_74;
    packed_word_75 = packed_word_75 & 0xffffffe0u;
    packed_word_75 = packed_word_74 + (0u - packed_word_75);
    packed_word_138 = pack_half(packed_word_73, packed_word_72);
    packed_word_73 = uint32_t(int32_t(packed_word_154) >> 0x1fu);
    packed_word_73 = uint32_t(combine_words(packed_word_73, 0u) << 0x5u >> 32) + packed_word_150;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_138), "r"(packed_word_75), "r"(0x1fu), "r"(__activemask()));
        packed_word_75 = loaded_word;
    }
    packed_word_148 = packed_word_73 & 0xffffffe0u;
    packed_word_149 = packed_word_150 + (0u - packed_word_148);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_138), "r"(packed_word_149), "r"(0x1fu), "r"(__activemask()));
        packed_word_149 = loaded_word;
    }
    packed_word_127 = transpose_half_tile(packed_word_127);
    packed_word_125 = permute_signed(packed_word_75, packed_word_75, 0x5410u);
    packed_word_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_146),
                                          select_half_operand<2, false, false>(packed_word_125));
    packed_word_126 = transpose_half_tile(packed_word_126);
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_125));
    packed_word_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_146),
                                    select_half_operand<2, false, false>(0u));
    packed_word_129 = transpose_half_tile(packed_word_129);
    packed_word_123 = permute_signed(packed_word_149, packed_word_149, 0x5410u);
    packed_word_149 = transpose_half_tile(packed_word_95);
    packed_word_77 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                   select_half_operand<2, false, false>(packed_word_146));
    packed_word_128 = transpose_half_tile(packed_word_128);
    packed_word_146 = transpose_half_tile(packed_word_89);
    packed_word_94 = transpose_half_tile(packed_word_94);
    packed_word_89 = transpose_half_tile(packed_word_91);
    split_address_words(packed_word_162, packed_word_163,
                        int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_word_162, packed_word_163));
    packed_word_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_127),
                                    select_half_operand<2, false, false>(0u));
    packed_word_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_141),
                                          select_half_operand<2, false, false>(packed_word_125));
    packed_word_126 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_126),
                                    select_half_operand<2, false, false>(packed_word_127));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x5070ull, packed_word_72, packed_word_73,
        packed_word_74, packed_word_75, memory_ranges, error_counts);
    packed_word_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_129),
                                    select_half_operand<2, false, false>(0u));
    packed_word_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_149),
                                    select_half_operand<2, false, false>(0u));
    packed_word_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                    select_half_operand<2, false, false>(packed_word_127));
    packed_word_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_141),
                                    select_half_operand<2, false, false>(0u));
    packed_word_141 = transpose_half_tile(packed_word_88);
    packed_word_91 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_94),
                                   select_half_operand<2, false, false>(packed_word_149));
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_89),
                                   select_half_operand<2, false, false>(0u));
    packed_word_88 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_80, packed_word_81) + 0x5470ull, memory_ranges, error_counts);
    packed_word_89 = load_aligned_word<CheckBounds>(
        combine_words(packed_word_82, packed_word_83) + 0x5470ull, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x5270ull, packed_word_80, packed_word_81,
        packed_word_82, packed_word_83, memory_ranges, error_counts);
    packed_word_79 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_79),
                                         select_half_operand<2, false, false>(packed_word_125));
    packed_word_79 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(packed_word_128));
    packed_word_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_139),
                                         select_half_operand<2, false, false>(packed_word_123));
    packed_word_140 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_140),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_129 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                    select_half_operand<2, false, false>(0u));
    packed_word_78 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_78),
                                         select_half_operand<2, false, false>(packed_word_123));
    packed_word_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(packed_word_129));
    packed_word_129 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_140),
                                    select_half_operand<2, false, false>(0u));
    packed_word_128 = transpose_half_tile(packed_word_113);
    packed_word_93 = transpose_half_tile(packed_word_93);
    packed_word_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_78),
                                   select_half_operand<2, false, false>(packed_word_129));
    packed_word_139 = transpose_half_tile(packed_word_115);
    packed_word_129 = transpose_half_tile(packed_word_90);
    packed_word_95 = transpose_half_tile(packed_word_112);
    packed_word_92 = transpose_half_tile(packed_word_92);
    packed_word_140 = transpose_half_tile(packed_word_114);
    packed_word_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_146),
                                    select_half_operand<2, false, false>(0u));
    packed_word_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                    select_half_operand<2, false, false>(0u));
    packed_word_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                   select_half_operand<2, false, false>(0u));
    packed_word_139 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_139),
                                    select_half_operand<2, false, false>(0u));
    packed_word_90 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_141),
                                   select_half_operand<2, false, false>(packed_word_146));
    packed_word_94 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_129),
                                   select_half_operand<2, false, false>(packed_word_94));
    packed_word_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_95),
                                   select_half_operand<2, false, false>(packed_word_128));
    packed_word_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(packed_word_93));
    packed_word_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_140),
                                   select_half_operand<2, false, false>(packed_word_139));
    multiply_accumulate_half(packed_word_112, packed_word_113, packed_word_76, packed_word_77,
                             packed_word_78, packed_word_79, packed_word_126, packed_word_127, 0u,
                             0u);
    packed_word_140 = uint32_t(int64_t(int32_t(packed_word_154)) * (1ll << 0x1eu) >> 32) + 0x18u;
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_76, packed_word_77,
                             packed_word_78, packed_word_79, packed_word_90, packed_word_91, 0u,
                             0u);
    packed_word_139 = uint32_t(int32_t(packed_word_140) >> 0x1fu);
    multiply_accumulate_half(packed_word_128, packed_word_129, packed_word_76, packed_word_77,
                             packed_word_78, packed_word_79, packed_word_94, packed_word_95, 0u,
                             0u);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_76, packed_word_77,
                             packed_word_78, packed_word_79, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_78 = uint32_t(int64_t(int32_t(packed_word_154)) * (1ll << 0x1eu) >> 32) + 0x10u;
    packed_word_79 = uint32_t(int32_t(packed_word_78) >> 0x1fu);
    packed_word_79 = uint32_t(combine_words(packed_word_79, 0u) << 0x5u >> 32) + packed_word_78;
    packed_word_79 = packed_word_79 & 0xffffffe0u;
    packed_word_141 = uint32_t(combine_words(packed_word_139, 0u) << 0x5u >> 32) + packed_word_140;
    packed_word_139 = packed_word_78 + (0u - packed_word_79);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_138), "r"(packed_word_139), "r"(0x1fu), "r"(__activemask()));
        packed_word_139 = loaded_word;
    }
    packed_word_141 = packed_word_141 & 0xffffffe0u;
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_54, packed_word_55, 0u,
                             0u);
    packed_word_141 = packed_word_140 + (0u - packed_word_141);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_138), "r"(packed_word_141), "r"(0x1fu), "r"(__activemask()));
        packed_word_141 = loaded_word;
    }
    multiply_accumulate_half(packed_word_78, packed_word_79, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_52, packed_word_53, 0u,
                             0u);
    packed_word_146 = permute_signed(packed_word_139, packed_word_139, 0x5410u);
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                         select_half_operand<2, false, false>(packed_word_146));
    packed_word_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                         select_half_operand<2, false, false>(packed_word_146));
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_56, packed_word_57, 0u,
                             0u);
    packed_word_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_12),
                                   select_half_operand<2, false, false>(0u));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_57));
    packed_word_10 = transpose_half_tile(packed_word_55);
    packed_word_149 = permute_signed(packed_word_141, packed_word_141, 0x5410u);
    multiply_accumulate_half(packed_word_58, packed_word_59, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_58, packed_word_59, 0u,
                             0u);
    packed_word_57 = transpose_half_tile(packed_word_54);
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_13),
                                         select_half_operand<2, false, false>(packed_word_149));
    packed_word_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                         select_half_operand<2, false, false>(packed_word_149));
    packed_word_111 = transpose_half_tile(packed_word_111);
    packed_word_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_13),
                                   select_half_operand<2, false, false>(0u));
    packed_word_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_136),
                                          select_half_operand<2, false, false>(packed_word_125));
    packed_word_109 = transpose_half_tile(packed_word_109);
    packed_word_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_134),
                                          select_half_operand<2, false, false>(packed_word_125));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_56));
    packed_word_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                        select_half_operand<2, false, false>(packed_word_125));
    packed_word_56 = transpose_half_tile(packed_word_110);
    packed_word_125 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_7),
                                          select_half_operand<2, false, false>(packed_word_125));
    packed_word_79 = transpose_half_tile(packed_word_79);
    packed_word_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_147),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_135),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_7 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                        select_half_operand<2, false, false>(packed_word_123));
    packed_word_108 = transpose_half_tile(packed_word_108);
    packed_word_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_6),
                                          select_half_operand<2, false, false>(packed_word_123));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(0u));
    packed_word_6 = transpose_half_tile(packed_word_78);
    packed_word_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_96),
                                         select_half_operand<2, false, false>(packed_word_146));
    packed_word_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_57),
                                   select_half_operand<2, false, false>(packed_word_10));
    packed_word_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_97),
                                         select_half_operand<2, false, false>(packed_word_149));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_123),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                         select_half_operand<2, false, false>(packed_word_149));
    packed_word_139 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                    select_half_operand<2, false, false>(0u));
    packed_word_107 = transpose_half_tile(packed_word_107);
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(0u));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                   select_half_operand<2, false, false>(packed_word_10));
    packed_word_53 = transpose_half_tile(packed_word_53);
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_125),
                                   select_half_operand<2, false, false>(0u));
    packed_word_133 = transpose_half_tile(packed_word_133);
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_96));
    packed_word_7 = transpose_half_tile(packed_word_59);
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_11));
    packed_word_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                         select_half_operand<2, false, false>(packed_word_146));
    packed_word_106 = transpose_half_tile(packed_word_106);
    packed_word_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                    select_half_operand<2, false, false>(0u));
    packed_word_52 = transpose_half_tile(packed_word_52);
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_134),
                                  select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_135),
                                  select_half_operand<2, false, false>(0u));
    packed_word_96 = transpose_half_tile(packed_word_132);
    packed_word_97 = transpose_half_tile(packed_word_58);
    packed_word_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_56),
                                   select_half_operand<2, false, false>(packed_word_111));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_136),
                                  select_half_operand<2, false, false>(packed_word_9));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_147),
                                  select_half_operand<2, false, false>(packed_word_8));
    packed_word_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                    select_half_operand<2, false, false>(0u));
    packed_word_79 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_79),
                                   select_half_operand<2, false, false>(0u));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(packed_word_139));
    packed_word_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(packed_word_109));
    packed_word_79 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                   select_half_operand<2, false, false>(packed_word_79));
    multiply_accumulate_half(packed_word_112, packed_word_113, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_56, packed_word_57,
                             packed_word_112, packed_word_113);
    packed_word_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                    select_half_operand<2, false, false>(0u));
    packed_word_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_53),
                                   select_half_operand<2, false, false>(0u));
    packed_word_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_133),
                                    select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_138, packed_word_139, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_126, packed_word_127, 0u,
                             0u);
    packed_word_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                   select_half_operand<2, false, false>(packed_word_107));
    packed_word_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_52),
                                   select_half_operand<2, false, false>(packed_word_53));
    packed_word_96 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_96),
                                   select_half_operand<2, false, false>(packed_word_133));
    multiply_accumulate_half(packed_word_140, packed_word_141, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_90, packed_word_91, 0u,
                             0u);
    packed_word_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_97),
                                   select_half_operand<2, false, false>(packed_word_6));
    packed_word_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_142),
                                          select_half_operand<2, false, false>(packed_word_149));
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_94, packed_word_95, 0u,
                             0u);
    packed_word_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_143),
                                          select_half_operand<2, false, false>(packed_word_146));
    packed_word_4 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_4),
                                        select_half_operand<2, false, false>(packed_word_146));
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_5 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_5),
                                        select_half_operand<2, false, false>(packed_word_149));
    multiply_accumulate_half(packed_word_114, packed_word_115, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_78, packed_word_79,
                             packed_word_114, packed_word_115);
    packed_word_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_144),
                                          select_half_operand<2, false, false>(packed_word_149));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_142),
                                  select_half_operand<2, false, false>(0u));
    packed_word_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_145),
                                          select_half_operand<2, false, false>(packed_word_146));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                   select_half_operand<2, false, false>(0u));
    packed_word_6 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_137),
                                        select_half_operand<2, false, false>(packed_word_146));
    packed_word_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_3),
                                        select_half_operand<2, false, false>(packed_word_149));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_144),
                                  select_half_operand<2, false, false>(packed_word_7));
    multiply_accumulate_half(packed_word_128, packed_word_129, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_58, packed_word_59,
                             packed_word_128, packed_word_129);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_3),
                                  select_half_operand<2, false, false>(packed_word_14));
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_112),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_96, packed_word_97,
                             packed_word_76, packed_word_77);
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_143),
                                   select_half_operand<2, false, false>(0u));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_4),
                                   select_half_operand<2, false, false>(0u));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_145),
                                  select_half_operand<2, false, false>(packed_word_10));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                  select_half_operand<2, false, false>(packed_word_11));
    multiply_accumulate_half(packed_word_138, packed_word_139, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_56, packed_word_57,
                             packed_word_138, packed_word_139);
    multiply_accumulate_half(packed_word_140, packed_word_141, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_78, packed_word_79,
                             packed_word_140, packed_word_141);
    multiply_accumulate_half(packed_word_10, packed_word_11, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_58, packed_word_59,
                             packed_word_54, packed_word_55);
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_96, packed_word_97,
                             packed_word_12, packed_word_13);
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_114),
                                  select_half_operand<2, false, false>(packed_word_3));
    packed_word_114 = blockIdx.x;
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_113),
                                  select_half_operand<2, false, false>(0u));
    packed_word_113 = blockIdx.y;
    multiply_accumulate_half(packed_word_60, packed_word_61, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_60, packed_word_61, 0u,
                             0u);
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_8),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_62, packed_word_63, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_62, packed_word_63, 0u,
                             0u);
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_9),
                                  select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_64, packed_word_65, 0u,
                             0u);
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_12),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_68, packed_word_69, packed_word_68, packed_word_69,
                             packed_word_70, packed_word_71, packed_word_66, packed_word_67, 0u,
                             0u);
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_138),
                                  select_half_operand<2, false, false>(0u));
    uniform_word_5 = uint32_t(int32_t(uniform_word_9) >> 0x1fu);
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_13),
                                   select_half_operand<2, false, false>(0u));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_115),
                                  select_half_operand<2, false, false>(packed_word_6));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_128),
                                  select_half_operand<2, false, false>(packed_word_7));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_129),
                                  select_half_operand<2, false, false>(packed_word_8));
    uniform_word_5 = uint32_t(combine_words(uniform_word_5, 0u) << 0x2u >> 32) + uniform_word_9;
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_140),
                                  select_half_operand<2, false, false>(packed_word_3));
    packed_word_3 = packed_word_114 * 0x2u;
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_12));
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                         select_half_operand<2, false, false>(packed_word_104));
    uniform_word_4 = uint32_t(int32_t(uniform_word_8) >> 0x1fu);
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                         select_half_operand<2, false, false>(packed_word_102));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_139),
                                   select_half_operand<2, false, false>(0u));
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_122));
    uniform_word_5 = uint32_t(int32_t(uniform_word_5) >> 0x2u);
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_122));
    uniform_word_4 = uint32_t(combine_words(uniform_word_4, 0u) << 0x2u >> 32) + uniform_word_8;
    packed_word_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                      select_half_operand<2, false, false>(packed_word_100),
                                      select_half_operand<2, false, false>(packed_word_12));
    packed_word_112 = packed_word_3 + 0x1u;
    packed_word_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                      select_half_operand<2, false, false>(packed_word_98),
                                      select_half_operand<2, false, false>(packed_word_13));
    packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_141),
                                  select_half_operand<2, false, false>(packed_word_14));
    packed_word_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_15));
    packed_word_41 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_124));
    uniform_word_4 = uint32_t(int32_t(uniform_word_4) >> 0x2u);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_39 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_64 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_13));
    packed_word_113 = packed_word_113 * 0x2u;
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                         select_half_operand<2, false, false>(packed_word_76));
    condition_3 = int32_t(packed_word_112) >= int32_t(uniform_word_5);
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_68));
    multiply_accumulate_half(packed_word_70, packed_word_71, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_72, packed_word_73,
                             packed_word_36, packed_word_37);
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                         select_half_operand<2, false, false>(packed_word_105));
    packed_word_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_77));
    packed_word_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                      select_half_operand<2, false, false>(packed_word_60),
                                      select_half_operand<2, false, false>(packed_word_13));
    packed_word_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                      select_half_operand<2, false, false>(packed_word_62),
                                      select_half_operand<2, false, false>(packed_word_15));
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_69));
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                         select_half_operand<2, false, false>(packed_word_103));
    condition_4 = int32_t(packed_word_113) >= int32_t(uniform_word_4) || condition_3;
    packed_word_36 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                      select_half_operand<2, false, false>(packed_word_63),
                                      select_half_operand<2, false, false>(packed_word_36));
    multiply_accumulate_half(packed_word_106, packed_word_107, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_74, packed_word_75,
                             packed_word_40, packed_word_41);
    packed_word_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                      select_half_operand<2, false, false>(packed_word_101),
                                      select_half_operand<2, false, false>(packed_word_12));
    packed_word_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_99),
                                      select_half_operand<2, false, false>(packed_word_37));
    packed_word_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                      select_half_operand<2, false, false>(packed_word_61),
                                      select_half_operand<2, false, false>(packed_word_14));
    packed_word_15 = add_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                    select_half_operand<2, false, false>(packed_word_13));
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                         select_half_operand<2, false, false>(packed_word_88));
    packed_word_49 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                         select_half_operand<2, false, false>(packed_word_88));
    multiply_accumulate_half(packed_word_72, packed_word_73, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_72, packed_word_73,
                             packed_word_38, packed_word_39);
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_37));
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                    select_half_operand<2, false, false>(packed_word_14));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_15), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_36 = loaded_word;
    }
    multiply_accumulate_half(packed_word_74, packed_word_75, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_74, packed_word_75,
                             packed_word_42, packed_word_43);
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_13 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_65 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_word_115 = loaded_word;
    }
    multiply_accumulate_half(packed_word_108, packed_word_109, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_80, packed_word_81,
                             packed_word_48, packed_word_49);
    condition_5 = int32_t(packed_word_3) >= int32_t(uniform_word_5);
    multiply_accumulate_half(packed_word_110, packed_word_111, packed_word_8, packed_word_9,
                             packed_word_10, packed_word_11, packed_word_82, packed_word_83,
                             packed_word_50, packed_word_51);
    if (!condition_4) {
        packed_word_8 = parameters[54];
        packed_word_9 = parameters[55];
    }
    packed_word_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_75),
                                   select_half_operand<2, false, false>(0u));
    condition_6 = int32_t(packed_word_113) >= int32_t(uniform_word_4) || condition_5;
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_15),
                                     select_half_operand<2, false, false>(packed_word_36));
    packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_73),
                                   select_half_operand<2, false, false>(0u));
    if (!condition_4) {
        packed_word_10 = packed_word_113 * uniform_word_5 + packed_word_112;
    }
    packed_word_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_88));
    if (!condition_4) {
        packed_word_36 = permute_signed(packed_word_11, packed_word_38, 0x5410u);
    }
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_88));
    if (!condition_4) {
        packed_word_11 = packed_word_10 * 0x80u;
    }
    packed_word_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_47 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_64 = add_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                    select_half_operand<2, false, false>(packed_word_13));
    packed_word_65 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_65));
    packed_word_115 = add_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                     select_half_operand<2, false, false>(packed_word_115));
    if (!condition_4) {
        split_address_words(packed_word_8, packed_word_9,
                            uint64_t(uint32_t(packed_word_11)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(packed_word_8, packed_word_9));
    }
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_72),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_80, packed_word_81, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_80, packed_word_81,
                             packed_word_44, packed_word_45);
    packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_74),
                                   select_half_operand<2, false, false>(0u));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_64), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_125 = loaded_word;
    }
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_109),
                                   select_half_operand<2, false, false>(0u));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_111),
                                   select_half_operand<2, false, false>(0u));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_134 = loaded_word;
    }
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_108),
                                   select_half_operand<2, false, false>(0u));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_110),
                                   select_half_operand<2, false, false>(0u));
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_123), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_132 = loaded_word;
    }
    if (!condition_4) {
        packed_word_37 = permute_signed(packed_word_37, packed_word_10, 0x5410u);
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_115), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_word_128 = loaded_word;
    }
    multiply_accumulate_half(packed_word_82, packed_word_83, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_82, packed_word_83,
                             packed_word_46, packed_word_47);
    if (!condition_4) {
        packed_word_14 = permute_signed(packed_word_14, packed_word_13, 0x5410u);

        split_address_words(packed_word_66, packed_word_67,
                            int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_8, packed_word_9));

        packed_word_15 = permute_signed(packed_word_15, packed_word_12, 0x5410u);
    }
    if (!condition_6) {
        packed_word_4 = parameters[54];
        packed_word_5 = parameters[55];
        packed_word_6 = packed_word_113 * uniform_word_5 + packed_word_3;
        packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_83),
                                      select_half_operand<2, false, false>(0u));
        packed_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_82),
                                      select_half_operand<2, false, false>(0u));
        packed_word_7 = packed_word_6 * 0x80u;
        packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_107),
                                      select_half_operand<2, false, false>(0u));
        packed_word_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_81),
                                       select_half_operand<2, false, false>(packed_word_8));
        packed_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_80),
                                       select_half_operand<2, false, false>(packed_word_9));
        packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_71),
                                       select_half_operand<2, false, false>(packed_word_6));
        packed_word_6 = packed_word_11;
        split_address_words(packed_word_4, packed_word_5,
                            uint64_t(uint32_t(packed_word_7)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(packed_word_4, packed_word_5));
        packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_106),
                                      select_half_operand<2, false, false>(0u));
        packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_70),
                                       select_half_operand<2, false, false>(packed_word_7));
        split_address_words(packed_word_8, packed_word_9,
                            int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_4, packed_word_5));
        packed_word_7 = packed_word_10;
        packed_word_5 = packed_word_12;
        packed_word_4 = packed_word_13;
        // Publish four packed feature words to the neural input buffer.
        store_four_words<CheckBounds>(combine_words(packed_word_8, packed_word_9), packed_word_4,
                                      packed_word_5, packed_word_6, packed_word_7, memory_ranges,
                                      error_counts);
    }
    if (!condition_4) {
        packed_word_4 = packed_word_37;

        packed_word_5 = packed_word_36;

        packed_word_6 = packed_word_15;

        packed_word_7 = packed_word_14;

        store_four_words<CheckBounds>(combine_words(packed_word_66, packed_word_67), packed_word_4,
                                      packed_word_5, packed_word_6, packed_word_7, memory_ranges,
                                      error_counts);
    }
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4260ull, packed_word_8, packed_word_9,
        packed_word_10, packed_word_11, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4860ull, packed_word_40, packed_word_41,
        packed_word_42, packed_word_43, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4a60ull, packed_word_44, packed_word_45,
        packed_word_46, packed_word_47, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4460ull, packed_word_12, packed_word_13,
        packed_word_14, packed_word_15, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4060ull, packed_word_4, packed_word_5,
        packed_word_6, packed_word_7, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4c60ull, packed_word_48, packed_word_49,
        packed_word_50, packed_word_51, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4660ull, packed_word_36, packed_word_37,
        packed_word_38, packed_word_39, memory_ranges, error_counts);
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x4e60ull, packed_word_52, packed_word_53,
        packed_word_54, packed_word_55, memory_ranges, error_counts);
    packed_word_64 = add_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                    select_half_operand<2, false, false>(packed_word_125));
    packed_word_65 = add_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                    select_half_operand<2, false, false>(packed_word_134));
    packed_word_67 = permute_signed(packed_word_64, packed_word_64, 0x5432u);
    packed_word_134 = permute_signed(packed_word_65, packed_word_65, 0x5432u);
    packed_word_64 = add_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                    select_half_operand<2, false, false>(packed_word_67));
    packed_word_65 = add_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                    select_half_operand<2, false, false>(packed_word_134));
    packed_word_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_64), 0x4100410u);
    packed_word_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_65), 0x4100410u);
    packed_word_66 = half_to_float_bits((packed_word_64 & 65535u) * 65537u);
    packed_word_125 = half_to_float_bits((packed_word_65 & 65535u) * 65537u);
    packed_word_64 = half_to_float_bits((packed_word_64 >> 16) * 65537u);
    packed_word_65 = half_to_float_bits((packed_word_65 >> 16) * 65537u);
    packed_word_66 = rsqrt_bits(packed_word_66);
    packed_word_67 = rsqrt_bits(packed_word_64);
    packed_word_125 = rsqrt_bits(packed_word_125);
    packed_word_134 = rsqrt_bits(packed_word_65);
    packed_word_67 = pack_half(packed_word_67, packed_word_66);
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                         select_half_operand<2, false, false>(packed_word_67));
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                          select_half_operand<2, false, false>(packed_word_67));
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                          select_half_operand<2, false, false>(packed_word_67));
    packed_word_134 = pack_half(packed_word_134, packed_word_125);
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<2, false, false>(packed_word_67));
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                         select_half_operand<2, false, false>(packed_word_134));
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                          select_half_operand<2, false, false>(packed_word_134));
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                          select_half_operand<2, false, false>(packed_word_134));
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                          select_half_operand<2, false, false>(packed_word_134));
    packed_word_98 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_98),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_100),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_99),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_101),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_102 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_102),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_104),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_103),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_105),
                                          select_half_operand<0, false, false>(packed_word_0));
    packed_word_64 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_98),
                                   select_half_operand<2, false, false>(0u));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_100),
                                   select_half_operand<2, false, false>(0u));
    packed_word_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_99),
                                   select_half_operand<2, false, false>(0u));
    packed_word_66 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_101),
                                   select_half_operand<2, false, false>(0u));
    packed_word_102 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_102),
                                    select_half_operand<2, false, false>(0u));
    packed_word_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_104),
                                   select_half_operand<2, false, false>(0u));
    packed_word_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_103),
                                    select_half_operand<2, false, false>(0u));
    packed_word_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_105),
                                   select_half_operand<2, false, false>(0u));
    packed_word_64 = permute_signed(packed_word_64, packed_word_65, 0x5410u);
    packed_word_65 = permute_signed(packed_word_99, packed_word_66, 0x5410u);
    packed_word_66 = permute_signed(packed_word_102, packed_word_67, 0x5410u);
    packed_word_67 = permute_signed(packed_word_103, packed_word_98, 0x5410u);
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_132));
    packed_word_115 = add_half_pairs(select_half_operand<2, false, false>(packed_word_115),
                                     select_half_operand<2, false, false>(packed_word_128));
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x1u;
        condition_6 = loaded_word != 0;
    }
    multiply_accumulate_half(packed_word_98, packed_word_99, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_18, packed_word_19,
                             packed_word_4, packed_word_5);
    packed_word_4 = permute_signed(packed_word_123, packed_word_123, 0x5432u);
    multiply_accumulate_half(packed_word_100, packed_word_101, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_16, packed_word_17,
                             packed_word_6, packed_word_7);
    packed_word_123 = add_half_pairs(select_half_operand<2, false, false>(packed_word_123),
                                     select_half_operand<2, false, false>(packed_word_4));
    packed_word_6 = permute_signed(packed_word_115, packed_word_115, 0x5432u);
    packed_word_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_123), 0x4100410u);
    packed_word_115 = add_half_pairs(select_half_operand<2, false, false>(packed_word_115),
                                     select_half_operand<2, false, false>(packed_word_6));
    packed_word_4 = half_to_float_bits((packed_word_123 & 65535u) * 65537u);
    packed_word_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_115), 0x4100410u);
    packed_word_123 = half_to_float_bits((packed_word_123 >> 16) * 65537u);
    packed_word_5 = half_to_float_bits((packed_word_115 & 65535u) * 65537u);
    packed_word_115 = half_to_float_bits((packed_word_115 >> 16) * 65537u);
    packed_word_4 = rsqrt_bits(packed_word_4);
    packed_word_123 = rsqrt_bits(packed_word_123);
    packed_word_5 = rsqrt_bits(packed_word_5);
    packed_word_6 = rsqrt_bits(packed_word_115);
    packed_word_4 = pack_half(packed_word_123, packed_word_4);
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_4));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_4));
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                         select_half_operand<2, false, false>(packed_word_4));
    packed_word_6 = pack_half(packed_word_6, packed_word_5);
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<2, false, false>(packed_word_4));
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_6));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_6));
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<2, false, false>(packed_word_6));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_6));
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_76 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_68 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_68),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_77 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_77),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<0, false, false>(packed_word_0));
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                  select_half_operand<2, false, false>(0u));
    packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                  select_half_operand<2, false, false>(0u));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(0u));
    packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                  select_half_operand<2, false, false>(0u));
    packed_word_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(0u));
    packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_68),
                                  select_half_operand<2, false, false>(0u));
    packed_word_77 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                   select_half_operand<2, false, false>(0u));
    packed_word_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_69),
                                   select_half_operand<2, false, false>(0u));
    packed_word_4 = permute_signed(packed_word_4, packed_word_5, 0x5410u);
    packed_word_5 = permute_signed(packed_word_61, packed_word_6, 0x5410u);
    packed_word_6 = permute_signed(packed_word_76, packed_word_7, 0x5410u);
    packed_word_7 = permute_signed(packed_word_77, packed_word_60, 0x5410u);
    multiply_accumulate_half(packed_word_8, packed_word_9, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_86, packed_word_87,
                             packed_word_8, packed_word_9);
    multiply_accumulate_half(packed_word_10, packed_word_11, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_84, packed_word_85,
                             packed_word_10, packed_word_11);
    multiply_accumulate_half(packed_word_40, packed_word_41, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_18, packed_word_19,
                             packed_word_40, packed_word_41);
    multiply_accumulate_half(packed_word_44, packed_word_45, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_86, packed_word_87,
                             packed_word_44, packed_word_45);
    multiply_accumulate_half(packed_word_42, packed_word_43, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_16, packed_word_17,
                             packed_word_42, packed_word_43);
    multiply_accumulate_half(packed_word_46, packed_word_47, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_84, packed_word_85,
                             packed_word_46, packed_word_47);
    packed_word_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_98),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_100),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_8),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_12, packed_word_13, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_118, packed_word_119,
                             packed_word_12, packed_word_13);
    packed_word_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_10),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_116, packed_word_117,
                             packed_word_14, packed_word_15);
    packed_word_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_9),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3c203c20u);
    packed_word_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_40),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_48, packed_word_49, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_118, packed_word_119,
                             packed_word_48, packed_word_49);
    packed_word_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_44),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3c203c20u);
    packed_word_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_42),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_50, packed_word_51, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_116, packed_word_117,
                             packed_word_50, packed_word_51);
    packed_word_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3c203c20u);
    packed_word_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_46),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0x3c203c20u);
    packed_word_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_98), 0x3e473e47u);
    packed_word_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_100), 0x3e473e47u);
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3c203c20u);
    packed_word_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3e473e47u);
    packed_word_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_10), 0x3e473e47u);
    packed_word_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x3c203c20u);
    packed_word_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x3c203c20u);
    packed_word_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x3c203c20u);
    packed_word_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_46), 0x3c203c20u);
    multiply_accumulate_half(packed_word_36, packed_word_37, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_120, packed_word_121,
                             packed_word_36, packed_word_37);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3e473e47u);
    packed_word_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_40), 0x3e473e47u);
    packed_word_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_44), 0x3e473e47u);
    packed_word_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_42), 0x3e473e47u);
    packed_word_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_46), 0x3e473e47u);
    packed_word_40 = (packed_word_98 << 0x5u) + 0x7ff88000u;
    packed_word_61 = (packed_word_100 << 0x5u) + 0x7ff88000u;
    packed_word_44 = (packed_word_8 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_38, packed_word_39, packed_word_64, packed_word_65,
                             packed_word_66, packed_word_67, packed_word_130, packed_word_131,
                             packed_word_38, packed_word_39);
    packed_word_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_99),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_46 = (packed_word_9 << 0x5u) + 0x7ff88000u;
    packed_word_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_101),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_11),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_65 = (packed_word_10 << 0x5u) + 0x7ff88000u;
    packed_word_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_12),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_17 = (packed_word_17 << 0x5u) + 0x7ff88000u;
    packed_word_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_14),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_16 = (packed_word_16 << 0x5u) + 0x7ff88000u;
    multiply_accumulate_half(packed_word_52, packed_word_53, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_120, packed_word_121,
                             packed_word_52, packed_word_53);
    packed_word_18 = (packed_word_18 << 0x5u) + 0x7ff88000u;
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                   select_half_operand<2, false, false>(packed_word_61));
    packed_word_19 = (packed_word_19 << 0x5u) + 0x7ff88000u;
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                   select_half_operand<2, false, false>(packed_word_65));
    packed_word_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_41),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_43),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_45),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_47),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    multiply_accumulate_half(packed_word_54, packed_word_55, packed_word_4, packed_word_5,
                             packed_word_6, packed_word_7, packed_word_130, packed_word_131,
                             packed_word_54, packed_word_55);
    packed_word_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_48),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_50),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3c203c20u);
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3c203c20u);
    packed_word_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0x3c203c20u);
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                   select_half_operand<2, false, false>(packed_word_16));
    packed_word_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3c203c20u);
    packed_word_60 = add_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                    select_half_operand<2, false, false>(packed_word_19));
    packed_word_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3c203c20u);
    packed_word_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x3c203c20u);
    packed_word_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x3c203c20u);
    packed_word_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x3c203c20u);
    packed_word_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0x3c203c20u);
    packed_word_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x3c203c20u);
    packed_word_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x3c203c20u);
    packed_word_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_13),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_99), 0x3e473e47u);
    packed_word_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_101), 0x3e473e47u);
    packed_word_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_11), 0x3e473e47u);
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                   select_half_operand<2, false, false>(packed_word_60));
    packed_word_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_12), 0x3e473e47u);
    packed_word_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_14), 0x3e473e47u);
    packed_word_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_41), 0x3e473e47u);
    packed_word_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x3e473e47u);
    packed_word_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_45), 0x3e473e47u);
    packed_word_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0x3e473e47u);
    packed_word_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_48), 0x3e473e47u);
    packed_word_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x3e473e47u);
    packed_word_42 = (packed_word_99 << 0x5u) + 0x7ff88000u;
    packed_word_63 = (packed_word_101 << 0x5u) + 0x7ff88000u;
    packed_word_67 = (packed_word_11 << 0x5u) + 0x7ff88000u;
    packed_word_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3c203c20u);
    packed_word_43 = (packed_word_12 << 0x5u) + 0x7ff88000u;
    packed_word_50 = (packed_word_14 << 0x5u) + 0x7ff88000u;
    packed_word_47 = (packed_word_4 << 0x5u) + 0x7ff88000u;
    packed_word_62 = (packed_word_62 << 0x5u) + 0x7ff88000u;
    packed_word_64 = (packed_word_45 << 0x5u) + 0x7ff88000u;
    packed_word_69 = (packed_word_69 << 0x5u) + 0x7ff88000u;
    packed_word_45 = (packed_word_5 << 0x5u) + 0x7ff88000u;
    packed_word_60 = (packed_word_60 << 0x5u) + 0x7ff88000u;
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                    select_half_operand<2, false, false>(packed_word_63));
    packed_word_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_13), 0x3e473e47u);
    packed_word_11 = add_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                    select_half_operand<2, false, false>(packed_word_67));
    packed_word_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_15),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_4 = add_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(packed_word_50));
    packed_word_7 = add_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                    select_half_operand<2, false, false>(packed_word_69));
    packed_word_6 = add_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_60));
    packed_word_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_49),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_51),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3c203c20u);
    packed_word_11 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                    select_half_operand<2, false, false>(packed_word_11));
    packed_word_41 = (packed_word_13 << 0x5u) + 0x7ff88000u;
    packed_word_10 = add_half_pairs(select_half_operand<2, false, false>(packed_word_8),
                                    select_half_operand<2, false, false>(packed_word_4));
    packed_word_13 = add_half_pairs(select_half_operand<2, false, false>(packed_word_7),
                                    select_half_operand<2, false, false>(packed_word_12));
    packed_word_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_36),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_37),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_38),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_39),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x3c203c20u);
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_9),
                                    select_half_operand<2, false, false>(packed_word_6));
    packed_word_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x3c203c20u);
    packed_word_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_52),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_15), 0x3e473e47u);
    packed_word_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_53),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_54),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_word_55),
        select_half_operand<1, false, false>(packed_word_0), 0x3d343d34u);
    packed_word_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x3c203c20u);
    packed_word_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x3c203c20u);
    packed_word_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x3c203c20u);
    packed_word_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x3c203c20u);
    packed_word_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_49), 0x3e473e47u);
    packed_word_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_51), 0x3e473e47u);
    packed_word_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x3c203c20u);
    packed_word_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3c203c20u);
    packed_word_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3c203c20u);
    packed_word_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x3c203c20u);
    packed_word_48 = (packed_word_15 << 0x5u) + 0x7ff88000u;
    packed_word_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_36), 0x3e473e47u);
    packed_word_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_37), 0x3e473e47u);
    packed_word_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_38), 0x3e473e47u);
    packed_word_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_4), 0x3e473e47u);
    packed_word_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_52), 0x3e473e47u);
    packed_word_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_8), 0x3e473e47u);
    packed_word_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_9), 0x3e473e47u);
    packed_word_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_word_55), 0x3e473e47u);
    packed_word_0 = (packed_word_49 << 0x5u) + 0x7ff88000u;
    packed_word_37 = (packed_word_51 << 0x5u) + 0x7ff88000u;
    packed_word_5 = add_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_38 = (packed_word_36 << 0x5u) + 0x7ff88000u;
    packed_word_39 = (packed_word_39 << 0x5u) + 0x7ff88000u;
    packed_word_53 = (packed_word_53 << 0x5u) + 0x7ff88000u;
    packed_word_54 = (packed_word_54 << 0x5u) + 0x7ff88000u;
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                    select_half_operand<2, false, false>(packed_word_37));
    packed_word_36 = (packed_word_52 << 0x5u) + 0x7ff88000u;
    packed_word_49 = (packed_word_8 << 0x5u) + 0x7ff88000u;
    packed_word_51 = (packed_word_9 << 0x5u) + 0x7ff88000u;
    packed_word_52 = (packed_word_55 << 0x5u) + 0x7ff88000u;
    packed_word_11 = add_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                    select_half_operand<2, false, false>(packed_word_5));
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x5070ull, packed_word_4, packed_word_5,
        packed_word_6, packed_word_7, memory_ranges, error_counts);
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                   select_half_operand<2, false, false>(packed_word_53));
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                   select_half_operand<2, false, false>(packed_word_54));
    packed_word_14 = add_half_pairs(select_half_operand<2, false, false>(packed_word_13),
                                    select_half_operand<2, false, false>(packed_word_14));
    packed_word_13 = add_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                    select_half_operand<2, false, false>(packed_word_51));
    packed_word_15 = add_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                    select_half_operand<2, false, false>(packed_word_52));
    packed_word_8 = add_half_pairs(select_half_operand<2, false, false>(packed_word_10),
                                   select_half_operand<2, false, false>(packed_word_8));
    packed_word_9 = add_half_pairs(select_half_operand<2, false, false>(packed_word_11),
                                   select_half_operand<2, false, false>(packed_word_9));
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_12),
                                    select_half_operand<2, false, false>(packed_word_13));
    packed_word_15 = add_half_pairs(select_half_operand<2, false, false>(packed_word_14),
                                    select_half_operand<2, false, false>(packed_word_15));
    packed_word_10 = condition_6 ? packed_word_9 : packed_word_8;
    packed_word_8 = condition_6 ? packed_word_8 : packed_word_9;
    packed_word_9 = condition_6 ? packed_word_15 : packed_word_12;
    packed_word_15 = condition_6 ? packed_word_12 : packed_word_15;
    packed_word_13 = !condition_0 ? packed_word_10 : packed_word_9;
    packed_word_12 = !condition_0 ? packed_word_8 : packed_word_15;
    packed_word_55 = !condition_0 ? packed_word_9 : packed_word_10;
    packed_word_14 = !condition_0 ? packed_word_15 : packed_word_8;
    load_four_boundary_words<CheckBounds, 0>(
        combine_words(packed_word_162, packed_word_163) + 0x5270ull, packed_word_8, packed_word_9,
        packed_word_10, packed_word_11, memory_ranges, error_counts);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_2), "r"(0x1fu), "r"(__activemask()));
        packed_word_13 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_12), "r"(packed_word_153), "r"(0x1fu), "r"(__activemask()));
        packed_word_12 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_55), "r"(packed_word_152), "r"(0x1fu), "r"(__activemask()));
        packed_word_55 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_14), "r"(packed_word_151), "r"(0x1fu), "r"(__activemask()));
        packed_word_14 = loaded_word;
    }
    packed_word_15 = !condition_1 ? packed_word_13 : packed_word_12;
    packed_word_66 = !condition_1 ? packed_word_12 : packed_word_13;
    packed_word_68 = !condition_1 ? packed_word_55 : packed_word_14;
    packed_word_77 = !condition_1 ? packed_word_14 : packed_word_55;
    packed_word_76 = !condition_2 ? packed_word_15 : packed_word_68;
    packed_word_85 = !condition_2 ? packed_word_66 : packed_word_77;
    packed_word_15 = !condition_2 ? packed_word_68 : packed_word_15;
    packed_word_66 = !condition_2 ? packed_word_77 : packed_word_66;
    packed_word_76 = add_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                    select_half_operand<2, false, false>(packed_word_85));
    packed_word_76 = add_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                    select_half_operand<2, false, false>(packed_word_15));
    packed_word_76 = add_half_pairs(select_half_operand<2, false, false>(packed_word_76),
                                    select_half_operand<2, false, false>(packed_word_66));
    packed_word_76 = add_half_pairs(select_half_operand<0, false, false>(packed_word_76),
                                    select_half_operand<1, false, false>(packed_word_76));
    packed_word_76 = permute_signed(packed_word_76, packed_word_76, 0x5410u);
    packed_word_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_word_76), 0x4100410u);
    packed_word_2 = half_to_float_bits((packed_word_76 & 65535u) * 65537u);
    packed_word_76 = half_to_float_bits((packed_word_76 >> 16) * 65537u);
    packed_word_2 = reciprocal_bits(packed_word_2);
    packed_word_14 = packed_word_150 + 0x8u;
    packed_word_13 = reciprocal_bits(packed_word_76);
    packed_word_15 = uint32_t(int32_t(packed_word_14) >> 0x1fu);
    packed_word_15 = uint32_t(combine_words(packed_word_15, 0u) << 0x5u >> 32) + packed_word_14;
    packed_word_15 = packed_word_15 & 0xffffffe0u;
    packed_word_55 = uint32_t(int64_t(int32_t(packed_word_154)) * (1ll << 0x1eu) >> 32) + 0x18u;
    packed_word_12 = packed_word_150 + (0u - packed_word_148);
    packed_word_14 = packed_word_14 + (0u - packed_word_15);
    packed_word_13 = pack_half(packed_word_13, packed_word_2);
    packed_word_2 = uint32_t(int64_t(int32_t(packed_word_154)) * (1ll << 0x1eu) >> 32) + 0x10u;
    packed_word_66 = uint32_t(int32_t(packed_word_55) >> 0x1fu);
    packed_word_15 = uint32_t(int32_t(packed_word_2) >> 0x1fu);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_12), "r"(0x1fu), "r"(__activemask()));
        packed_word_12 = loaded_word;
    }
    packed_word_66 = uint32_t(combine_words(packed_word_66, 0u) << 0x5u >> 32) + packed_word_55;
    packed_word_15 = uint32_t(combine_words(packed_word_15, 0u) << 0x5u >> 32) + packed_word_2;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_14), "r"(0x1fu), "r"(__activemask()));
        packed_word_14 = loaded_word;
    }
    packed_word_66 = packed_word_66 & 0xffffffe0u;
    packed_word_15 = packed_word_15 & 0xffffffe0u;
    packed_word_66 = packed_word_55 + (0u - packed_word_66);
    packed_word_68 = packed_word_2 + (0u - packed_word_15);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_66), "r"(0x1fu), "r"(__activemask()));
        packed_word_98 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_13), "r"(packed_word_68), "r"(0x1fu), "r"(__activemask()));
        packed_word_68 = loaded_word;
    }
    packed_word_2 = permute_signed(packed_word_12, packed_word_12, 0x5410u);
    packed_word_55 = permute_signed(packed_word_14, packed_word_14, 0x5410u);
    packed_word_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_61),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_65 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_65),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_63 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_63),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_67),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_61),
                                   select_half_operand<2, false, false>(0u));
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_63),
                                   select_half_operand<2, false, false>(0u));
    packed_word_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_44),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_65 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_65),
                                   select_half_operand<2, false, false>(0u));
    packed_word_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_42),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_67),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_68 = permute_signed(packed_word_68, packed_word_68, 0x5410u);
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_12),
                                   select_half_operand<2, false, false>(packed_word_61));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_42),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(packed_word_65));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_40));
    packed_word_98 = permute_signed(packed_word_98, packed_word_98, 0x5410u);
    packed_word_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_16),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_19 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_19),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_62),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_69),
                                         select_half_operand<2, false, false>(packed_word_98));
    multiply_accumulate_half(packed_word_84, packed_word_85, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_90, packed_word_91, 0u,
                             0u);
    packed_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_19),
                                   select_half_operand<2, false, false>(0u));
    packed_word_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_62),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_126, packed_word_127, 0u,
                             0u);
    packed_word_43 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                         select_half_operand<2, false, false>(packed_word_2));
    packed_word_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                         select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_66, packed_word_67, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_94, packed_word_95, 0u,
                             0u);
    packed_word_38 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_38),
                                         select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_17),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_16),
                                   select_half_operand<2, false, false>(0u));
    packed_word_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_18),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_69),
                                   select_half_operand<2, false, false>(0u));
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_64),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_12),
                                   select_half_operand<2, false, false>(packed_word_17));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_13),
                                   select_half_operand<2, false, false>(packed_word_62));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(packed_word_19));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_16));
    packed_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                        select_half_operand<2, false, false>(packed_word_2));
    packed_word_48 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_48),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_54),
                                         select_half_operand<2, false, false>(packed_word_55));
    multiply_accumulate_half(packed_word_126, packed_word_127, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_126, packed_word_127, 0u,
                             0u);
    packed_word_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_48),
                                   select_half_operand<2, false, false>(0u));
    packed_word_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_2),
                                   select_half_operand<2, false, false>(0u));
    packed_word_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_54),
                                   select_half_operand<2, false, false>(0u));
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_92, packed_word_93, 0u,
                             0u);
    packed_word_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_60),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                         select_half_operand<2, false, false>(packed_word_98));
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_90, packed_word_91, 0u,
                             0u);
    packed_word_52 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_52),
                                         select_half_operand<2, false, false>(packed_word_98));
    multiply_accumulate_half(packed_word_94, packed_word_95, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_94, packed_word_95, 0u,
                             0u);
    packed_word_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(0u));
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                         select_half_operand<2, false, false>(packed_word_55));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_38),
                                   select_half_operand<2, false, false>(packed_word_17));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(packed_word_12));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_13),
                                   select_half_operand<2, false, false>(packed_word_48));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_54));
    packed_word_45 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_51),
                                   select_half_operand<2, false, false>(0u));
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_36),
                                         select_half_operand<2, false, false>(packed_word_68));
    packed_word_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_52),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_0),
                                        select_half_operand<2, false, false>(packed_word_98));
    multiply_accumulate_half(packed_word_16, packed_word_17, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_78, packed_word_79,
                             packed_word_84, packed_word_85);
    multiply_accumulate_half(packed_word_76, packed_word_77, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_56, packed_word_57,
                             packed_word_76, packed_word_77);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_58, packed_word_59,
                             packed_word_66, packed_word_67);
    multiply_accumulate_half(packed_word_86, packed_word_87, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_96, packed_word_97,
                             packed_word_86, packed_word_87);
    packed_word_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                         select_half_operand<2, false, false>(packed_word_98));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_60),
                                   select_half_operand<2, false, false>(0u));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_37),
                                   select_half_operand<2, false, false>(0u));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_45),
                                   select_half_operand<2, false, false>(packed_word_12));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_0),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_36),
                                   select_half_operand<2, false, false>(packed_word_51));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_52));
    multiply_accumulate_half(packed_word_126, packed_word_127, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_56, packed_word_57,
                             packed_word_126, packed_word_127);
    multiply_accumulate_half(packed_word_92, packed_word_93, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_96, packed_word_97,
                             packed_word_92, packed_word_93);
    multiply_accumulate_half(packed_word_90, packed_word_91, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_78, packed_word_79,
                             packed_word_90, packed_word_91);
    multiply_accumulate_half(packed_word_14, packed_word_15, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_58, packed_word_59,
                             packed_word_94, packed_word_95);
    packed_word_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_86),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_77),
                                  select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_87),
                                  select_half_operand<2, false, false>(0u));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_76),
                                   select_half_operand<2, false, false>(0u));
    packed_word_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_18),
                                   select_half_operand<2, false, false>(packed_word_37));
    packed_word_37 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_16),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_17),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_19),
                                   select_half_operand<2, false, false>(packed_word_2));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_126),
                                   select_half_operand<2, false, false>(0u));
    packed_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_127),
                                  select_half_operand<2, false, false>(0u));
    packed_word_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_92),
                                   select_half_operand<2, false, false>(0u));
    packed_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_93),
                                  select_half_operand<2, false, false>(0u));
    packed_word_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_30),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_90),
                                   select_half_operand<2, false, false>(packed_word_13));
    packed_word_31 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_31),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_91),
                                   select_half_operand<2, false, false>(packed_word_0));
    packed_word_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_28),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_14),
                                   select_half_operand<2, false, false>(packed_word_33));
    packed_word_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                         select_half_operand<2, false, false>(packed_word_124));
    packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_15),
                                   select_half_operand<2, false, false>(packed_word_2));
    multiply_accumulate_half(packed_word_30, packed_word_31, packed_word_16, packed_word_17,
                             packed_word_18, packed_word_19, packed_word_6, packed_word_7,
                             packed_word_30, packed_word_31);
    packed_word_0 = uint32_t(uint32_t(packed_word_165) >> 0x1u);
    packed_word_2 = uint32_t(combine_words(packed_word_150, 0u) << 0x2u >> 32) + packed_word_150;
    packed_word_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_34),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_35 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_35),
                                         select_half_operand<2, false, false>(packed_word_122));
    multiply_accumulate_half(packed_word_6, packed_word_7, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_6, packed_word_7,
                             packed_word_28, packed_word_29);
    packed_word_29 = packed_word_165 & 0x13u;
    packed_word_29 = packed_word_29 | 0x4u & packed_word_0;
    packed_word_0 = packed_word_114 * 0x4u;
    packed_word_26 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_26),
                                         select_half_operand<2, false, false>(packed_word_88));
    packed_word_2 = packed_word_2 & 0xfffffffcu;
    packed_word_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_27),
                                         select_half_operand<2, false, false>(packed_word_88));
    packed_word_33 = packed_word_0 & 0x7ffffffcu;
    packed_word_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                         select_half_operand<2, false, false>(packed_word_89));
    {
        uint32_t loaded_word;
        loaded_word = packed_word_165 & 0x4u;
        condition_0 = loaded_word != 0;
    }
    packed_word_36 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_32),
                                         select_half_operand<2, false, false>(packed_word_122));
    packed_word_150 = packed_word_33 + packed_word_150 + (0u - packed_word_2);
    packed_word_32 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                         select_half_operand<2, false, false>(packed_word_88));
    multiply_accumulate_half(packed_word_34, packed_word_35, packed_word_16, packed_word_17,
                             packed_word_18, packed_word_19, packed_word_4, packed_word_5,
                             packed_word_34, packed_word_35);
    packed_word_33 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23),
                                         select_half_operand<2, false, false>(packed_word_88));
    packed_word_88 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_20),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_22 = !condition_0 ? packed_word_70 : packed_word_72;
    packed_word_89 = multiply_half_pairs(select_half_operand<2, false, false>(packed_word_21),
                                         select_half_operand<2, false, false>(packed_word_89));
    packed_word_21 = !condition_0 ? packed_word_106 : packed_word_74;
    packed_word_23 = !condition_0 ? packed_word_72 : packed_word_70;
    packed_word_2 = !condition_0 ? packed_word_73 : packed_word_71;
    multiply_accumulate_half(packed_word_26, packed_word_27, packed_word_16, packed_word_17,
                             packed_word_18, packed_word_19, packed_word_8, packed_word_9,
                             packed_word_26, packed_word_27);
    multiply_accumulate_half(packed_word_18, packed_word_19, packed_word_16, packed_word_17,
                             packed_word_18, packed_word_19, packed_word_10, packed_word_11,
                             packed_word_24, packed_word_25);
    packed_word_17 = !condition_0 ? packed_word_71 : packed_word_73;
    packed_word_16 = !condition_0 ? packed_word_107 : packed_word_75;
    packed_word_24 = !condition_2 ? packed_word_23 : packed_word_2;
    packed_word_23 = !condition_2 ? packed_word_2 : packed_word_23;
    packed_word_25 = !condition_2 ? packed_word_22 : packed_word_17;
    packed_word_22 = !condition_2 ? packed_word_17 : packed_word_22;
    multiply_accumulate_half(packed_word_4, packed_word_5, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_4, packed_word_5,
                             packed_word_36, packed_word_37);
    packed_word_2 = !condition_0 ? packed_word_80 : packed_word_108;
    packed_word_37 = !condition_2 ? packed_word_21 : packed_word_16;
    packed_word_21 = !condition_2 ? packed_word_16 : packed_word_21;
    multiply_accumulate_half(packed_word_16, packed_word_17, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_8, packed_word_9,
                             packed_word_32, packed_word_33);
    packed_word_9 = !condition_0 ? packed_word_81 : packed_word_109;
    packed_word_43 = !condition_2 ? packed_word_2 : packed_word_9;
    packed_word_45 = !condition_2 ? packed_word_9 : packed_word_2;
    packed_word_9 = blockIdx.y;
    multiply_accumulate_half(packed_word_10, packed_word_11, packed_word_12, packed_word_13,
                             packed_word_14, packed_word_15, packed_word_10, packed_word_11,
                             packed_word_88, packed_word_89);
    packed_word_28 = packed_word_165 * 0x2u;
    packed_word_8 = !condition_0 ? packed_word_82 : packed_word_110;
    packed_word_2 = !condition_0 ? packed_word_34 : packed_word_4;
    packed_word_28 = packed_word_29 | 0x8u & packed_word_28;
    packed_word_29 = !condition_0 ? packed_word_83 : packed_word_111;
    packed_word_13 = !condition_0 ? packed_word_35 : packed_word_5;
    packed_word_113 = packed_word_113 + 0x1u;
    packed_word_12 = !condition_0 ? packed_word_30 : packed_word_6;
    packed_word_39 = !condition_0 ? packed_word_31 : packed_word_7;
    packed_word_53 = !condition_2 ? packed_word_8 : packed_word_29;
    packed_word_55 = !condition_2 ? packed_word_29 : packed_word_8;
    packed_word_8 = !condition_0 ? packed_word_4 : packed_word_34;
    packed_word_15 = !condition_0 ? packed_word_5 : packed_word_35;
    packed_word_29 = !condition_2 ? packed_word_2 : packed_word_13;
    packed_word_33 = !condition_2 ? packed_word_13 : packed_word_2;
    packed_word_2 = !condition_0 ? packed_word_6 : packed_word_30;
    packed_word_13 = !condition_0 ? packed_word_7 : packed_word_31;
    condition_5 = int32_t(packed_word_113) >= int32_t(uniform_word_4) || condition_5;
    packed_word_41 = !condition_2 ? packed_word_12 : packed_word_39;
    packed_word_32 = !condition_2 ? packed_word_8 : packed_word_15;
    packed_word_36 = !condition_2 ? packed_word_15 : packed_word_8;
    packed_word_39 = !condition_2 ? packed_word_39 : packed_word_12;
    packed_word_8 = !condition_0 ? packed_word_26 : packed_word_16;
    packed_word_15 = !condition_0 ? packed_word_27 : packed_word_17;
    packed_word_12 = !condition_0 ? packed_word_16 : packed_word_26;
    packed_word_47 = !condition_0 ? packed_word_17 : packed_word_27;
    packed_word_40 = !condition_2 ? packed_word_2 : packed_word_13;
    packed_word_42 = !condition_2 ? packed_word_13 : packed_word_2;
    packed_word_2 = packed_word_9 * 0x4u;
    packed_word_59 = packed_word_165;
    uniform_word_6 = uint32_t(combine_words(uniform_word_8, 0u) << 0x1u >> 32) + uniform_word_8;
    uniform_word_7 = uint32_t(combine_words(uniform_word_9, 0u) << 0x1u >> 32) + uniform_word_9;
    packed_word_20 = !condition_0 ? packed_word_74 : packed_word_106;
    packed_word_75 = !condition_0 ? packed_word_75 : packed_word_107;
    packed_word_80 = !condition_0 ? packed_word_108 : packed_word_80;
    packed_word_81 = !condition_0 ? packed_word_109 : packed_word_81;
    packed_word_82 = !condition_0 ? packed_word_110 : packed_word_82;
    packed_word_83 = !condition_0 ? packed_word_111 : packed_word_83;
    packed_word_49 = !condition_2 ? packed_word_8 : packed_word_15;
    packed_word_51 = !condition_2 ? packed_word_15 : packed_word_8;
    packed_word_46 = !condition_2 ? packed_word_12 : packed_word_47;
    packed_word_48 = !condition_2 ? packed_word_47 : packed_word_12;
    packed_word_154 = packed_word_154 & 0xfffffffcu;
    packed_word_59 =
        uint32_t(int64_t(int32_t(packed_word_59)) * (1ll << 0x1cu) >> 32) + packed_word_2;
    packed_word_8 = !condition_0 ? packed_word_18 : packed_word_10;
    packed_word_13 = !condition_0 ? packed_word_19 : packed_word_11;
    packed_word_12 = !condition_0 ? packed_word_10 : packed_word_18;
    packed_word_15 = !condition_0 ? packed_word_11 : packed_word_19;
    uniform_word_6 = uint32_t(int32_t(uniform_word_6) >> 0x1u);
    uniform_word_7 = uint32_t(int32_t(uniform_word_7) >> 0x1u);
    condition_3 = int32_t(packed_word_113) >= int32_t(uniform_word_4) || condition_3;
    packed_word_38 = !condition_2 ? packed_word_20 : packed_word_75;
    packed_word_44 = !condition_2 ? packed_word_80 : packed_word_81;
    packed_word_50 = !condition_2 ? packed_word_82 : packed_word_83;
    packed_word_20 = !condition_2 ? packed_word_75 : packed_word_20;
    packed_word_80 = !condition_2 ? packed_word_81 : packed_word_80;
    packed_word_82 = !condition_2 ? packed_word_83 : packed_word_82;
    packed_word_47 = !condition_2 ? packed_word_8 : packed_word_13;
    packed_word_57 = !condition_2 ? packed_word_13 : packed_word_8;
    packed_word_52 = !condition_2 ? packed_word_12 : packed_word_15;
    packed_word_54 = !condition_2 ? packed_word_15 : packed_word_12;
    packed_word_154 = 0u - packed_word_154 + packed_word_165;
    packed_word_61 = packed_word_59 + 0x2u;
    if (!condition_5) {
        packed_word_12 = parameters[54];
        packed_word_13 = parameters[55];
        packed_word_3 = packed_word_113 * uniform_word_5 + packed_word_3;
        packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_19),
                                       select_half_operand<2, false, false>(0u));
        packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_31),
                                      select_half_operand<2, false, false>(0u));
        packed_word_3 = packed_word_3 * 0x80u;
        packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_18),
                                       select_half_operand<2, false, false>(0u));
        packed_word_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_27),
                                       select_half_operand<2, false, false>(packed_word_14));
        packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_35),
                                      select_half_operand<2, false, false>(packed_word_8));
        packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_26),
                                       select_half_operand<2, false, false>(packed_word_15));
        packed_word_15 = packed_word_27;
        split_address_words(packed_word_12, packed_word_13,
                            uint64_t(uint32_t(packed_word_3)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));
        packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_30),
                                      select_half_operand<2, false, false>(0u));
        packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_34),
                                      select_half_operand<2, false, false>(packed_word_3));
        split_address_words(packed_word_18, packed_word_19,
                            int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_12, packed_word_13));
        packed_word_13 = packed_word_8;
        packed_word_12 = packed_word_3;
        store_four_words<CheckBounds>(combine_words(packed_word_18, packed_word_19), packed_word_12,
                                      packed_word_13, packed_word_14, packed_word_15, memory_ranges,
                                      error_counts);
    }
    if (!condition_3) {
        packed_word_12 = parameters[54];
        packed_word_13 = parameters[55];
        packed_word_112 = packed_word_113 * uniform_word_5 + packed_word_112;
        packed_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_11),
                                       select_half_operand<2, false, false>(0u));
        packed_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_10),
                                       select_half_operand<2, false, false>(0u));
        packed_word_3 = packed_word_112 * 0x80u;
        packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_7),
                                      select_half_operand<2, false, false>(0u));
        packed_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_17),
                                      select_half_operand<2, false, false>(packed_word_14));
        packed_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_5),
                                      select_half_operand<2, false, false>(packed_word_8));
        split_address_words(packed_word_10, packed_word_11,
                            uint64_t(uint32_t(packed_word_3)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));
        packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_6),
                                      select_half_operand<2, false, false>(0u));
        packed_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_16),
                                      select_half_operand<2, false, false>(packed_word_15));
        packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_4),
                                      select_half_operand<2, false, false>(packed_word_3));
        split_address_words(packed_word_10, packed_word_11,
                            int64_t(int32_t(packed_word_165)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_word_10, packed_word_11));
        store_four_words<CheckBounds>(combine_words(packed_word_10, packed_word_11), packed_word_4,
                                      packed_word_5, packed_word_6, packed_word_7, memory_ranges,
                                      error_counts);
    }
    packed_word_4 = 0x10u ^ packed_word_28;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_25), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_25 = loaded_word;
    }
    packed_word_5 = 0x14u ^ packed_word_28;
    packed_word_3 = 0x4u ^ packed_word_28;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_22), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_22 = loaded_word;
    }
    condition_0 = int32_t(packed_word_59) >= int32_t(uniform_word_6);
    condition_1 = int32_t(packed_word_150) >= int32_t(uniform_word_7);
    condition_2 = int32_t(packed_word_61) >= int32_t(uniform_word_6);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_23), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_23 = loaded_word;
    }
    condition_0 = int32_t(packed_word_59) < int32_t(0u) || condition_0;
    condition_1 = int32_t(packed_word_150) < int32_t(0u) || condition_1;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_24), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_24 = loaded_word;
    }
    condition_2 = int32_t(packed_word_61) < int32_t(0u) || condition_2;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_29), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_29 = loaded_word;
    }
    condition_0 = condition_0 | condition_1;
    condition_2 = condition_2 | condition_1;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_41), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_41 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_32), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_32 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_40), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_40 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_33), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_33 = loaded_word;
    }
    packed_word_12 = add_half_pairs(select_half_operand<2, false, false>(packed_word_22),
                                    select_half_operand<2, false, false>(packed_word_23));
    if (!condition_2) {
        packed_word_7 = packed_word_61 + uniform_word_6;

        packed_word_11 = packed_word_61 * uniform_word_7 + packed_word_150;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_36), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_36 = loaded_word;
    }
    packed_word_24 = add_half_pairs(select_half_operand<2, false, false>(packed_word_25),
                                    select_half_operand<2, false, false>(packed_word_24));
    if (!condition_2) {
        packed_word_17 = packed_word_7 * uniform_word_7 + packed_word_150;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_39), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_39 = loaded_word;
    }
    packed_word_24 = add_half_pairs(select_half_operand<2, false, false>(packed_word_24),
                                    select_half_operand<2, false, false>(packed_word_12));
    if (!condition_2) {
        packed_word_11 = packed_word_11 * 0x4u + packed_word_154;
    }
    packed_word_12 = parameters[62];
    packed_word_13 = parameters[63];
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_42), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_42 = loaded_word;
    }
    packed_word_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_24), 0x34003400u);
    if (!condition_2) {
        packed_word_17 = packed_word_17 * 0x4u + packed_word_154;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_37), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_37 = loaded_word;
    }
    packed_word_29 = add_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                    select_half_operand<2, false, false>(packed_word_32));
    packed_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_24),
                                   select_half_operand<2, false, false>(0u));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_38), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_38 = loaded_word;
    }
    packed_word_40 = add_half_pairs(select_half_operand<2, false, false>(packed_word_41),
                                    select_half_operand<2, false, false>(packed_word_40));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_21), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_21 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_20), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_20 = loaded_word;
    }
    packed_word_33 = add_half_pairs(select_half_operand<2, false, false>(packed_word_33),
                                    select_half_operand<2, false, false>(packed_word_36));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_43), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_43 = loaded_word;
    }
    packed_word_29 = add_half_pairs(select_half_operand<2, false, false>(packed_word_29),
                                    select_half_operand<2, false, false>(packed_word_33));
    condition_1 = combine_words(packed_word_12, packed_word_13) != 0ull;
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_53), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_53 = loaded_word;
    }
    packed_word_39 = add_half_pairs(select_half_operand<2, false, false>(packed_word_39),
                                    select_half_operand<2, false, false>(packed_word_42));
    packed_word_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_29), 0x34003400u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_50), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_50 = loaded_word;
    }
    packed_word_39 = add_half_pairs(select_half_operand<2, false, false>(packed_word_40),
                                    select_half_operand<2, false, false>(packed_word_39));
    packed_word_37 = add_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                    select_half_operand<2, false, false>(packed_word_38));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_55), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_55 = loaded_word;
    }
    packed_word_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_39), 0x34003400u);
    packed_word_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_29),
                                   select_half_operand<2, false, false>(0u));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_82), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_82 = loaded_word;
    }
    packed_word_20 = add_half_pairs(select_half_operand<2, false, false>(packed_word_21),
                                    select_half_operand<2, false, false>(packed_word_20));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_44), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_44 = loaded_word;
    }
    packed_word_20 = add_half_pairs(select_half_operand<2, false, false>(packed_word_37),
                                    select_half_operand<2, false, false>(packed_word_20));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_45), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_45 = loaded_word;
    }
    packed_word_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_20), 0x34003400u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_80), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_80 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_49), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_49 = loaded_word;
    }
    packed_word_50 = add_half_pairs(select_half_operand<2, false, false>(packed_word_53),
                                    select_half_operand<2, false, false>(packed_word_50));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_47), "r"(packed_word_28), "r"(0x1fu), "r"(__activemask()));
        packed_word_47 = loaded_word;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_46), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_46 = loaded_word;
    }
    packed_word_55 = add_half_pairs(select_half_operand<2, false, false>(packed_word_55),
                                    select_half_operand<2, false, false>(packed_word_82));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_52), "r"(packed_word_3), "r"(0x1fu), "r"(__activemask()));
        packed_word_52 = loaded_word;
    }
    packed_word_43 = add_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                    select_half_operand<2, false, false>(packed_word_44));
    packed_word_50 = add_half_pairs(select_half_operand<2, false, false>(packed_word_50),
                                    select_half_operand<2, false, false>(packed_word_55));
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_51), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_51 = loaded_word;
    }
    packed_word_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_50), 0x34003400u);
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_57), "r"(packed_word_4), "r"(0x1fu), "r"(__activemask()));
        packed_word_57 = loaded_word;
    }
    packed_word_45 = add_half_pairs(select_half_operand<2, false, false>(packed_word_45),
                                    select_half_operand<2, false, false>(packed_word_80));
    if (!condition_0) {
        packed_word_3 = packed_word_59 + uniform_word_6;

        packed_word_59 = packed_word_59 * uniform_word_7 + packed_word_150;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_48), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_48 = loaded_word;
    }
    packed_word_43 = add_half_pairs(select_half_operand<2, false, false>(packed_word_43),
                                    select_half_operand<2, false, false>(packed_word_45));
    packed_word_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_50),
                                   select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        packed_word_3 = packed_word_3 * uniform_word_7 + packed_word_150;
    }
    {
        uint32_t loaded_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(loaded_word)
            : "r"(packed_word_54), "r"(packed_word_5), "r"(0x1fu), "r"(__activemask()));
        packed_word_54 = loaded_word;
    }
    packed_word_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_43), 0x34003400u);
    packed_word_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_39),
                                  select_half_operand<2, false, false>(0u));
    if (!condition_0) {
        packed_word_15 = packed_word_3 * 0x4u + packed_word_154;
    }
    packed_word_46 = add_half_pairs(select_half_operand<2, false, false>(packed_word_49),
                                    select_half_operand<2, false, false>(packed_word_46));
    if (!condition_0) {
        packed_word_7 = packed_word_59 * 0x4u + packed_word_154;
    }
    packed_word_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_20),
                                  select_half_operand<2, false, false>(0u));
    packed_word_47 = add_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                    select_half_operand<2, false, false>(packed_word_52));
    packed_word_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_43),
                                   select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        packed_word_29 = permute_signed(packed_word_29, packed_word_4, 0x5410u);
    }
    if (!condition_0) {
        split_address_words(packed_word_4, packed_word_5,
                            int64_t(int32_t(packed_word_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));

        packed_word_3 = permute_signed(packed_word_24, packed_word_3, 0x5410u);

        packed_word_43 = permute_signed(packed_word_43, packed_word_50, 0x5410u);
    }
    if (!condition_2) {
        split_address_words(packed_word_6, packed_word_7,
                            int64_t(int32_t(packed_word_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));
    }
    if (!condition_0) {
        write_boundary_word<CheckBounds>(combine_words(packed_word_4, packed_word_5), packed_word_3,
                                         memory_ranges, error_counts);
    }
    packed_word_23 = add_half_pairs(select_half_operand<2, false, false>(packed_word_51),
                                    select_half_operand<2, false, false>(packed_word_48));
    if (!condition_0) {
        split_address_words(packed_word_10, packed_word_11,
                            int64_t(int32_t(packed_word_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));
    }
    if (!condition_2) {
        write_boundary_word<CheckBounds>(combine_words(packed_word_6, packed_word_7),
                                         packed_word_29, memory_ranges, error_counts);
    }
    packed_word_54 = add_half_pairs(select_half_operand<2, false, false>(packed_word_57),
                                    select_half_operand<2, false, false>(packed_word_54));
    if (!condition_2) {
        split_address_words(packed_word_12, packed_word_13,
                            int64_t(int32_t(packed_word_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_word_12, packed_word_13));
    }
    packed_word_23 = add_half_pairs(select_half_operand<2, false, false>(packed_word_46),
                                    select_half_operand<2, false, false>(packed_word_23));
    if (!condition_0) {
        write_boundary_word<CheckBounds>(combine_words(packed_word_10, packed_word_11),
                                         packed_word_43, memory_ranges, error_counts);
    }
    packed_word_47 = add_half_pairs(select_half_operand<2, false, false>(packed_word_47),
                                    select_half_operand<2, false, false>(packed_word_54));
    packed_word_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_23), 0x34003400u);
    packed_word_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_word_47), 0x34003400u);
    packed_word_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_23),
                                   select_half_operand<2, false, false>(0u));
    packed_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_word_47),
                                  select_half_operand<2, false, false>(0u));
    if (!condition_2) {
        packed_word_23 = permute_signed(packed_word_23, packed_word_8, 0x5410u);

        write_boundary_word<CheckBounds>(combine_words(packed_word_12, packed_word_13),
                                         packed_word_23, memory_ranges, error_counts);
    }
    if (!condition_1) {
        return;
    }
    uniform_word_6 = parameters[64];
    uniform_word_7 = parameters[65];
    uniform_word_4 = uniform_word_8 + 0x1u;
    uniform_word_5 = uniform_word_9 + 0x1u;
    uniform_word_4 = uint32_t(combine_words(uniform_word_4, 0u) << 0x1u >> 32) + uniform_word_4;
    uniform_word_5 = uint32_t(combine_words(uniform_word_5, 0u) << 0x1u >> 32) + uniform_word_5;
    uniform_word_4 = uint32_t(int32_t(uniform_word_4) >> 0x1u);
    uniform_word_5 = uint32_t(int32_t(uniform_word_5) >> 0x1u);
    packed_word_11 = uniform_word_7;
    packed_word_25 = uniform_word_6;
    condition_0 = int32_t(packed_word_11) <= int32_t(uniform_word_5);
    condition_1 = int32_t(packed_word_25) <= int32_t(uniform_word_4);
    if (condition_0) {
        return;
    }
    uniform_word_8 = blockDim.x;
    packed_word_8 = threadIdx.z;
    packed_word_3 = packed_word_2 + 0x4u;
    packed_word_4 = packed_word_0 + 0x4u;
    uniform_word_9 = blockDim.y;
    packed_word_15 = threadIdx.y;
    packed_word_17 = blockDim.z;
    condition_0 = int32_t(packed_word_3) > int32_t(uniform_word_4);
    uniform_word_6 = gridDim.y;
    packed_word_19 = threadIdx.x;
    condition_1 = int32_t(packed_word_4) > int32_t(uniform_word_5);
    packed_word_4 = packed_word_11 * 0x4u;
    condition_0 = int32_t(packed_word_2) < int32_t(packed_word_25) && condition_0;
    condition_1 = int32_t(packed_word_0) < int32_t(packed_word_11) && condition_1;
    packed_word_3 = packed_word_4 * packed_word_25;
    condition_0 = condition_0 | condition_1;
    uniform_word_7 = uniform_word_8 * uniform_word_9;
    uniform_word_6 = (uniform_word_6 << 0x2u) + 0xfffffffcu;
    uniform_word_6 = uniform_word_6 & 0x7ffffffcu;
    packed_word_5 = packed_word_17 * uniform_word_7;
    uniform_word_6 = uniform_word_6 + 0x4u;
    packed_word_6 = packed_word_8 * uniform_word_9 + packed_word_15;
    packed_word_7 = min(int32_t(packed_word_25), int32_t(uniform_word_6));
    packed_word_6 = packed_word_6 * uniform_word_8 + packed_word_19;
    if (condition_0) {
        __syncthreads();
        condition_0 = uint32_t(packed_word_6) > uint32_t(0x3fu);
        uniform_word_12 = parameters[62];
        uniform_word_13 = parameters[63];

        uniform_word_14 = parameters[62];
        uniform_word_15 = parameters[63];
        if (!condition_0) {
            const uint32_t next_boundary_index =
                ((packed_word_8 + packed_word_17) * uniform_word_9 + packed_word_15) *
                    uniform_word_8 + packed_word_19;
            packed_word_20 = packed_word_6;
            packed_word_10 = boundary_tail_steps<false>(next_boundary_index, 64u, packed_word_5);
            packed_word_8 = packed_word_10 + 0x1u;
            {
                uint32_t loaded_word;
                loaded_word = packed_word_8 & 0x3u;
                packed_word_8 = loaded_word;
                condition_0 = loaded_word != 0;
            }
            if (condition_0) {
                packed_word_18 = parameters[64];
                packed_word_19 = parameters[65];
                packed_word_8 = 0u - packed_word_8;
                packed_word_20 = packed_word_6;
                do {
                    packed_word_11 = uint32_t(uint32_t(packed_word_20) >> 0x2u);

                    packed_word_15 =
                        uint32_t(combine_words(packed_word_20, 0u) << 0x1cu >> 32) + packed_word_2;
                    packed_word_11 = packed_word_11 & 0x3u;
                    packed_word_25 = packed_word_18;
                    packed_word_8 = packed_word_8 + 0x1u;
                    packed_word_14 = packed_word_0 + packed_word_11;
                    packed_word_11 = packed_word_19;
                    condition_1 = int32_t(packed_word_14) >= int32_t(uniform_word_5);
                    condition_0 = int32_t(packed_word_14) >= int32_t(packed_word_11);
                    condition_1 = int32_t(packed_word_15) < int32_t(uniform_word_4) && !condition_1;
                    condition_0 = int32_t(packed_word_15) >= int32_t(packed_word_25) || condition_0;
                    condition_0 = condition_0 | condition_1;
                    condition_1 = uint32_t(packed_word_8) != 0u;
                    if (!condition_0) {
                        packed_word_12 = packed_word_20 & 0x3u;
                        packed_word_14 = packed_word_14 * 0x4u;
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_12)) *
                                                    int64_t(int32_t(packed_word_3)) +
                                                0ull);
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_4)) *
                                                    int64_t(int32_t(packed_word_15)) +
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_15 = 0u;
                        split_address_words(packed_word_12, packed_word_13,
                                            combine_words(packed_word_12, packed_word_13) +
                                                packed_word_14);
                        const uint64_t byte_address_0 =
                            scaled_word_address(combine_words(uniform_word_12, uniform_word_13),
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_14 = uint32_t(byte_address_0);
                        packed_word_15 = uint32_t(byte_address_0 >> 32);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15), 0u, memory_ranges,
                            error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x4ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x8ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }

                    packed_word_20 = packed_word_5 + packed_word_20;
                } while (condition_1);
            }

            condition_0 = uint32_t(packed_word_10) >= uint32_t(0x3u);
            if (condition_0) {
                packed_word_34 = parameters[64];
                packed_word_35 = parameters[65];
                packed_word_10 = packed_word_5 + packed_word_20;
                packed_word_8 = packed_word_17 * uniform_word_9;
                packed_word_16 = packed_word_20 & 0x3u;
                packed_word_18 = packed_word_10 & 0x3u;
                packed_word_21 = packed_word_8 * uniform_word_8;
                split_address_words(
                    packed_word_16, packed_word_17,
                    int64_t(int32_t(packed_word_16)) * int64_t(int32_t(packed_word_3)) + 0ull);
                split_address_words(
                    packed_word_18, packed_word_19,
                    int64_t(int32_t(packed_word_18)) * int64_t(int32_t(packed_word_3)) + 0ull);
                packed_word_8 = packed_word_21 * 0x2u + packed_word_20;
                packed_word_22 = packed_word_21 * 0x3u + packed_word_20;
                do {
                    packed_word_12 = uint32_t(uint32_t(packed_word_20) >> 0x2u);

                    packed_word_14 = uint32_t(uint32_t(packed_word_10) >> 0x2u);
                    packed_word_13 = packed_word_12 & 0x3u;
                    packed_word_15 = packed_word_14 & 0x3u;
                    packed_word_11 = packed_word_35;
                    packed_word_14 = packed_word_0 + packed_word_13;
                    packed_word_27 =
                        uint32_t(combine_words(packed_word_20, 0u) << 0x1cu >> 32) + packed_word_2;
                    packed_word_25 = packed_word_34;
                    packed_word_12 = uint32_t(uint32_t(packed_word_8) >> 0x2u);
                    condition_2 = int32_t(packed_word_14) >= int32_t(uniform_word_5);
                    condition_0 = int32_t(packed_word_14) >= int32_t(packed_word_11);
                    packed_word_23 = uint32_t(uint32_t(packed_word_22) >> 0x2u);
                    packed_word_13 = packed_word_12 & 0x3u;
                    packed_word_23 = packed_word_23 & 0x3u;
                    condition_2 = int32_t(packed_word_27) < int32_t(uniform_word_4) && !condition_2;
                    condition_3 = int32_t(packed_word_27) >= int32_t(packed_word_25) || condition_0;
                    packed_word_24 = packed_word_0 + packed_word_15;
                    packed_word_26 = packed_word_0 + packed_word_13;
                    packed_word_28 = packed_word_0 + packed_word_23;
                    packed_word_29 =
                        uint32_t(combine_words(packed_word_10, 0u) << 0x1cu >> 32) + packed_word_2;
                    condition_2 = condition_3 | condition_2;
                    packed_word_31 =
                        uint32_t(combine_words(packed_word_8, 0u) << 0x1cu >> 32) + packed_word_2;
                    packed_word_33 =
                        uint32_t(combine_words(packed_word_22, 0u) << 0x1cu >> 32) + packed_word_2;
                    condition_4 = int32_t(packed_word_24) >= int32_t(packed_word_11);
                    condition_5 = int32_t(packed_word_24) >= int32_t(uniform_word_5);
                    condition_6 = int32_t(packed_word_26) >= int32_t(packed_word_11);
                    condition_0 = int32_t(packed_word_26) >= int32_t(uniform_word_5);
                    condition_1 = int32_t(packed_word_28) >= int32_t(packed_word_11);
                    condition_3 = int32_t(packed_word_28) >= int32_t(uniform_word_5);
                    condition_4 = int32_t(packed_word_29) >= int32_t(packed_word_25) || condition_4;
                    condition_5 = int32_t(packed_word_29) < int32_t(uniform_word_4) && !condition_5;
                    condition_6 = int32_t(packed_word_31) >= int32_t(packed_word_25) || condition_6;
                    condition_0 = int32_t(packed_word_31) < int32_t(uniform_word_4) && !condition_0;
                    condition_1 = int32_t(packed_word_33) >= int32_t(packed_word_25) || condition_1;
                    condition_3 = int32_t(packed_word_33) < int32_t(uniform_word_4) && !condition_3;
                    condition_4 = condition_4 | condition_5;
                    condition_0 = condition_6 | condition_0;
                    condition_1 = condition_1 | condition_3;
                    if (!condition_2) {
                        packed_word_14 = packed_word_14 * 0x4u;
                        packed_word_15 = 0u;
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_4)) *
                                                    int64_t(int32_t(packed_word_27)) +
                                                combine_words(packed_word_16, packed_word_17));
                        split_address_words(packed_word_12, packed_word_13,
                                            combine_words(packed_word_12, packed_word_13) +
                                                packed_word_14);
                        const uint64_t byte_address_1 =
                            scaled_word_address(combine_words(uniform_word_14, uniform_word_15),
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_14 = uint32_t(byte_address_1);
                        packed_word_15 = uint32_t(byte_address_1 >> 32);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15), 0u, memory_ranges,
                            error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x4ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x8ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }

                    if (!condition_4) {
                        packed_word_14 = packed_word_24 * 0x4u;
                        packed_word_15 = 0u;
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_4)) *
                                                    int64_t(int32_t(packed_word_29)) +
                                                combine_words(packed_word_18, packed_word_19));
                        split_address_words(packed_word_12, packed_word_13,
                                            combine_words(packed_word_12, packed_word_13) +
                                                packed_word_14);
                        const uint64_t byte_address_2 =
                            scaled_word_address(combine_words(uniform_word_14, uniform_word_15),
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_14 = uint32_t(byte_address_2);
                        packed_word_15 = uint32_t(byte_address_2 >> 32);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15), 0u, memory_ranges,
                            error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x4ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x8ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }

                    if (!condition_0) {
                        packed_word_12 = packed_word_8 & 0x3u;
                        packed_word_14 = packed_word_26 * 0x4u;
                        packed_word_15 = 0u;
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_12)) *
                                                    int64_t(int32_t(packed_word_3)) +
                                                0ull);
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_4)) *
                                                    int64_t(int32_t(packed_word_31)) +
                                                combine_words(packed_word_12, packed_word_13));
                        split_address_words(packed_word_12, packed_word_13,
                                            combine_words(packed_word_12, packed_word_13) +
                                                packed_word_14);
                        const uint64_t byte_address_3 =
                            scaled_word_address(combine_words(uniform_word_14, uniform_word_15),
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_14 = uint32_t(byte_address_3);
                        packed_word_15 = uint32_t(byte_address_3 >> 32);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15), 0u, memory_ranges,
                            error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x4ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x8ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }

                    packed_word_20 = packed_word_5 + packed_word_20 + packed_word_5;
                    if (!condition_1) {
                        packed_word_12 = packed_word_22 & 0x3u;
                        packed_word_14 = packed_word_28 * 0x4u;
                        packed_word_15 = 0u;
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_12)) *
                                                    int64_t(int32_t(packed_word_3)) +
                                                0ull);
                        split_address_words(packed_word_12, packed_word_13,
                                            int64_t(int32_t(packed_word_4)) *
                                                    int64_t(int32_t(packed_word_33)) +
                                                combine_words(packed_word_12, packed_word_13));
                        split_address_words(packed_word_12, packed_word_13,
                                            combine_words(packed_word_12, packed_word_13) +
                                                packed_word_14);
                        const uint64_t byte_address_4 =
                            scaled_word_address(combine_words(uniform_word_14, uniform_word_15),
                                                combine_words(packed_word_12, packed_word_13));
                        packed_word_14 = uint32_t(byte_address_4);
                        packed_word_15 = uint32_t(byte_address_4 >> 32);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15), 0u, memory_ranges,
                            error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x4ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0x8ull, 0u,
                            memory_ranges, error_counts);
                        write_boundary_word<CheckBounds>(
                            combine_words(packed_word_14, packed_word_15) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }

                    packed_word_20 = packed_word_5 + packed_word_20 + packed_word_5;
                    packed_word_8 = packed_word_21 * 0x4u + packed_word_8;
                    packed_word_22 = packed_word_21 * 0x4u + packed_word_22;
                    condition_0 = uint32_t(packed_word_20) >= uint32_t(0x40u);
                    packed_word_10 = packed_word_21 * 0x4u + packed_word_10;
                } while (!condition_0);
            }
        }
    }
    uniform_word_6 = blockIdx.z;
    {
        uint32_t loaded_word;
        loaded_word = packed_word_114 | packed_word_9 | uniform_word_6;
        condition_0 = loaded_word != 0;
    }
    if (condition_0) {
        return;
    }
    uniform_word_6 = gridDim.x;
    packed_word_2 = max(int32_t(packed_word_7), int32_t(uniform_word_4));

    uniform_word_7 = parameters[65];
    packed_word_15 = 0u - packed_word_2 + packed_word_25;
    uniform_word_14 = parameters[65];
    condition_0 = int32_t(packed_word_15) >= int32_t(0x1u);
    uniform_word_8 = parameters[62];
    uniform_word_9 = parameters[63];
    uniform_word_12 = parameters[62];
    uniform_word_13 = parameters[63];
    uniform_word_6 = (uniform_word_6 << 0x3u) + 0xfffffff8u;
    uniform_word_6 = uint32_t(int64_t(int32_t(uniform_word_6)) * (1ll << 0x1fu) >> 32) + 0x4u;
    packed_word_0 = min(int32_t(packed_word_11), int32_t(uniform_word_6));
    packed_word_0 = max(int32_t(packed_word_0), int32_t(uniform_word_5));
    if (condition_0) {
        packed_word_15 = packed_word_4 * packed_word_15;
        condition_0 = int32_t(packed_word_6) >= int32_t(packed_word_15);
        if (!condition_0) {
            packed_word_14 = packed_word_6;
            packed_word_9 = boundary_tail_steps<true>(
                packed_word_5 + packed_word_6, packed_word_15, packed_word_5);
            packed_word_8 = packed_word_9 + 0x1u;
            condition_0 = uint32_t(packed_word_9) >= uint32_t(0x3u);
            {
                uint32_t loaded_word;
                loaded_word = packed_word_8 & 0x3u;
                packed_word_10 = loaded_word;
                condition_1 = loaded_word != 0;
            }
            if (condition_1) {
                packed_word_18 = signed_word_magnitude(packed_word_11);
                condition_2 = int32_t(packed_word_11) != int32_t(0u);
                packed_word_10 = 0u - packed_word_10;
                packed_word_12 = __float_as_uint(__int2float_ru(packed_word_18));
                packed_word_17 = ~packed_word_11;
                packed_word_14 = packed_word_6;
                packed_word_12 = reciprocal_bits(packed_word_12);
                packed_word_8 = packed_word_12 + 0xffffffeu;
                packed_word_9 = __float2uint_rz(__uint_as_float(packed_word_8));
                packed_word_8 = 0u;
                packed_word_13 = 0u - packed_word_9;
                packed_word_13 = packed_word_13 * packed_word_18;
                packed_word_16 = uint32_t(uint64_t(packed_word_9) * packed_word_13 +
                                              combine_words(packed_word_8, packed_word_9) >>
                                          32);
                do {
                    packed_word_20 = uint32_t(uint32_t(packed_word_14) >> 0x2u);
                    packed_word_11 = uniform_word_7;
                    packed_word_10 = packed_word_10 + 0x1u;
                    packed_word_9 =
                        signed_word_magnitude(packed_word_20);
                    packed_word_22 =
                        signed_word_magnitude(packed_word_11);
                    packed_word_8 = uint32_t(uint64_t(packed_word_16) * packed_word_9 >> 32);
                    packed_word_12 = 0u - packed_word_8;
                    packed_word_9 = packed_word_22 * packed_word_12 + packed_word_9;
                    packed_word_12 = packed_word_14 & 0x3u;
                    packed_word_14 = packed_word_5 + packed_word_14;
                    condition_3 = uint32_t(packed_word_18) > uint32_t(packed_word_9);
                    if (!condition_3) {
                        packed_word_9 = packed_word_9 + (0u - packed_word_22);

                        packed_word_8 = packed_word_8 + 0x1u;
                    }
                    condition_1 = uint32_t(packed_word_9) >= uint32_t(packed_word_18);
                    packed_word_9 = packed_word_11 ^ packed_word_20;
                    condition_3 = int32_t(packed_word_9) >= int32_t(0u);
                    if (condition_1) {
                        packed_word_8 = packed_word_8 + 0x1u;
                    }
                    if (!condition_3) {
                        packed_word_8 = 0u - packed_word_8;
                    }
                    packed_word_13 = !condition_2 ? packed_word_17 : packed_word_8;
                    split_address_words(
                        packed_word_8, packed_word_9,
                        int64_t(int32_t(packed_word_12)) * int64_t(int32_t(packed_word_3)) + 0ull);
                    packed_word_19 = packed_word_2 + packed_word_13;
                    packed_word_12 = 0u - packed_word_13;
                    split_address_words(packed_word_8, packed_word_9,
                                        int64_t(int32_t(packed_word_4)) *
                                                int64_t(int32_t(packed_word_19)) +
                                            combine_words(packed_word_8, packed_word_9));
                    packed_word_13 = packed_word_11 * packed_word_12 + packed_word_20;
                    split_address_words(packed_word_8, packed_word_9,
                                        uint64_t(uint32_t(packed_word_13)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_word_8, packed_word_9));
                    const uint64_t byte_address_5 =
                        scaled_word_address(combine_words(uniform_word_8, uniform_word_9),
                                            combine_words(packed_word_8, packed_word_9));
                    packed_word_12 = uint32_t(byte_address_5);
                    packed_word_13 = uint32_t(byte_address_5 >> 32);
                    condition_1 = uint32_t(packed_word_10) != 0u;
                    write_boundary_word<CheckBounds>(combine_words(packed_word_12, packed_word_13),
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_12, packed_word_13) +
                                                         0x4ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_12, packed_word_13) +
                                                         0x8ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_12, packed_word_13) +
                                                         0xcull,
                                                     0u, memory_ranges, error_counts);
                } while (condition_1);
            }

            if (condition_0) {
                packed_word_35 = signed_word_magnitude(packed_word_11);
                packed_word_8 = packed_word_5 + packed_word_14;
                packed_word_10 = packed_word_14 & 0x3u;
                packed_word_16 = __float_as_uint(__int2float_ru(packed_word_35));
                packed_word_12 = packed_word_8 & 0x3u;
                split_address_words(
                    packed_word_8, packed_word_9,
                    int64_t(int32_t(packed_word_10)) * int64_t(int32_t(packed_word_3)) + 0ull);
                packed_word_10 = 0u;
                split_address_words(
                    packed_word_12, packed_word_13,
                    int64_t(int32_t(packed_word_12)) * int64_t(int32_t(packed_word_3)) + 0ull);
                packed_word_16 = reciprocal_bits(packed_word_16);
                packed_word_11 = packed_word_16 + 0xffffffeu;
                packed_word_11 = __float2uint_rz(__uint_as_float(packed_word_11));
                packed_word_18 = 0u - packed_word_11;
                packed_word_17 = packed_word_18 * packed_word_35;
                packed_word_37 = uint32_t(uint64_t(packed_word_11) * packed_word_17 +
                                              combine_words(packed_word_10, packed_word_11) >>
                                          32);
                do {
                    packed_word_11 = uniform_word_14;
                    packed_word_24 = uint32_t(uint32_t(packed_word_14) >> 0x2u);
                    packed_word_18 = packed_word_5 + packed_word_14;
                    packed_word_23 =
                        signed_word_magnitude(packed_word_11);
                    packed_word_16 =
                        signed_word_magnitude(packed_word_24);
                    packed_word_20 = __float_as_uint(__int2float_ru(packed_word_23));
                    packed_word_26 = uint32_t(uint32_t(packed_word_18) >> 0x2u);
                    packed_word_10 = uint32_t(uint64_t(packed_word_37) * packed_word_16 >> 32);
                    packed_word_28 = packed_word_5 + packed_word_18;
                    packed_word_18 =
                        signed_word_magnitude(packed_word_26);
                    packed_word_25 = packed_word_11 ^ packed_word_24;
                    packed_word_19 = 0u - packed_word_10;
                    packed_word_32 = packed_word_5 + packed_word_28;
                    packed_word_30 = uint32_t(uint32_t(packed_word_28) >> 0x2u);
                    condition_4 = int32_t(packed_word_25) >= int32_t(0u);
                    packed_word_14 = packed_word_23 * packed_word_19 + packed_word_16;
                    packed_word_20 = reciprocal_bits(packed_word_20);
                    packed_word_16 = 0u;
                    packed_word_34 = uint32_t(uint32_t(packed_word_32) >> 0x2u);
                    condition_0 = uint32_t(packed_word_35) > uint32_t(packed_word_14);
                    packed_word_19 =
                        signed_word_magnitude(packed_word_30);
                    packed_word_22 =
                        signed_word_magnitude(packed_word_34);
                    condition_3 = int32_t(packed_word_11) != int32_t(0u);
                    packed_word_31 = ~packed_word_11;
                    packed_word_28 = packed_word_28 & 0x3u;
                    if (!condition_0) {
                        packed_word_14 = packed_word_14 + (0u - packed_word_23);
                    }
                    packed_word_17 = packed_word_20 + 0xffffffeu;
                    if (!condition_0) {
                        packed_word_10 = packed_word_10 + 0x1u;
                    }
                    condition_1 = uint32_t(packed_word_14) >= uint32_t(packed_word_35);
                    packed_word_35 = packed_word_23;
                    packed_word_17 = __float2uint_rz(__uint_as_float(packed_word_17));
                    if (condition_1) {
                        packed_word_10 = packed_word_10 + 0x1u;
                    }
                    packed_word_20 = 0u - packed_word_17;
                    if (!condition_4) {
                        packed_word_10 = 0u - packed_word_10;
                    }
                    packed_word_37 = packed_word_20 * packed_word_23;
                    packed_word_37 = uint32_t(uint64_t(packed_word_17) * packed_word_37 +
                                                  combine_words(packed_word_16, packed_word_17) >>
                                              32);
                    packed_word_16 = packed_word_18;
                    packed_word_18 = uint32_t(uint64_t(packed_word_37) * packed_word_16 >> 32);
                    packed_word_20 = uint32_t(uint64_t(packed_word_37) * packed_word_19 >> 32);
                    packed_word_17 = 0u - packed_word_18;
                    packed_word_21 = uint32_t(uint64_t(packed_word_37) * packed_word_22 >> 32);
                    packed_word_14 = packed_word_23 * packed_word_17 + packed_word_16;
                    packed_word_16 = 0u - packed_word_20;
                    packed_word_17 = 0u - packed_word_21;
                    condition_6 = uint32_t(packed_word_35) > uint32_t(packed_word_14);
                    packed_word_16 = packed_word_23 * packed_word_16 + packed_word_19;
                    packed_word_19 = packed_word_23 * packed_word_17 + packed_word_22;
                    packed_word_17 = !condition_3 ? packed_word_31 : packed_word_10;
                    condition_0 = uint32_t(packed_word_35) > uint32_t(packed_word_16);
                    condition_1 = uint32_t(packed_word_35) > uint32_t(packed_word_19);
                    packed_word_10 = 0u - packed_word_17;
                    if (!condition_6) {
                        packed_word_14 = packed_word_14 + (0u - packed_word_23);

                        packed_word_18 = packed_word_18 + 0x1u;
                    }
                    condition_2 = uint32_t(packed_word_14) >= uint32_t(packed_word_35);
                    packed_word_14 = packed_word_11 ^ packed_word_26;
                    if (!condition_1) {
                        packed_word_19 = packed_word_19 + (0u - packed_word_23);
                    }
                    if (!condition_0) {
                        packed_word_16 = packed_word_16 + (0u - packed_word_23);
                    }
                    if (!condition_1) {
                        packed_word_21 = packed_word_21 + 0x1u;
                    }
                    condition_4 = int32_t(packed_word_14) >= int32_t(0u);
                    packed_word_14 = packed_word_11 ^ packed_word_30;
                    condition_6 = uint32_t(packed_word_19) >= uint32_t(packed_word_35);
                    packed_word_19 = packed_word_2 + packed_word_17;
                    condition_5 = uint32_t(packed_word_16) >= uint32_t(packed_word_35);
                    if (!condition_0) {
                        packed_word_20 = packed_word_20 + 0x1u;
                    }
                    if (condition_2) {
                        packed_word_18 = packed_word_18 + 0x1u;
                    }
                    condition_2 = int32_t(packed_word_14) >= int32_t(0u);
                    packed_word_14 = packed_word_11 ^ packed_word_34;
                    split_address_words(packed_word_16, packed_word_17,
                                        int64_t(int32_t(packed_word_4)) *
                                                int64_t(int32_t(packed_word_19)) +
                                            combine_words(packed_word_8, packed_word_9));
                    packed_word_19 = packed_word_11 * packed_word_10 + packed_word_24;
                    packed_word_10 = packed_word_18;
                    condition_0 = int32_t(packed_word_14) >= int32_t(0u);
                    if (condition_6) {
                        packed_word_21 = packed_word_21 + 0x1u;
                    }
                    split_address_words(packed_word_16, packed_word_17,
                                        uint64_t(uint32_t(packed_word_19)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_word_16, packed_word_17));
                    if (condition_5) {
                        packed_word_20 = packed_word_20 + 0x1u;
                    }
                    if (!condition_4) {
                        packed_word_10 = 0u - packed_word_10;
                    }
                    packed_word_24 = packed_word_32 & 0x3u;
                    packed_word_14 = packed_word_21;
                    const uint64_t byte_address_6 =
                        scaled_word_address(combine_words(uniform_word_12, uniform_word_13),
                                            combine_words(packed_word_16, packed_word_17));
                    packed_word_18 = uint32_t(byte_address_6);
                    packed_word_21 = !condition_3 ? packed_word_31 : packed_word_10;
                    if (!condition_2) {
                        packed_word_20 = 0u - packed_word_20;
                    }
                    packed_word_19 = uint32_t(byte_address_6 >> 32);
                    split_address_words(
                        packed_word_24, packed_word_25,
                        int64_t(int32_t(packed_word_24)) * int64_t(int32_t(packed_word_3)) + 0ull);
                    if (!condition_0) {
                        packed_word_14 = 0u - packed_word_14;
                    }
                    packed_word_17 = packed_word_2 + packed_word_21;
                    packed_word_10 = 0u - packed_word_21;
                    packed_word_27 = !condition_3 ? packed_word_31 : packed_word_20;
                    split_address_words(
                        packed_word_20, packed_word_21,
                        int64_t(int32_t(packed_word_28)) * int64_t(int32_t(packed_word_3)) + 0ull);
                    packed_word_31 = !condition_3 ? packed_word_31 : packed_word_14;
                    write_boundary_word<CheckBounds>(combine_words(packed_word_18, packed_word_19),
                                                     0u, memory_ranges, error_counts);
                    split_address_words(packed_word_16, packed_word_17,
                                        int64_t(int32_t(packed_word_4)) *
                                                int64_t(int32_t(packed_word_17)) +
                                            combine_words(packed_word_12, packed_word_13));
                    packed_word_29 = packed_word_2 + packed_word_27;
                    packed_word_33 = packed_word_2 + packed_word_31;
                    packed_word_14 = 0u - packed_word_31;
                    packed_word_23 = packed_word_11 * packed_word_10 + packed_word_26;
                    packed_word_10 = 0u - packed_word_27;
                    write_boundary_word<CheckBounds>(combine_words(packed_word_18, packed_word_19) +
                                                         0x4ull,
                                                     0u, memory_ranges, error_counts);
                    split_address_words(packed_word_24, packed_word_25,
                                        int64_t(int32_t(packed_word_4)) *
                                                int64_t(int32_t(packed_word_33)) +
                                            combine_words(packed_word_24, packed_word_25));
                    write_boundary_word<CheckBounds>(combine_words(packed_word_18, packed_word_19) +
                                                         0x8ull,
                                                     0u, memory_ranges, error_counts);
                    split_address_words(packed_word_16, packed_word_17,
                                        uint64_t(uint32_t(packed_word_23)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_word_16, packed_word_17));
                    write_boundary_word<CheckBounds>(combine_words(packed_word_18, packed_word_19) +
                                                         0xcull,
                                                     0u, memory_ranges, error_counts);
                    split_address_words(packed_word_22, packed_word_23,
                                        int64_t(int32_t(packed_word_4)) *
                                                int64_t(int32_t(packed_word_29)) +
                                            combine_words(packed_word_20, packed_word_21));
                    const uint64_t byte_address_7 =
                        scaled_word_address(combine_words(uniform_word_12, uniform_word_13),
                                            combine_words(packed_word_16, packed_word_17));
                    packed_word_20 = uint32_t(byte_address_7);
                    packed_word_21 = packed_word_11 * packed_word_10 + packed_word_30;
                    packed_word_27 = packed_word_11 * packed_word_14 + packed_word_34;
                    packed_word_14 = packed_word_5 + packed_word_32;
                    split_address_words(packed_word_22, packed_word_23,
                                        uint64_t(uint32_t(packed_word_21)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_word_22, packed_word_23));
                    packed_word_21 = uint32_t(byte_address_7 >> 32);
                    split_address_words(packed_word_24, packed_word_25,
                                        uint64_t(uint32_t(packed_word_27)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_word_24, packed_word_25));
                    const uint64_t byte_address_8 =
                        scaled_word_address(combine_words(uniform_word_12, uniform_word_13),
                                            combine_words(packed_word_22, packed_word_23));
                    packed_word_16 = uint32_t(byte_address_8);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_20, packed_word_21),
                                                     0u, memory_ranges, error_counts);
                    const uint64_t byte_address_9 =
                        scaled_word_address(combine_words(uniform_word_12, uniform_word_13),
                                            combine_words(packed_word_24, packed_word_25));
                    packed_word_26 = uint32_t(byte_address_9);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_20, packed_word_21) +
                                                         0x4ull,
                                                     0u, memory_ranges, error_counts);
                    packed_word_17 = uint32_t(byte_address_8 >> 32);
                    packed_word_27 = uint32_t(byte_address_9 >> 32);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_20, packed_word_21) +
                                                         0x8ull,
                                                     0u, memory_ranges, error_counts);
                    condition_0 = int32_t(packed_word_14) >= int32_t(packed_word_15);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_20, packed_word_21) +
                                                         0xcull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_16, packed_word_17),
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_16, packed_word_17) +
                                                         0x4ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_16, packed_word_17) +
                                                         0x8ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_16, packed_word_17) +
                                                         0xcull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_26, packed_word_27),
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_26, packed_word_27) +
                                                         0x4ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_26, packed_word_27) +
                                                         0x8ull,
                                                     0u, memory_ranges, error_counts);
                    write_boundary_word<CheckBounds>(combine_words(packed_word_26, packed_word_27) +
                                                         0xcull,
                                                     0u, memory_ranges, error_counts);
                } while (!condition_0);
            }
        }
    }

    uniform_word_4 = parameters[64];
    packed_word_2 = 0u - packed_word_0 + packed_word_11;
    packed_word_8 = min(int32_t(packed_word_2), int32_t(uniform_word_4));
    condition_0 = int32_t(packed_word_8) >= int32_t(0x1u);
    if (!condition_0) {
        return;
    }
    packed_word_7 = packed_word_7 * packed_word_2;
    packed_word_23 = packed_word_7 * 0x4u;
    condition_0 = int32_t(packed_word_6) >= int32_t(packed_word_23);
    if (condition_0) {
        return;
    }
    // Clear the final column-padding region only. Earlier tile/edge publication
    // and its synchronization stay here; see docs/BOUNDARY_KERNELS.md.
    const BoundaryVectorLayout padding_layout{
        combine_words(parameters[62], parameters[63]),
        packed_word_3, // component stride, uint32 words
        packed_word_4, // row stride, uint32 words
        packed_word_0}; // column origin, signed-word bits
    const uint32_t padding_width = packed_word_2;
    const uint32_t first_padding_item = packed_word_6;
    const uint32_t padding_extent = packed_word_23;
    const uint32_t block_thread_count = packed_word_5;
    clear_boundary_rows<2>(padding_layout, padding_width, first_padding_item,
                           padding_extent, block_thread_count, [&](uint64_t address) {
        write_boundary_word<CheckBounds>(address, 0u, memory_ranges, error_counts);
    });
}

extern "C" __global__
__launch_bounds__(32) void phoenix_input_preparation(KernelArguments arguments) {
    run_input_preparation<false>(arguments.words, nullptr, nullptr);
}
