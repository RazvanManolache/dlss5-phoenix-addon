// Adapted from DLSSNR-AMD windows/shaders/rdna4/include/coopmm.glsl.
// Copyright (c) 2026 mochizuki0323, MIT; see third_party/amd_nr/LICENSE.
// Phoenix owns the surrounding graph, half-buffer ABI and publication arithmetic.
// The selected device supplies K; no vendor-specific fragment lane mapping is used.
#extension GL_EXT_float_e4m3 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#define PHX_FRAG_A8 coopmat<floate4m3_t, gl_ScopeSubgroup, 16, FP8_K, gl_MatrixUseA>
#define PHX_FRAG_B8 coopmat<floate4m3_t, gl_ScopeSubgroup, FP8_K, 16, gl_MatrixUseB>
#define PHX_LOAD_A8(frag, arr) coopMatLoad(frag, arr, 0, FP8_K, gl_CooperativeMatrixLayoutRowMajor)
// B is staged [N][K], matching the AMD implementation's column-major load.
#define PHX_LOAD_B8(frag, arr)                                                                     \
    coopMatLoad(frag, arr, 0, FP8_K, gl_CooperativeMatrixLayoutColumnMajor)
shared floate4m3_t a_fp8[16 * FP8_K];
shared floate4m3_t b_fp8[16 * FP8_K];
shared uint fp8_inexact;
