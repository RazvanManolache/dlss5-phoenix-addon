#version 450
// 4x4 register tiles for subgroup-independent mobile matrix multiplication.
// Vector lanes are separate output columns, so each scalar keeps ascending-K FMA order.
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_16bit_storage : require
#include "publish_workgroup.glsl"
layout(local_size_x = 64) in;
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer HalfBuf { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant, scalar) uniform Push {
    HalfBuf a, b;
    FloatBuf c, d;
    uint m, n, k, batch, sa, sb, sc, flags;
    float p0, p1, p2, p3;
    uint lda, ldb, ldc, spare;
    uint64_t residual_cos;
    uint image_h, image_w, window_cols, window_pad;
    uint64_t qkv_scale;
} pc;
#include "specialize.glsl"
#include "packed_weight_workgroup.glsl"
#ifndef MOBILE_REDUCTION
#define MOBILE_REDUCTION 32
#endif
shared float activation_tile[32 * MOBILE_REDUCTION];
shared float weight_tile[MOBILE_REDUCTION * 32];
void store_row(uint row, uint column, uint output_batch, vec4 values) {
    if(row>=pc.m)return;
    uint flags=operation_flags();
    uint stride=pc.ldc==0u?pc.n:pc.ldc;
    for(uint i=0u;i<4u;++i) {
        uint col=column+i;
        if(col>=pc.n||((flags&0x10000u)!=0u&&col>=4u))continue;
        uint target=output_batch+row*stride+col;
        float value=publish((flags>>8u)&15u,values[i]);
        if((flags&0x1000u)!=0u)HalfBuf(uint64_t(pc.c)).v[target]=uint16_t(phoenix_half_bits(value));
        else pc.c.v[target]=value;
    }
}
void main() {
    uint flags=operation_flags(),lane=gl_LocalInvocationID.x;
    bool transposed=(flags&1u)!=0u;
    uint left_stride=pc.lda==0u?pc.k:pc.lda;
    uint right_stride=pc.ldb==0u?(transposed?pc.k:pc.n):pc.ldb;
    uint row_tiles=(pc.m+31u)/32u;
    uint tile=gl_WorkGroupID.y+gl_WorkGroupID.z*gl_NumWorkGroups.y;
    uint batch=tile/row_tiles;
    if(batch>=max(pc.batch,1u))return;
    uint first_row=(tile%row_tiles)*32u, first_col=gl_WorkGroupID.x*32u;
    uint ao=batch*pc.sa,bo=batch*pc.sb,co=batch*pc.sc;
    uint row=(lane/8u)*4u,col=(lane%8u)*4u;
    precise vec4 sum0=vec4(0.0),sum1=vec4(0.0),sum2=vec4(0.0),sum3=vec4(0.0);
    for(uint kb=0u;kb<pc.k;kb+=MOBILE_REDUCTION) {
        for(uint e=lane;e<32u*MOBILE_REDUCTION;e+=64u) {
            uint r=first_row+e/MOBILE_REDUCTION,k=kb+e%MOBILE_REDUCTION;
            activation_tile[e]=r<pc.m&&k<pc.k?phoenix_half_value(uint(pc.a.v[ao+r*left_stride+k])):0.0;
        }
        for(uint e=lane;e<MOBILE_REDUCTION*32u;e+=64u) {
            uint k=transposed?e%MOBILE_REDUCTION:e/32u;
            uint c=transposed?e/MOBILE_REDUCTION:e%32u;
            uint reduction=kb+k,column=first_col+c;
            weight_tile[k*32u+c]=reduction<pc.k&&column<pc.n?
                matrix_b(bo+(transposed?column*right_stride+reduction:reduction*right_stride+column)):0.0;
        }
        barrier();
        for(uint k=0u;k<min(uint(MOBILE_REDUCTION),pc.k-kb);++k) {
            uint at=k*32u+col;
            vec4 weights=vec4(weight_tile[at],weight_tile[at+1u],weight_tile[at+2u],weight_tile[at+3u]);
            sum0=fma(vec4(activation_tile[row*MOBILE_REDUCTION+k]),weights,sum0);
            sum1=fma(vec4(activation_tile[(row+1u)*MOBILE_REDUCTION+k]),weights,sum1);
            sum2=fma(vec4(activation_tile[(row+2u)*MOBILE_REDUCTION+k]),weights,sum2);
            sum3=fma(vec4(activation_tile[(row+3u)*MOBILE_REDUCTION+k]),weights,sum3);
        }
        barrier();
    }
    store_row(first_row+row,first_col+col,co,sum0);
    store_row(first_row+row+1u,first_col+col,co,sum1);
    store_row(first_row+row+2u,first_col+col,co,sum2);
    store_row(first_row+row+3u,first_col+col,co,sum3);
}
