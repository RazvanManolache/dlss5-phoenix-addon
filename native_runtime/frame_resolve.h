#pragma once
#include "half_rgba.h"
#include "frame_contract.h"

namespace phoenix_cpp {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
// Keep the baseline DLL usable without AVX/F16C. Dispatch only after checking
// CPU and OS support; explicit multiplies/adds retain the scalar rounding order.
__declspec(noinline) inline void resolve_neural_row(
    const float* color, const float* head, const float* prior, const uint8_t* valid,
    float gain, float* next, unsigned width) {
    const auto zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1.f),quarter=_mm256_set1_ps(.25f);
    unsigned p=0;
    for(;p+1<width;p+=2){
        const float a=resolve_gate(head[p*4+3],valid[p]!=0,gain);
        const float b=resolve_gate(head[p*4+7],valid[p+1]!=0,gain);
        const auto current=_mm256_setr_ps(color[p*3],color[p*3+1],color[p*3+2],0.f,
            color[p*3+3],color[p*3+4],color[p*3+5],0.f);
        const auto previous=_mm256_setr_ps(prior[p*3],prior[p*3+1],prior[p*3+2],0.f,
            prior[p*3+3],prior[p*3+4],prior[p*3+5],0.f);
        // Clamp comparisons select the original operand for unordered values,
        // matching std::clamp rather than silently consuming a nonfinite head.
        // The fourth channel is a logit, not an RGB correction. Mask it before
        // arithmetic so an unused signaling NaN cannot raise an FP exception.
        const auto correction=_mm256_blend_ps(_mm256_loadu_ps(head+p*4),zero,0x88);
        auto network=_mm256_add_ps(current,_mm256_mul_ps(correction,quarter));
        network=_mm256_blendv_ps(network,zero,_mm256_cmp_ps(network,zero,_CMP_LT_OQ));
        network=_mm256_blendv_ps(network,one,_mm256_cmp_ps(network,one,_CMP_GT_OQ));
        network=_mm256_add_ps(network,_mm256_mul_ps(_mm256_sub_ps(previous,network),
            _mm256_setr_ps(a,a,a,0.f,b,b,b,0.f)));
        alignas(32) float packed[8];
        _mm256_store_ps(packed,_mm256_cvtph_ps(_mm256_cvtps_ph(network,0)));
        std::copy_n(packed,3,next+p*3);std::copy_n(packed+4,3,next+p*3+3);
    }
    if(p<width){
        const float gate=resolve_gate(head[p*4+3],valid[p]!=0,gain);
        for(unsigned c=0;c<3;++c){
            float value=std::clamp(color[p*3+c]+head[p*4+c]*.25f,0.f,1.f);
            value+=(prior[p*3+c]-value)*gate;
            const auto half=_mm_cvtps_ph(_mm_set_ss(value),0);
            next[p*3+c]=_mm_cvtss_f32(_mm_cvtph_ps(half));
        }
    }
}
#endif
}
