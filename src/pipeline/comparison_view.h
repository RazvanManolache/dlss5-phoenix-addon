#pragma once
#include "options.h"
namespace phoenix_pipeline {
// Fits in an existing unused root-constant word. Low four bits remain owned
// by the host's transfer/reset flags, so view changes never touch recurrence.
inline unsigned comparison_view_bits(const Options& p){
    if(p.compare_zoom==1)return 0;
    return (unsigned(std::lround((p.compare_zoom-1)*16))<<4)|
        (unsigned(std::lround(p.loupe_x*127))<<11)|(unsigned(std::lround(p.loupe_y*127))<<18)|(unsigned(p.loupe)<<25);
}
inline std::array<unsigned,2> comparison_point(unsigned x,unsigned y,unsigned width,unsigned height,unsigned flags){
    const float zoom=1+float((flags>>4)&127)/16;
    if(zoom<=1)return {x,y};
    const float cx=float((flags>>11)&127)/127*width,cy=float((flags>>18)&127)/127*height;
    const float dx=float(x)+.5f-cx,dy=float(y)+.5f-cy;
    if((flags&(1u<<25))&&std::sqrt(dx*dx+dy*dy)>float(std::min(width,height))*.18f)return {x,y};
    return {unsigned(std::clamp(std::floor(dx/zoom+cx),0.f,float(width-1))),
        unsigned(std::clamp(std::floor(dy/zoom+cy),0.f,float(height-1)))};
}
}
#define PHOENIX_COMPARISON_VIEW_HLSL \
"uint2 ComparisonPoint(uint2 pixel,uint width,uint height,uint flags){\n" \
" float zoom=1+float((flags>>4)&127)/16;\n" \
" float2 mapped=float2(pixel);\n" \
" float2 center=float2((flags>>11)&127,(flags>>18)&127)/127*float2(width,height);\n" \
" bool magnify=zoom>1&&((flags&(1u<<25))==0||distance(float2(pixel)+.5,center)<=float(min(width,height))*.18);\n" \
" if(magnify){float2 p=(float2(pixel)+.5-center)/zoom+center-.5;mapped=clamp(floor(p+.5),0,float2(width-1,height-1));}\n" \
" return uint2(mapped);\n}\n"
