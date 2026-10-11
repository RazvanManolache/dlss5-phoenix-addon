#pragma once
namespace phoenix_addon::phoenix {
// Backend-neutral D3D12 presentation kernels. Inference can run elsewhere.
// Guide texture: current->previous MV in output pixels, positive linear depth,
// validity. Map: source pixel-centre XY, validity, previous-frame depth.
// Mode 0 initializes a captured frame's map, 1 advances it, 2 composes residual.
inline constexpr const char* async_shader = R"(
Texture2D<float4> first:register(t0);
Texture2D<float4> second:register(t1);
Texture2D<float4> third:register(t2);
RWTexture2D<float4> destination:register(u0);
cbuffer Params:register(b0){uint width,height,reset,mode;uint a,b;float divider,gain;uint resolve;};
float4 bilinear(Texture2D<float4> tex,float2 p) {
    float2 q=p-.5;int2 base=int2(floor(q));float2 f=frac(q);
    int2 hi=int2(width-1,height-1);
    float4 v00=tex.Load(int3(clamp(base,int2(0,0),hi),0));
    float4 v10=tex.Load(int3(clamp(base+int2(1,0),int2(0,0),hi),0));
    float4 v01=tex.Load(int3(clamp(base+int2(0,1),int2(0,0),hi),0));
    float4 v11=tex.Load(int3(clamp(base+int2(1,1),int2(0,0),hi),0));
    return lerp(lerp(v00,v10,f.x),lerp(v01,v11,f.x),f.y);
}
bool inside(float2 p){return all(p>=.5)&&all(p<=float2(width,height)-.5);}
bool depth_valid(float z){return (resolve&1)?(z>=0&&z<=1):z>0;}
[numthreads(16,16,1)]void main(uint3 tid:SV_DispatchThreadID){
    if(tid.x>=width||tid.y>=height)return;
    uint2 p=tid.xy;float2 here=float2(p)+.5;
    if(mode==0){
        float4 g=second.Load(int3(p,0));
        destination[p]=float4(here,(all(isfinite(g))&&depth_valid(g.z)&&g.w>0)?1:0,g.z);
    }else if(mode==1){
        float4 g=second.Load(int3(p,0));
        if(!all(isfinite(g))||!depth_valid(g.z)||g.w<=0){destination[p]=float4(here,0,0);return;}
        float2 previous=here+g.xy;
        // Binary validity must not be bilinearly averaged and thresholded near
        // one: that dilates every hole once per frame until nothing survives.
        // Require the nearest correspondence, then interpolate only compatible
        // valid taps. Invalid center pixels stay invalid (no hole filling).
        int2 hi=int2(width-1,height-1);
        float4 center=first.Load(int3(clamp(int2(floor(previous)),int2(0,0),hi),0));
        float tolerance=max(gain,0)*((resolve&1)?1:max(center.w,g.z));
        bool valid=inside(previous)&&all(isfinite(center))&&center.z>.9999&&depth_valid(center.w)&&
            abs(center.w-g.z)<=tolerance&&inside(center.xy);
        float2 q=previous-.5;int2 base=int2(floor(q));float2 f=frac(q);
        float2 coordinate=0;float weight=0;
        [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x){
            float4 tap=first.Load(int3(clamp(base+int2(x,y),int2(0,0),hi),0));
            float w=(x?f.x:1-f.x)*(y?f.y:1-f.y);
            bool accept=all(isfinite(tap))&&tap.z>.9999&&depth_valid(tap.w)&&inside(tap.xy)&&
                abs(tap.w-g.z)<=max(gain,0)*((resolve&1)?1:max(tap.w,g.z));
            if(accept){coordinate+=tap.xy*w;weight+=w;}
        }
        valid=valid&&weight>0;
        destination[p]=float4(valid?coordinate/max(weight,1e-8):here,valid?1:0,g.z);
    }else{
        float4 color=first.Load(int3(p,0));float4 map=third.Load(int3(p,0));
        // Host supplies generation/age eligibility in reset. It is an explicit
        // keep-input switch, not an inference-success flag.
        if(reset!=0||!all(isfinite(map))||map.z<.9999||!inside(map.xy)){destination[p]=color;return;}
        float3 residual=bilinear(second,map.xy).rgb;
        if(!all(isfinite(residual))){destination[p]=color;return;}
        destination[p]=float4(saturate(color.rgb+residual),color.a);
    }
})";
}
