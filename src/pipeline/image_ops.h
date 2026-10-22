#pragma once
#include "../native/support/compute_pass.h"

namespace phoenix_pipeline {
// SDR proxy-domain stages. Motion uses nearest sampling and scales pixel-space
// displacement with geometry. Image reduction averages the source footprint.
inline constexpr char image_ops_shader[]=R"hlsl(
Texture2D<float4> Source:register(t0);Texture2D<float4> Other:register(t1);
RWTexture2D<float4> Output:register(u0);
cbuffer Config:register(b0){uint W,H,Mode,Aux;float A,B,C,D;uint Unused;};
float4 sampleImage(Texture2D<float4> image,float2 p,uint2 size){
 int2 b=int2(floor(p)),hi=int2(size)-1;float2 f=frac(p);
 return lerp(lerp(image.Load(int3(clamp(b,0,hi),0)),image.Load(int3(clamp(b+int2(1,0),0,hi),0)),f.x),
 lerp(image.Load(int3(clamp(b+int2(0,1),0,hi),0)),image.Load(int3(clamp(b+1,0,hi),0)),f.x),f.y);
}
[numthreads(16,16,1)]void main(uint3 t:SV_DispatchThreadID){
 if(t.x>=W||t.y>=H)return;uint sw,sh;Source.GetDimensions(sw,sh);
 if((Mode==0||Mode==1)&&sw==W&&sh==H){Output[t.xy]=Source.Load(int3(t.xy,0));return;}
 float2 size=float2(sw,sh),ratio=size/float2(W,H),p=(float2(t.xy)+.5)*ratio-.5;
 float4 v;
 if(Mode==1){int2 q=clamp(int2(floor(p+.5)),0,int2(sw,sh)-1);v=Source.Load(int3(q,0));v.xy*=float2(W,H)/size;}
 else if(Mode==2){v=Source.Load(int3(t.xy,0));float l=dot(v.rgb,float3(.2126,.7152,.0722));
   v.rgb=lerp(l.xxx,v.rgb,C);v.rgb=saturate((v.rgb-.5)*B+.5+A);v.rgb=pow(v.rgb,1/max(D,.01));}
 else if(Mode==3){v=lerp(Source.Load(int3(t.xy,0)),Other.Load(int3(t.xy,0)),A);}
 else if(Mode==5){v=Source.Load(int3(t.xy,0));uint ow,oh;Other.GetDimensions(ow,oh);
   v.a=sampleImage(Other,(float2(t.xy)+.5)*float2(ow,oh)/float2(W,H)-.5,uint2(ow,oh)).a;}
 else if(Mode==4){v=Source.Load(int3(t.xy,0));float4 original=Other.Load(int3(t.xy,0));
   float3 delta=v.rgb-original.rgb;float l=dot(delta,float3(.2126,.7152,.0722));
   v.rgb=saturate(original.rgb+l.xxx+A*(delta-l.xxx));v.a=original.a;}
 else{float2 o=max(ratio-1,0)*.25;
   v=.25*(sampleImage(Source,p+o,uint2(sw,sh))+sampleImage(Source,p-o,uint2(sw,sh))+
     sampleImage(Source,p+float2(o.x,-o.y),uint2(sw,sh))+sampleImage(Source,p+float2(-o.x,o.y),uint2(sw,sh)));}
 Output[t.xy]=v;
})hlsl";
struct ImageConstants {unsigned width=0,height=0,mode=0,aux=0;float a=0,b=1,c=1,d=1;unsigned unused=0;};
static_assert(sizeof(ImageConstants)==36);
} // namespace phoenix_pipeline
