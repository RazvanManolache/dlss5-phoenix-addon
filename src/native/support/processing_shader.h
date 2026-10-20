#pragma once
#include "processing.h"
namespace phoenix_addon::phoenix::direct {
struct ProcessingConstants {
    unsigned width, height, flags, left, top, image_width, image_height;
    float white, exposure;
};
inline ProcessingConstants processing_constants(const Processing& p, const Resolution& g,
                                                unsigned operation, bool srgb,
                                                float host_white = 100, float host_exposure = 1) {
    return {operation == 3 ? g.output_width : g.width,
            operation == 3 ? g.output_height : g.height,
            unsigned(p.input_encoding(srgb)) | unsigned(p.signal << 3) | unsigned(p.domain << 5) |
                unsigned(p.downsample << 7) | unsigned(p.upsample << 9) |
                unsigned(p.reconstruction << 11) | (operation << 12),
            g.left,
            g.top,
            g.image_width,
            g.image_height,
            p.diffuse_white(host_white),
            p.effective_exposure(host_exposure)};
}
// All filtering is performed after decoding the source transfer. Alpha is
// independent of NR and is restored from the full-resolution source.
inline constexpr const char* processing_shader = R"hlsl(
Texture2D<float4> Source : register(t0);
Texture2D<float4> Original : register(t1);
Texture2D<float4> Baseline : register(t2);
RWTexture2D<float4> Destination : register(u0);
cbuffer Params : register(b0) {
 uint width, height, flags, left, top, image_width, image_height;
 float white, exposure;
};
float3 srgb_decode(float3 v) {
 float3 a=abs(v); return sign(v)*lerp(a/12.92,pow((a+.055)/1.055,2.4),step(.04045,a));
}
float3 srgb_encode(float3 v) {
 float3 a=abs(v); return sign(v)*lerp(a*12.92,1.055*pow(a,1.0/2.4)-.055,step(.0031308,a));
}
float3 pq_encode(float3 v) {
 float3 a=pow(min(abs(v),1),.1593017578125);
 return sign(v)*pow((.8359375+18.8515625*a)/(1+18.6875*a),78.84375);
}
float3 pq_decode(float3 v) {
 float3 a=pow(min(abs(v),1),1.0/78.84375);
 return sign(v)*pow(max(a-.8359375,0)/max(18.8515625-18.6875*a,1e-6),1.0/.1593017578125);
}
float3 decode(float3 v) {
 uint encoding=flags&7;
 float3 result=v;
 if(encoding==2)result=srgb_decode(v);
 else if(encoding==3) {
  float3 decoded_pq=pq_decode(v)*(10000/white);
  result=mul(float3x3(1.660491,-.587641,-.072850,-.124550,1.132900,-.008349,-.018151,-.100579,1.118730),decoded_pq);
 }
 else if(encoding==4)result=v*(80/white);
 else if(encoding==5)result=srgb_decode(v)*(80/white);
 return result;
}
float3 encode(float3 v) {
 uint encoding=flags&7;
 float3 result=v;
 if(encoding==2)result=srgb_encode(v);
 else if(encoding==3) {
  float3 encoded_pq=mul(float3x3(.627404,.329283,.043313,.069097,.919540,.011362,.016391,.088013,.895595),v);
  result=pq_encode(encoded_pq*(white/10000));
 }
 else if(encoding==4)result=v*(white/80);
 else if(encoding==5)result=srgb_encode(v*(white/80));
 return result;
}
float3 to_domain(float3 v) {
 uint domain=(flags>>5)&3;
 return domain==0?srgb_encode(v*(white/80)):domain==1?pq_encode(v*(white/10000)):v;
}
float3 from_domain(float3 v) {
 uint domain=(flags>>5)&3;
 return domain==0?srgb_decode(v)*(80/white):domain==1?pq_decode(v)*(10000/white):v;
}
float3 nr_encode(float3 v) {
 uint signal=(flags>>3)&3;
 return signal==0?srgb_encode(v):signal==1?pq_encode(v*(white/10000)):v;
}
float3 nr_decode(float3 v) {
 uint signal=(flags>>3)&3;
 return signal==0?srgb_decode(v):signal==1?pq_decode(v)*(10000/white):v;
}
float4 tap(int2 p,int2 lo,int2 hi,bool residual) {
 p=clamp(p,lo,hi);
 float4 v=Source.Load(int3(p,0));
 float3 decoded=decode(v.rgb);
 if(residual)decoded-=decode(Baseline.Load(int3(p,0)).rgb);
 return float4(residual?decoded:to_domain(decoded),v.a);
}
float cubic(float x) {
 x=abs(x);return x<1?1.5*x*x*x-2.5*x*x+1:x<2?-.5*x*x*x+2.5*x*x-4*x+2:0;
}
float4 sample_image(float2 position,float2 footprint,int2 lo,int2 hi,uint filter,bool residual,bool down) {
 float4 sum=0;float total=0;
 if(filter==2) {sum=tap(int2(floor(position+.5)),lo,hi,residual);total=1;}
 else if(down&&filter==0&&any(footprint>1)) {
  float2 start=position+.5-footprint*.5, finish=position+.5+footprint*.5;
  int2 first=int2(floor(start)),last=int2(ceil(finish));
  [loop]for(int y=first.y;y<last.y;++y)[loop]for(int x=first.x;x<last.x;++x) {
   float2 coverage=max(0,min(float2(x+1,y+1),finish)-max(float2(x,y),start));
   float weight=coverage.x*coverage.y;
   sum+=tap(int2(x,y),lo,hi,residual)*weight;total+=weight;
  }
 } else if(!down&&filter==1) {
  int2 base=int2(floor(position));
  [unroll]for(int y=-1;y<=2;++y)[unroll]for(int x=-1;x<=2;++x) {
   float weight=cubic(position.x-base.x-x)*cubic(position.y-base.y-y);
   sum+=tap(base+int2(x,y),lo,hi,residual)*weight;total+=weight;
  }
 } else {
  int2 base=int2(floor(position));float2 f=frac(position);
  sum=lerp(lerp(tap(base,lo,hi,residual),tap(base+int2(1,0),lo,hi,residual),f.x),
              lerp(tap(base+int2(0,1),lo,hi,residual),tap(base+1,lo,hi,residual),f.x),f.y);
  total=1;
 }
 return sum/max(total,1e-8);
}
[numthreads(16,16,1)]void main(uint3 tid:SV_DispatchThreadID) {
 uint2 p=tid.xy;if(p.x>=width||p.y>=height)return;
 uint operation=(flags>>12)&15;
 if(operation==0) { // Host transfer -> explicit NR signal.
  Destination[p]=float4(nr_encode(decode(Source.Load(int3(p,0)).rgb)*exposure),1);return;
 }
 if(operation==1) { // NR signal -> host transfer. Preserve input at zero exposure.
  float4 original=Original.Load(int3(p,0));
  Destination[p]=float4(exposure>0?encode(nr_decode(Source.Load(int3(p,0)).rgb)/exposure):original.rgb,original.a);return;
 }
 uint sw,sh;Source.GetDimensions(sw,sh);
 if(operation==2) {
  if(p.x<left||p.y<top||p.x>=left+image_width||p.y>=top+image_height) {Destination[p]=float4(0,0,0,1);return;}
  float2 footprint=float2(sw,sh)/float2(image_width,image_height);
  float2 position=(float2(p)-float2(left,top)+.5)*footprint-.5;
  float4 v=sample_image(position,footprint,int2(0,0),int2(sw-1,sh-1),(flags>>7)&3,false,true);
  Destination[p]=float4(encode(from_domain(v.rgb)),v.a);return;
 }
 // Upsample either the processed image or its linear RGB residual. Residual
 // mode preserves original high-frequency detail when NR runs below 100%.
 bool residual=((flags>>11)&1)==0;
 float2 position=float2(left,top)+(float2(p)+.5)*float2(image_width,image_height)/float2(width,height)-.5;
 float4 v=sample_image(position,1,int2(left,top),int2(left+image_width-1,top+image_height-1),(flags>>9)&3,residual,false);
 float4 original=Original.Load(int3(p,0));
 float3 result=residual?decode(original.rgb)+v.rgb:from_domain(v.rgb);
 Destination[p]=float4(encode(result),original.a);
}
)hlsl";
} // namespace phoenix_addon::phoenix::direct
