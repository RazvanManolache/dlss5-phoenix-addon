#pragma once
namespace phoenix_pipeline {
// Native implementation of the Apache-2.0 Phase4Generator deploy graph.
// Conv padding is zero; the global bicubic bypass uses PyTorch's a=-0.75,
// align_corners=false convention. Learned channels carry no assigned semantics.
inline constexpr const char* tiny_shader = R"hlsl(
StructuredBuffer<float> Input : register(t0);
StructuredBuffer<float> Weights : register(t1);
StructuredBuffer<float> Head : register(t2);
Texture2D<float4> Original : register(t3);
RWStructuredBuffer<float> Output : register(u0);
RWTexture2D<float4> Image : register(u1);
cbuffer Parameters : register(b0) {
    uint Width, Height, InChannels, OutChannels;
    uint WeightOffset, BiasOffset, ActivationOffset, Mode;
    uint SkipScale, SkipBias, ResidualGain, UpscaleActivation;
};
float activate(float value, uint channel, uint offset) {
    return value >= 0 ? value : value * Weights[offset + channel];
}
float sample_feature(int2 position, uint channel) {
    uint w = Mode == 4 ? Width * 2 : Width;
    uint h = Mode == 4 ? Height * 2 : Height;
    float result=0;
    if (all(position >= 0) && position.x < int(w) && position.y < int(h)) {
    if (Mode == 0) result=Original.Load(int3(position,0))[min(channel,2u)];
    else if (Mode == 4) {
        uint2 positive=uint2(position);
        uint shuffled = channel * 4 + (positive.y & 1) * 2 + (positive.x & 1);
        float v = Input[shuffled * Width * Height + (positive.y >> 1) * Width + (positive.x >> 1)];
        result=activate(v, channel, UpscaleActivation);
    } else {
        result=Input[channel * Width * Height + position.y * Width + position.x];
    }
    }
    return result;
}
float convolution(uint2 pixel, uint channel) {
    float value = Weights[BiasOffset + channel];
    for (uint input_channel=0; input_channel<InChannels; ++input_channel)
    for (int y=0; y<3; ++y)
    for (int x=0; x<3; ++x) {
        uint coefficient = ((channel * InChannels + input_channel) * 3 + y) * 3 + x;
        value += sample_feature(int2(pixel)+int2(x-1,y-1), input_channel) * Weights[WeightOffset + coefficient];
    }
    return value;
}
float cubic(float distance) {
    float x=abs(distance), a=-0.75;
    float inner=((a+2)*x-(a+3))*x*x+1;
    float outer=((a*x-5*a)*x+8*a)*x-4*a;
    return x<=1 ? inner : x<2 ? outer : 0;
}
float4 bypass(uint2 pixel) {
    float2 position=(float2(pixel)+0.5)*0.5-0.5;
    int2 base=int2(floor(position)), upper=int2(Width-1,Height-1);
    float4 sum=0;
    for(int y=-1;y<=2;++y) for(int x=-1;x<=2;++x)
        sum += Original.Load(int3(clamp(base+int2(x,y),int2(0,0),upper),0)) *
            cubic(position.x-base.x-x)*cubic(position.y-base.y-y);
    // Opacity is independent of the learned RGB graph; keep it within coverage bounds.
    sum.a=saturate(sum.a); return sum;
}
[numthreads(8,8,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint2 pixel=thread.xy;
    uint w=Mode==4?Width*2:Width, h=Mode==4?Height*2:Height;
    if(pixel.x>=w || pixel.y>=h) return;
    if(Mode==4) {
        float4 result=bypass(pixel);
        [unroll] for(uint c=0;c<3;++c)
            result[c]=saturate(result[c]*Weights[SkipScale+c]+Weights[SkipBias+c]+
                convolution(pixel,c)*Weights[ResidualGain]);
        Image[pixel]=result; return;
    }
    uint channel=thread.z;
    if(channel>=OutChannels) return;
    uint index=channel*Width*Height+pixel.y*Width+pixel.x;
    float value=convolution(pixel,channel);
    if(Mode!=3) value=activate(value,channel,ActivationOffset);
    if(Mode==1) value=Output[index]+value*tanh(value);
    if(Mode==2) value+=Head[index];
    Output[index]=value;
}
)hlsl";
}
