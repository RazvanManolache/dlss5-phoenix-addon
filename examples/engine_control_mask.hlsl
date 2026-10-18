// Engine-side producer: object/material IDs stay outside the neural model.
// Dispatch at output extent after the ID buffer has been generated. The host
// transitions Mask to the feeder's required state before supplying it.
Texture2D<uint> ObjectOrMaterialID : register(t0);
struct ObjectControl { uint id; float intensity; float tone; float structure; };
StructuredBuffer<ObjectControl> Controls : register(t1);
RWTexture2D<float4> Mask : register(u0);
cbuffer Selection : register(b0) { uint Width,Height,Count,Reserved; };
[numthreads(16,16,1)] void main(uint3 pixel:SV_DispatchThreadID) {
    if(pixel.x>=Width||pixel.y>=Height)return;
    uint id=ObjectOrMaterialID.Load(int3(pixel.xy,0));float3 strengths=1;
    for(uint i=0;i<Count;++i)if(Controls[i].id==id) {
        strengths=saturate(float3(Controls[i].intensity,Controls[i].tone,Controls[i].structure));break;
    }
    Mask[pixel.xy]=float4(strengths,1); // alpha reserved by Phoenix RGB/1
}
