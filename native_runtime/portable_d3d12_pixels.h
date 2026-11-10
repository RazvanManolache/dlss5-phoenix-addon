#pragma once
#include "../src/native/support/shader_compiler.h"
#include "../src/native/network/production_network.h"
#include <functional>

namespace phoenix_portable_pixels {
// The selected neural provider still receives its canonical float planes.
// Only the public D3D12 texture codec moves to compute. Raw buffer copies keep
// source alpha bits intact, including signed zero and noncanonical NaN payloads.
class D3D12Codec {
    template<class T> using ComPtr=Microsoft::WRL::ComPtr<T>;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> unpack, pack;
    ComPtr<ID3D12Resource> color, motion, rgb, vectors, read_rgb, read_vectors, upload, packed;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT color_fp{}, motion_fp{};
    uint64_t color_bytes=0, motion_bytes=0, pixels=0;
    struct Constants { unsigned width,height,color_pitch,motion_pitch;float sx,sy;unsigned has_motion,pad; } constants{};
    void bind(ID3D12GraphicsCommandList* command, ID3D12PipelineState* pipeline,
              ID3D12Resource* second, ID3D12Resource* first_output, ID3D12Resource* second_output) {
        command->SetComputeRootSignature(root.Get());command->SetPipelineState(pipeline);
        command->SetComputeRoot32BitConstants(0,8,&constants,0);
        command->SetComputeRootShaderResourceView(1,color->GetGPUVirtualAddress());
        command->SetComputeRootShaderResourceView(2,second->GetGPUVirtualAddress());
        command->SetComputeRootUnorderedAccessView(3,first_output->GetGPUVirtualAddress());
        command->SetComputeRootUnorderedAccessView(4,second_output->GetGPUVirtualAddress());
        command->Dispatch((constants.width+15)/16,(constants.height+15)/16,1);
    }
public:
    D3D12Codec(ID3D12Device* device, unsigned width, unsigned height,
               const std::shared_ptr<phoenix_native::memory::Generation>& accounting) {
        using phoenix_native::hr;
        pixels=uint64_t(width)*height;constants.width=width;constants.height=height;
        auto footprint=[&](DXGI_FORMAT format,D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,uint64_t& bytes){
            D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;desc.Width=width;
            desc.Height=height;desc.DepthOrArraySize=1;desc.MipLevels=1;desc.Format=format;desc.SampleDesc.Count=1;
            device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        };
        footprint(DXGI_FORMAT_R16G16B16A16_FLOAT,color_fp,color_bytes);
        footprint(DXGI_FORMAT_R16G16_FLOAT,motion_fp,motion_bytes);
        constants.color_pitch=color_fp.Footprint.RowPitch;constants.motion_pitch=motion_fp.Footprint.RowPitch;
        auto buffer=[&](uint64_t bytes,D3D12_HEAP_TYPE heap,D3D12_RESOURCE_STATES state){
            auto value=phoenix_native::buffer(device,bytes,heap,state);
            phoenix_native::memory::resource(device,value.Get(),accounting,phoenix_native::memory::Kind::upload);
            return value;
        };
        color=buffer(color_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        motion=buffer(motion_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        rgb=buffer(pixels*12,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        vectors=buffer(pixels*8,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        packed=buffer(color_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        read_rgb=buffer(pixels*12,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        read_vectors=buffer(pixels*8,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        upload=buffer(pixels*12,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_ROOT_PARAMETER parameters[5]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;parameters[0].Constants={0,0,8};
        for(unsigned i=1;i<5;++i){parameters[i].ParameterType=i<3?D3D12_ROOT_PARAMETER_TYPE_SRV:D3D12_ROOT_PARAMETER_TYPE_UAV;
            parameters[i].Descriptor={i<3?i-1:i-3,0};}
        D3D12_ROOT_SIGNATURE_DESC desc{5,parameters,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> serialized,error;
        hr(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&error));
        hr(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root)));
        static constexpr char shader[]=R"hlsl(
cbuffer Config : register(b0) { uint width,height,color_pitch,motion_pitch;float sx,sy;uint has_motion,pad; };
ByteAddressBuffer a : register(t0);ByteAddressBuffer b : register(t1);
RWByteAddressBuffer c : register(u0);RWByteAddressBuffer d : register(u1);
float bounded(uint bits) {
    uint magnitude=bits&32767u;
    float value=f16tof32(bits);
    value=((bits&32768u)&&magnitude!=0u&&magnitude<=31744u)?0.0f:value;
    value=(!(bits&32768u)&&magnitude>15360u&&magnitude<=31744u)?1.0f:value;
    return value;
}
[numthreads(16,16,1)] void unpack(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=width||tid.y>=height)return;uint p=tid.y*width+tid.x;
    uint2 bits=a.Load2(tid.y*color_pitch+tid.x*8);
    c.Store3(p*12,asuint(float3(bounded(bits.x&65535),bounded(bits.x>>16),bounded(bits.y&65535))));
    if(has_motion){uint mv=b.Load(tid.y*motion_pitch+tid.x*4);
        precise float x=f16tof32(mv&65535)*sx;precise float y=f16tof32(mv>>16)*sy;
        d.Store2(p*8,asuint(float2(x,y)));}
}
// SM5 f32tof16 flushes small results on some drivers. Encode with integer
// nearest-even rounding so half subnormals and midpoint ties match the host.
uint encode_half(float value) {
    uint bits=asuint(value),magnitude=bits&0x7fffffffu,sign=(bits>>16)&32768u;
    uint result=0u;
    if(magnitude>=0x7f800000u)result=31744u|(magnitude>0x7f800000u?512u:0u);
    else if(magnitude>=0x477ff000u)result=31744u;
    else if(magnitude>0x33000000u){
        uint exponent=magnitude>>23;
        if(exponent<113u){
            uint mantissa=(magnitude&0x7fffffu)|0x800000u,shift=126u-exponent;
            result=(mantissa+((1u<<(shift-1u))-1u)+((mantissa>>shift)&1u))>>shift;
        }else result=((magnitude+4095u+((magnitude>>13)&1u))>>13)-114688u;
    }
    return sign|result;
}
[numthreads(16,16,1)] void pack(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=width||tid.y>=height)return;uint p=tid.y*width+tid.x,offset=tid.y*color_pitch+tid.x*8;
    float3 result=asfloat(b.Load3(p*12));uint alpha=a.Load(offset+4)&0xffff0000u;
    c.Store2(offset,uint2(encode_half(result.x)|(encode_half(result.y)<<16),encode_half(result.z)|alpha));
}
)hlsl";
        auto pipeline=[&](const char* entry,ComPtr<ID3D12PipelineState>& target){
            ComPtr<ID3DBlob> code,errors;
            const auto compiled=phoenix_shader::compile(shader,sizeof(shader)-1,"phoenix_portable_pixels",nullptr,nullptr,
                entry,"cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,&code,&errors);
            if(FAILED(compiled))throw std::runtime_error(errors?static_cast<const char*>(errors->GetBufferPointer()):"Portable pixel shader compile failed");
            D3D12_COMPUTE_PIPELINE_STATE_DESC state{};state.pRootSignature=root.Get();state.CS={code->GetBufferPointer(),code->GetBufferSize()};
            hr(device->CreateComputePipelineState(&state,IID_PPV_ARGS(&target)));
            phoenix_native::memory::track(target.Get(),accounting,phoenix_native::memory::Kind::pipeline);
        };
        pipeline("unpack",unpack);pipeline("pack",pack);
        phoenix_native::memory::track(root.Get(),accounting,phoenix_native::memory::Kind::pipeline);
    }
    void download(ID3D12GraphicsCommandList* command,ID3D12Resource* source,D3D12_RESOURCE_STATES source_state,
                   ID3D12Resource* mv,D3D12_RESOURCE_STATES mv_state,float sx,float sy) {
        auto snapshot=[&](ID3D12Resource* image,D3D12_RESOURCE_STATES state,ID3D12Resource* target,
                           const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint){
            phoenix_native::transition(command,image,state,D3D12_RESOURCE_STATE_COPY_SOURCE);
            phoenix_native::transition(command,target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION from{image,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},
                to{target,D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,footprint};
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            phoenix_native::transition(command,target,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::transition(command,image,D3D12_RESOURCE_STATE_COPY_SOURCE,state);
        };
        snapshot(source,source_state,color.Get(),color_fp);if(mv)snapshot(mv,mv_state,motion.Get(),motion_fp);
        constants.sx=sx;constants.sy=sy;constants.has_motion=mv!=nullptr;
        bind(command,unpack.Get(),motion.Get(),rgb.Get(),vectors.Get());
        auto read=[&](ID3D12Resource* from,ID3D12Resource* to,uint64_t bytes){
            phoenix_native::transition(command,from,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            command->CopyBufferRegion(to,0,from,0,bytes);
            phoenix_native::transition(command,from,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        };
        read(rgb.Get(),read_rgb.Get(),pixels*12);if(mv)read(vectors.Get(),read_vectors.Get(),pixels*8);
    }
    template<class Function> void planes(Function consume) {
        struct Mapping { ID3D12Resource* resource;void* pointer=nullptr;
            Mapping(ID3D12Resource* r,size_t bytes):resource(r){D3D12_RANGE range{0,bytes};phoenix_native::hr(resource->Map(0,&range,&pointer));}
            ~Mapping(){D3D12_RANGE empty{};resource->Unmap(0,&empty);} } first(read_rgb.Get(),size_t(pixels*12));
        if(constants.has_motion){Mapping second(read_vectors.Get(),size_t(pixels*8));consume(static_cast<const float*>(first.pointer),static_cast<const float*>(second.pointer));}
        else consume(static_cast<const float*>(first.pointer),nullptr);
    }
    void upload_output(ID3D12GraphicsCommandList* command,const float* result,ID3D12Resource* output,D3D12_RESOURCE_STATES state) {
        void* mapped=nullptr;D3D12_RANGE empty{};phoenix_native::hr(upload->Map(0,&empty,&mapped));
        std::memcpy(mapped,result,size_t(pixels*12));D3D12_RANGE written{0,size_t(pixels*12)};upload->Unmap(0,&written);
        bind(command,pack.Get(),upload.Get(),packed.Get(),vectors.Get());
        phoenix_native::transition(command,packed.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(command,output,state,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{packed.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,color_fp},
            to{output,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        phoenix_native::transition(command,output,D3D12_RESOURCE_STATE_COPY_DEST,state);
        phoenix_native::transition(command,packed.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
};
}
