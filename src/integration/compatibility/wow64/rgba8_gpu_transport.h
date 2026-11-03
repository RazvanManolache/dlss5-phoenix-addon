#pragma once
#include "../host/phoenix-format-adapter.h"
#include <stdexcept>
#include <algorithm>

namespace phoenix_wow64 {
// CPU staging remains necessary for legacy game devices. Keep those transfers
// RGBA8, however: expanding them on the x86 CPU doubles PCIe traffic and adds
// two full-frame conversions. Only the native worker owns these D3D12 objects.
class Rgba8GpuTransport {
    // Output table: model SRV, original-half SRV, packed UAV. Input table:
    // unused SRV plus uploaded-byte SRV, with a separate half-image UAV.
    enum Descriptor : unsigned {Enhanced=0,Original=1,Packed=2,HalfInput=3,InputTable=4,UploadBytes=5,DescriptorCount=6};
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D12Device> device_;
    Ptr<ID3D12Resource> color_,output_,upload_,packed_,readback_,zero_;
    Ptr<ID3D12RootSignature> root_;
    Ptr<ID3D12PipelineState> pipeline_,input_pipeline_;
    Ptr<ID3D12DescriptorHeap> descriptors_;
    unsigned width_,height_,increment_=0;
    uint64_t image_bytes_;
    void* mapped_upload_=nullptr;
    void* mapped_readback_=nullptr;

    static void check(HRESULT result) {
        if(FAILED(result))throw std::runtime_error("RGBA8 native GPU transport HRESULT="+std::to_string(uint32_t(result)));
    }
    static D3D12_RESOURCE_BARRIER barrier(ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};return b;
    }
    Ptr<ID3D12Resource> buffer(uint64_t bytes,D3D12_HEAP_TYPE heap,D3D12_RESOURCE_FLAGS flags=D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES properties{};properties.Type=heap;
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;
        d.Height=d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;d.Flags=flags;
        const auto state=heap==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:
            heap==D3D12_HEAP_TYPE_READBACK?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_COMMON;
        Ptr<ID3D12Resource> out;check(device_->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&out)));return out;
    }
    void create_pack_pipeline() {
        // The extra word reports nonfinite RGB instead of silently publishing a
        // black/sanitized image. Source alpha is preserved, as in the CPU path.
        static constexpr char shader[]=R"(
Texture2D<float4> enhanced : register(t0);
Texture2D<float4> original : register(t1);
RWByteAddressBuffer result : register(u0);
cbuffer Extent : register(b0) { uint width,height; }
[numthreads(8,8,1)] void main(uint3 p:SV_DispatchThreadID) {
    if(p.x>=width||p.y>=height)return;
    float3 value=enhanced.Load(int3(p.xy,0)).rgb;
    if(!all(isfinite(value))) { result.InterlockedOr(width*height*4,1);value=0; }
    precise float3 scaled=saturate(value)*255.0;
    uint3 integral=(uint3)scaled;
    uint3 rgb=integral+(uint3)(scaled-(float3)integral>=0.5);
    uint address=(p.y*width+p.x)*4;
    uint alpha=(uint)(original.Load(int3(p.xy,0)).a*255.0+0.5);
    result.Store(address,rgb.x|(rgb.y<<8)|(rgb.z<<16)|(alpha<<24));
})";
        static constexpr char input_shader[]=R"(
ByteAddressBuffer original:register(t1);
RWTexture2D<float4> color:register(u0);
cbuffer Shape:register(b0){uint width,height;};
[numthreads(8,8,1)]void main(uint3 p:SV_DispatchThreadID){
 if(p.x>=width||p.y>=height)return;
 uint word=original.Load((p.y*width+p.x)*4);
 uint4 bytes=uint4(word,word>>8,word>>16,word>>24)&255u;
 precise float4 value=float4(bytes)/255.0;
 // Typed half UAV stores can truncate. Preserve the existing nearest-even
 // half input contract explicitly, including zero.
 uint4 bits=asuint(value);
 uint4 rounded=bits+4095u+((bits>>13)&1u);
 uint4 half_bits=(rounded>>13)-(112u<<10);
 color[p.xy]=f16tof32(value==0?0:half_bits);
})";
        D3D12_DESCRIPTOR_RANGE ranges[]={{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,2,0,0,0},{D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,0,0,0}};
        D3D12_ROOT_PARAMETER parameters[3]{};
        for(unsigned i=0;i<2;++i){parameters[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;parameters[i].DescriptorTable={1,&ranges[i]};}
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;parameters[2].Constants={0,0,2};
        D3D12_ROOT_SIGNATURE_DESC desc{3,parameters,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Ptr<ID3DBlob> blob,errors;
        check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors));
        check(device_->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_)));
        check(phoenix_shader::compile(shader,sizeof(shader)-1,"phoenix-rgba8-native-transport",nullptr,nullptr,"main","cs_5_0",
            D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&blob,&errors));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};pso.pRootSignature=root_.Get();pso.CS={blob->GetBufferPointer(),blob->GetBufferSize()};
        check(device_->CreateComputePipelineState(&pso,IID_PPV_ARGS(&pipeline_)));
        check(phoenix_shader::compile(input_shader,sizeof(input_shader)-1,"phoenix-rgba8-direct-input",nullptr,nullptr,"main","cs_5_0",
            D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&blob,&errors));
        pso.CS={blob->GetBufferPointer(),blob->GetBufferSize()};
        check(device_->CreateComputePipelineState(&pso,IID_PPV_ARGS(&input_pipeline_)));
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,DescriptorCount,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        check(device_->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&descriptors_)));
        increment_=device_->GetDescriptorHandleIncrementSize(heap.Type);
        auto cpu=descriptors_->GetCPUDescriptorHandleForHeapStart();
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=output_->GetDesc().Format;srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;srv.Texture2D.MipLevels=1;
            device_->CreateShaderResourceView(output_.Get(),&srv,cpu);cpu.ptr+=increment_;
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC original{};original.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        original.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;original.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;original.Texture2D.MipLevels=1;
        device_->CreateShaderResourceView(color_.Get(),&original,cpu);cpu.ptr+=increment_;
        D3D12_SHADER_RESOURCE_VIEW_DESC source{};source.Format=DXGI_FORMAT_R32_TYPELESS;
        source.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;source.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        source.Buffer.NumElements=UINT(image_bytes_/4);source.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};uav.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;uav.Format=DXGI_FORMAT_R32_TYPELESS;
        uav.Buffer.NumElements=UINT(image_bytes_/4+1);uav.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(packed_.Get(),nullptr,&uav,cpu);
        cpu.ptr+=increment_;uav={};uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;uav.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        device_->CreateUnorderedAccessView(color_.Get(),nullptr,&uav,cpu);
        cpu.ptr+=increment_;device_->CreateShaderResourceView(output_.Get(),&original,cpu);
        cpu.ptr+=increment_;device_->CreateShaderResourceView(upload_.Get(),&source,cpu);
    }
  public:
    Rgba8GpuTransport(ID3D12Device* device,unsigned width,unsigned height):device_(device),width_(width),height_(height),image_bytes_(uint64_t(width)*height*4) {
        if(!width||!height||width>8192||height>8192)throw std::runtime_error("Invalid RGBA8 native transport extent");
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;
        d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&color_)));
        check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&output_)));
        upload_=buffer(image_bytes_,D3D12_HEAP_TYPE_UPLOAD);
        packed_=buffer(image_bytes_+4,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        readback_=buffer(image_bytes_+4,D3D12_HEAP_TYPE_READBACK);zero_=buffer(4,D3D12_HEAP_TYPE_UPLOAD);
        void* mapped=nullptr;D3D12_RANGE empty{};check(zero_->Map(0,&empty,&mapped));std::memset(mapped,0,4);zero_->Unmap(0,nullptr);
        // D3D12 upload/readback heaps support persistent CPU mappings. Queue
        // fences still govern reuse; mapping is not a synchronization primitive.
        check(upload_->Map(0,&empty,&mapped_upload_));
        D3D12_RANGE readable{0,size_t(image_bytes_+4)};check(readback_->Map(0,&readable,&mapped_readback_));
        create_pack_pipeline();
    }
    ~Rgba8GpuTransport() {
        if(mapped_upload_)upload_->Unmap(0,nullptr);
        if(mapped_readback_){D3D12_RANGE empty{};readback_->Unmap(0,&empty);}
    }
    ID3D12Resource* color()const{return color_.Get();}
    ID3D12Resource* output()const{return output_.Get();}
    uint64_t image_bytes()const{return image_bytes_;}
    // Caller submits this list before native evaluation on the SAME queue.
    // It must retire before the upload buffer or allocator is reused.
    void record_input(ID3D12GraphicsCommandList* list,const void* rgba) {
        std::memcpy(mapped_upload_,rgba,size_t(image_bytes_));
        auto b=barrier(color(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);list->ResourceBarrier(1,&b);
        ID3D12DescriptorHeap* heaps[]={descriptors_.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root_.Get());list->SetPipelineState(input_pipeline_.Get());
        auto gpu=descriptors_->GetGPUDescriptorHandleForHeapStart();gpu.ptr+=InputTable*increment_;list->SetComputeRootDescriptorTable(0,gpu);
        gpu=descriptors_->GetGPUDescriptorHandleForHeapStart();gpu.ptr+=HalfInput*increment_;list->SetComputeRootDescriptorTable(1,gpu);
        const unsigned extent[]={width_,height_};list->SetComputeRoot32BitConstants(2,2,extent,0);
        list->Dispatch((width_+7)/8,(height_+7)/8,1);
        std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(1,&b);
    }
    void record_output(ID3D12GraphicsCommandList* list) {
        auto b=barrier(packed_.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);list->ResourceBarrier(1,&b);
        list->CopyBufferRegion(packed_.Get(),image_bytes_,zero_.Get(),0,4);
        D3D12_RESOURCE_BARRIER barriers[]={barrier(packed_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            barrier(output(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            barrier(color(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(3,barriers);
        ID3D12DescriptorHeap* heaps[]={descriptors_.Get()};list->SetDescriptorHeaps(1,heaps);list->SetComputeRootSignature(root_.Get());list->SetPipelineState(pipeline_.Get());
        auto gpu=descriptors_->GetGPUDescriptorHandleForHeapStart();list->SetComputeRootDescriptorTable(0,gpu);gpu.ptr+=Packed*increment_;list->SetComputeRootDescriptorTable(1,gpu);
        const unsigned extent[]={width_,height_};list->SetComputeRoot32BitConstants(2,2,extent,0);list->Dispatch((width_+7)/8,(height_+7)/8,1);
        barriers[0]=barrier(packed_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        std::swap(barriers[1].Transition.StateBefore,barriers[1].Transition.StateAfter);std::swap(barriers[2].Transition.StateBefore,barriers[2].Transition.StateAfter);list->ResourceBarrier(3,barriers);
        list->CopyBufferRegion(readback_.Get(),0,packed_.Get(),0,image_bytes_+4);
        b=barrier(packed_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);list->ResourceBarrier(1,&b);
    }
    // A successful GPU fence must precede this call. Never publish invalid RGB.
    void read_output(void* rgba) {
        uint32_t invalid=0;std::memcpy(&invalid,static_cast<const char*>(mapped_readback_)+image_bytes_,4);
        if(!invalid)std::memcpy(rgba,mapped_readback_,size_t(image_bytes_));
        if(invalid)throw std::runtime_error("Native bridge returned nonfinite pixels");
    }
};
}
