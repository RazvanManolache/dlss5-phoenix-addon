#pragma once
#include "../native/support/compute_pass.h"
#include "tiny_shader.h"
#include "tiny_model.h"
#include <fstream>

namespace phoenix_pipeline {
class TinyUpscaler {
    template<class T> using Com = Microsoft::WRL::ComPtr<T>;
    struct Layer { unsigned weights, bias, activation, inputs, outputs; };
    struct Constants { unsigned width,height,inputs,outputs,weights,bias,activation,mode,
        skip_scale=0,skip_bias=3,residual_gain=6,up_activation=0; };
    Com<ID3D12Device> device_;
    Com<ID3D12RootSignature> root_;
    Com<ID3D12PipelineState> program_;
    Com<ID3D12DescriptorHeap> views_;
    Com<ID3D12Resource> weights_, current_, temporary_, head_, up_;
    std::array<Layer,12> layers_{};
    unsigned width_,height_,up_activation_=0;
    uint64_t owned_bytes_=0;
    ID3D12Resource* bound_source_=nullptr;
    ID3D12Resource* bound_output_=nullptr;

    static void require(bool ok, const char* why) { phoenix_bundle::require(ok,why); }
    static void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        phoenix_native::transition(cmd,resource,before,after);
    }
    Com<ID3D12Resource> buffer(uint64_t bytes, bool upload=false) {
        require(bytes && bytes<=UINT32_MAX,"Tiny tensor exceeds native buffer limits");
        auto result=phoenix_native::buffer(device_.Get(),bytes,upload?D3D12_HEAP_TYPE_UPLOAD:D3D12_HEAP_TYPE_DEFAULT,
            upload?D3D12_RESOURCE_STATE_GENERIC_READ:D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        owned_bytes_+=bytes; return result;
    }
    void dispatch(ID3D12GraphicsCommandList* cmd, const Layer& layer, unsigned mode,
        ID3D12Resource* input, ID3D12Resource* output) {
        Constants c{width_,height_,layer.inputs,layer.outputs,layer.weights,layer.bias,layer.activation,mode};
        c.up_activation=up_activation_;
        cmd->SetComputeRootShaderResourceView(0,input->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(1,weights_->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(2,head_->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3,output->GetGPUVirtualAddress());
        cmd->SetComputeRoot32BitConstants(4,12,&c,0);
        cmd->Dispatch(((mode==4?width_*2:width_)+7)/8,((mode==4?height_*2:height_)+7)/8,mode==4?1:layer.outputs);
    }
  public:
    static constexpr const char* weights_sha256=tiny_weights_sha256;
    static constexpr unsigned weights_bytes=tiny_weights_bytes;
    TinyUpscaler(ID3D12Device* device, unsigned width, unsigned height, const std::filesystem::path& file)
        :device_(device),width_(width),height_(height) {
        require(width&&height&&width<=8192&&height<=8192,"Tiny 2x input extent exceeds texture limits");
        require(file.is_absolute()&&std::filesystem::is_regular_file(file)&&
            !std::filesystem::is_symlink(file)&&std::filesystem::file_size(file)==weights_bytes,"Tiny 2x weights are missing or invalid");
        std::vector<uint8_t> data(weights_bytes); std::ifstream stream(file,std::ios::binary);
        require(bool(stream.read(reinterpret_cast<char*>(data.data()),data.size())),"Cannot read tiny weights");
        require(phoenix_model::weight_digest(data.data(),data.size())==weights_sha256,"Tiny 2x coefficient digest mismatch");
        unsigned offset=7;
        for(unsigned i=0;i<layers_.size();++i) {
            unsigned inputs=i==0?3:28, outputs=i==10?112:i==11?3:28;
            auto& layer=layers_[i]; layer={offset,offset+outputs*inputs*9,0,inputs,outputs};
            offset=layer.bias+outputs;
            if(i<10) { layer.activation=offset; offset+=outputs; }
            if(i==10) { up_activation_=offset;offset+=28; }
        }
        require(offset*4==weights_bytes,"Tiny 2x coefficient layout changed");
        weights_=buffer(data.size(),true); void* mapped=nullptr; D3D12_RANGE empty{};
        phoenix_native::hr(weights_->Map(0,&empty,&mapped)); memcpy(mapped,data.data(),data.size()); weights_->Unmap(0,nullptr);
        const uint64_t pixels=uint64_t(width)*height;
        current_=buffer(pixels*28*4); temporary_=buffer(pixels*28*4); head_=buffer(pixels*28*4); up_=buffer(pixels*112*4);
        D3D12_ROOT_PARAMETER params[6]{};
        for(unsigned i=0;i<3;++i) {params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[i].Descriptor.ShaderRegister=i;}
        params[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;params[3].Descriptor.ShaderRegister=0;
        params[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[4].Constants={0,0,12};
        D3D12_DESCRIPTOR_RANGE ranges[]={{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,3,0,0},{D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,1,0,1}};
        params[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[5].DescriptorTable={2,ranges};
        D3D12_ROOT_SIGNATURE_DESC description{6,params,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Com<ID3DBlob> signature,errors;
        phoenix_native::hr(D3D12SerializeRootSignature(&description,D3D_ROOT_SIGNATURE_VERSION_1,&signature,&errors));
        phoenix_native::hr(device->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),IID_PPV_ARGS(&root_)));
        auto code=phoenix_addon::phoenix::direct::Pass::compile(tiny_shader);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};pso.pRootSignature=root_.Get();pso.CS={code->GetBufferPointer(),code->GetBufferSize()};
        phoenix_native::hr(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&program_)));
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,2,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        phoenix_native::hr(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&views_)));
    }
    uint64_t workspace_bytes() const { return owned_bytes_; }
    // Immutable weights/workspace are shared on one serialized queue. Only the
    // view table needs a separate retirement slot for asynchronous recordings.
    TinyUpscaler(const TinyUpscaler& other)
        :device_(other.device_),root_(other.root_),program_(other.program_),weights_(other.weights_),
         current_(other.current_),temporary_(other.temporary_),head_(other.head_),up_(other.up_),
         layers_(other.layers_),width_(other.width_),height_(other.height_),up_activation_(other.up_activation_),owned_bytes_(other.owned_bytes_) {
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,2,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        phoenix_native::hr(device_->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&views_)));
    }
    // Caller serializes submissions and retires this recording before reuse/destruction.
    // Source and destination states are restored. No CPU pixels, fixed canvas or Python.
    void record(ID3D12GraphicsCommandList* cmd,ID3D12Resource* source,D3D12_RESOURCE_STATES source_state,
        ID3D12Resource* destination,D3D12_RESOURCE_STATES destination_state) {
        const auto src=source->GetDesc(),dst=destination->GetDesc();
        require(src.Width==width_&&src.Height==height_&&dst.Width==width_*2&&dst.Height==height_*2&&
            src.Format==DXGI_FORMAT_R16G16B16A16_FLOAT&&dst.Format==src.Format&&source!=destination,"Tiny texture extent/format mismatch");
        if(bound_source_!=source||bound_output_!=destination) {
        auto cpu=views_->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=src.Format;srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;srv.Texture2D.MipLevels=1;
        device_->CreateShaderResourceView(source,&srv,cpu);
        cpu.ptr+=device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};uav.Format=dst.Format;uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
        device_->CreateUnorderedAccessView(destination,nullptr,&uav,cpu);
        bound_source_=source;bound_output_=destination;
        }
        auto* heap=views_.Get();cmd->SetDescriptorHeaps(1,&heap);cmd->SetComputeRootSignature(root_.Get());cmd->SetPipelineState(program_.Get());
        cmd->SetComputeRootDescriptorTable(5,views_->GetGPUDescriptorHandleForHeapStart());
        constexpr auto read=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,write=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        transition(cmd,source,source_state,read); transition(cmd,destination,destination_state,write);
        dispatch(cmd,layers_[0],0,temporary_.Get(),current_.Get());
        transition(cmd,current_.Get(),write,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd,head_.Get(),write,D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyBufferRegion(head_.Get(),0,current_.Get(),0,uint64_t(width_)*height_*28*4);
        transition(cmd,current_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,read);transition(cmd,head_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,read);
        for(unsigned block=0;block<4;++block) {
            dispatch(cmd,layers_[1+block*2],5,current_.Get(),temporary_.Get()); // ordinary Conv + PReLU
            transition(cmd,temporary_.Get(),write,read);transition(cmd,current_.Get(),read,write);
            dispatch(cmd,layers_[2+block*2],1,temporary_.Get(),current_.Get());
            transition(cmd,current_.Get(),write,read);transition(cmd,temporary_.Get(),read,write);
        }
        dispatch(cmd,layers_[9],2,current_.Get(),temporary_.Get());transition(cmd,temporary_.Get(),write,read);
        dispatch(cmd,layers_[10],3,temporary_.Get(),up_.Get());transition(cmd,up_.Get(),write,read);
        transition(cmd,current_.Get(),read,write);
        dispatch(cmd,layers_[11],4,up_.Get(),current_.Get());
        transition(cmd,temporary_.Get(),read,write);transition(cmd,up_.Get(),read,write);transition(cmd,head_.Get(),read,write);
        transition(cmd,source,read,source_state);transition(cmd,destination,write,destination_state);
    }
};
}
