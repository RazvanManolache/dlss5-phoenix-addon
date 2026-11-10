#pragma once
#include "local_d3d12_identity.h"
#include "host/phoenix-format-adapter.h"
#include "../../pipeline/paired_screenshot.h"
#include "../../../native_runtime/pipeline_client.h"
#include <dxgi1_4.h>
#include <d3d11_1.h>
#include <array>

namespace phoenix_compat {
// The OS sharing interfaces are available on modern DX11 devices even when
// a game requests feature level 11.0. ReShade's abstraction requires 11.1 for
// its NT export flag; native allocation avoids using an unsupported API flag.
inline Microsoft::WRL::ComPtr<ID3D11Texture2D> portable_gpu_texture11(ID3D11Device* device,unsigned width,unsigned height,HANDLE* handle){
    D3D11_TEXTURE2D_DESC desc{};desc.Width=width;desc.Height=height;desc.MipLevels=desc.ArraySize=1;
    desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_DEFAULT;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED_NTHANDLE|D3D11_RESOURCE_MISC_SHARED;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> image;Microsoft::WRL::ComPtr<IDXGIResource1> shared;
    if(FAILED(device->CreateTexture2D(&desc,nullptr,&image))||FAILED(image.As(&shared))||
        FAILED(shared->CreateSharedHandle(nullptr,DXGI_SHARED_RESOURCE_READ|DXGI_SHARED_RESOURCE_WRITE,nullptr,handle)))
        throw std::runtime_error("Native DX11 NT texture export unavailable");
    return image;
}
// Duplicated before launching the worker. Resizing may release the render-facing
// resources while this owner and the worker's imported COM objects retire.
struct PortableGpuHandles {
    std::array<HANDLE,2> handles{};
    LUID adapter{};
    PortableGpuHandles(HANDLE color,HANDLE output,LUID luid):adapter(luid){
        HANDLE source[]{color,output};
        for(unsigned i=0;i<2;++i)if(!DuplicateHandle(GetCurrentProcess(),source[i],GetCurrentProcess(),&handles[i],0,FALSE,DUPLICATE_SAME_ACCESS)){
            for(auto handle:handles)if(handle)CloseHandle(handle);
            throw std::runtime_error("Cannot retain portable GPU mailbox");
        }
    }
    ~PortableGpuHandles(){for(auto handle:handles)if(handle)CloseHandle(handle);}
    PortableGpuHandles(const PortableGpuHandles&)=delete;
};
class PortableGpuFrames {
    template<class T>using Com=Microsoft::WRL::ComPtr<T>;
    struct Resources {
        Com<ID3D12Device> wrapper_device,device;
        Com<ID3D12CommandQueue> wrapper_queue,queue;
        Com<ID3D12CommandAllocator> allocator;
        Com<ID3D12GraphicsCommandList> command;
        Com<ID3D12Resource> color,output;
        Com<ID3D12Fence> fence;
        PhoenixFormatAdapter adapter{true};
        phoenix_pipeline::screenshots::D3D12Pair pair;
        HANDLE event{};uint64_t serial{};
        bool registered=false;
        ~Resources(){
            if(registered&&in_process_api.local_queue_lifetime)try{in_process_api.local_queue_lifetime(queue.Get(),false);}catch(...){in_process_gpu_uncertain=true;}
            if(event)CloseHandle(event);
        }
    };
    std::unique_ptr<Resources> r_=std::make_unique<Resources>();
    static void check(HRESULT value){if(FAILED(value))throw std::runtime_error("Portable GPU mailbox HRESULT "+std::to_string(uint32_t(value)));}
    void wait(){
        auto& r=*r_;const auto completed=r.fence->GetCompletedValue();
        if(completed==UINT64_MAX)throw std::runtime_error("Portable GPU mailbox device lost");
        if(completed<r.serial){check(r.fence->SetEventOnCompletion(r.serial,r.event));
            if(WaitForSingleObject(r.event,30000)!=WAIT_OBJECT_0)throw std::runtime_error("Portable GPU mailbox did not retire");}
        check(r.device->GetDeviceRemovedReason());
    }
    void begin(){wait();check(r_->allocator->Reset());check(r_->command->Reset(r_->allocator.Get(),nullptr));}
    void submit(){auto& r=*r_;check(r.command->Close());ID3D12CommandList* list[]{r.command.Get()};
        r.queue->ExecuteCommandLists(1,list);check(r.queue->Signal(r.fence.Get(),++r.serial));wait();}
public:
    PortableGpuFrames(const PortableGpuHandles& handles,unsigned width,unsigned height){
        auto& r=*r_;Com<IDXGIFactory4> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        Com<IDXGIAdapter> adapter;check(factory->EnumAdapterByLuid(handles.adapter,IID_PPV_ARGS(&adapter)));
        check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&r.device)));
        D3D12_COMMAND_QUEUE_DESC desc{};check(r.device->CreateCommandQueue(&desc,IID_PPV_ARGS(&r.queue)));
        resolve_local_d3d12_identity(r.device,r.queue,r.wrapper_device,r.wrapper_queue);
        if(in_process_api.local_queue_lifetime){in_process_api.local_queue_lifetime(r.queue.Get(),true);r.registered=true;}
        check(r.device->OpenSharedHandle(handles.handles[0],IID_PPV_ARGS(&r.color)));
        check(r.device->OpenSharedHandle(handles.handles[1],IID_PPV_ARGS(&r.output)));
        for(auto* image:{r.color.Get(),r.output.Get()}){const auto d=image->GetDesc();
            if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.Width!=width||d.Height!=height||
               d.MipLevels!=1||d.DepthOrArraySize!=1||d.SampleDesc.Count!=1||d.Format!=DXGI_FORMAT_R8G8B8A8_UNORM)
                throw std::runtime_error("Portable GPU mailbox texture mismatch");}
        check(r.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&r.allocator)));
        check(r.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,r.allocator.Get(),nullptr,IID_PPV_ARGS(&r.command)));check(r.command->Close());
        check(r.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&r.fence)));r.event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!r.event)throw std::runtime_error("Cannot create portable GPU completion event");
        check(r.adapter.Ensure(r.device.Get(),width,height,DXGI_FORMAT_R8G8B8A8_UNORM));
    }
    ~PortableGpuFrames(){if(r_&&r_->fence)try{wait();}catch(...){in_process_gpu_uncertain=true;r_.release();}}
    ID3D12Device* device()const{return r_->device.Get();}
    ID3D12CommandQueue* queue()const{return r_->queue.Get();}
    bool request_pair(const std::filesystem::path& folder){return r_->pair.output.request(folder);}
    Json pair_status()const{return r_->pair.output.result;}
    void frame(phoenix_cpp::PipelineClient& pipeline,const std::filesystem::path& root,const Json& runner,
        unsigned width,unsigned height,const Json& metadata,uint64_t sequence,int64_t timestamp,double age,const std::string& tiny){
        try{auto& r=*r_;begin();r.pair.start(r.command.Get(),r.color.Get(),D3D12_RESOURCE_STATE_COMMON,sequence,false,{{"route","portable shared GPU mailbox"}});
        r.adapter.RecordInput(r.device.Get(),r.command.Get(),r.color.Get());submit();
        pipeline.frame_gpu(root,runner,r.adapter.Color(),r.adapter.Output(),width,height,metadata,sequence,timestamp,age,tiny);
        begin();r.adapter.RecordOutput(r.device.Get(),r.command.Get(),r.output.Get());
        r.pair.finish(r.command.Get(),r.output.Get(),D3D12_RESOURCE_STATE_COMMON,r.serial+1);submit();
        r.pair.submitted(r.command.Get());r.pair.poll(r.fence->GetCompletedValue());
        }catch(...){in_process_gpu_uncertain=true;(void)r_.release();throw;}
    }
};
}
