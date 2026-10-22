#pragma once
// Shared, bounded D3D12 fixture. Software by default; hardware is opt-in.
#include "../../src/native/support/production_direct.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>
namespace phoenix_test {
using Microsoft::WRL::ComPtr;
using phoenix_native::hr;
using phoenix_bundle::require;
struct Gpu {
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> cmd;ComPtr<ID3D12Fence> fence;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    std::vector<ComPtr<ID3D12CommandAllocator>> retired_allocators;
    std::vector<ComPtr<ID3D12GraphicsCommandList>> retired_lists;
    uint64_t serial=0;
    Gpu(bool hardware=false){
        ComPtr<ID3D12Debug> debug;hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));ComPtr<IDXGIAdapter1> adapter;
        if(hardware){for(unsigned index=0;;++index){hr(factory->EnumAdapterByGpuPreference(index,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter)));
            DXGI_ADAPTER_DESC1 d{};hr(adapter->GetDesc1(&d));if(d.VendorId==0x10de&&!(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))break;adapter.Reset();}}
        else hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC desc{};hr(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)));
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&cmd)));
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    }
    auto texture(unsigned w,unsigned h,bool motion=false){
        const auto format=motion?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT;
        auto result=phoenix_native::image(device.Get(),w,h,format,D3D12_RESOURCE_STATE_COPY_DEST);
        const auto desc=result->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};uint64_t bytes=0;
        device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        auto upload=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* data=nullptr;D3D12_RANGE empty{};hr(upload->Map(0,&empty,&data));memset(data,0,size_t(bytes));
        if(!motion)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
            auto* p=reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(data)+fp.Footprint.RowPitch*y)+x*4;
            for(unsigned c=0;c<4;++c)p[c]=DirectX::PackedVector::XMConvertFloatToHalf(c==3?1.f:float((x*37+y*11+c*51)%256)/255);
        }
        upload->Unmap(0,nullptr);D3D12_TEXTURE_COPY_LOCATION from{upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,fp},to{result.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        cmd->CopyTextureRegion(&to,0,0,0,&from,nullptr);phoenix_native::transition(cmd.Get(),result.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        uploads.push_back(upload);return result;
    }
    auto readback(ID3D12Resource* image){
        const auto d=image->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};uint64_t bytes=0;device->GetCopyableFootprints(&d,0,1,0,&fp,nullptr,nullptr,&bytes);
        auto result=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        phoenix_native::transition(cmd.Get(),image,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{image,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},to{result.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,fp};
        cmd->CopyTextureRegion(&to,0,0,0,&from,nullptr);phoenix_native::transition(cmd.Get(),image,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        return std::pair{result,fp};
    }
    void next(){retired_allocators.push_back(allocator);retired_lists.push_back(cmd);allocator.Reset();cmd.Reset();
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&cmd)));}
    void submit(bool wait=true){hr(cmd->Close());ID3D12CommandList* lists[]{cmd.Get()};queue->ExecuteCommandLists(1,lists);hr(queue->Signal(fence.Get(),++serial));
        if(!wait)return;finish();}
    void finish(){
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"Fence event");hr(fence->SetEventOnCompletion(serial,event));const auto result=WaitForSingleObject(event,30000);CloseHandle(event);require(result==WAIT_OBJECT_0,"Chain timeout");hr(device->GetDeviceRemovedReason());}
    void validate(){ComPtr<ID3D12InfoQueue> messages;hr(device.As(&messages));
        for(UINT64 n=0;n<messages->GetNumStoredMessages();++n){SIZE_T bytes=0;hr(messages->GetMessage(n,nullptr,&bytes));std::vector<uint8_t> buffer(bytes);auto* m=reinterpret_cast<D3D12_MESSAGE*>(buffer.data());hr(messages->GetMessage(n,m,&bytes));if(m->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)throw std::runtime_error(m->pDescription);}}
};
}
