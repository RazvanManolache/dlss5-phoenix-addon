// Isolated D3D12 transfer test; no model, game, helper or WOW64 bootstrap.
#define NOMINMAX
#include "../../src/integration/compatibility/wow64/rgba8_gpu_transport.h"
#include "../../src/integration/compatibility/cpu_color_transfer.h"
#include <dxgi1_4.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <iostream>
using Microsoft::WRL::ComPtr;
static void check(HRESULT result){if(FAILED(result))throw std::runtime_error("GPU transport test HRESULT="+std::to_string(uint32_t(result)));}
int main()try {
    ComPtr<ID3D12Debug> debug;if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    for(unsigned i=0;;++i){check(factory->EnumAdapters1(i,&adapter));DXGI_ADAPTER_DESC1 d{};check(adapter->GetDesc1(&d));if(d.VendorId==0x10de&&!(d.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))break;adapter.Reset();}
    ComPtr<ID3D12Device> device;check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
    ComPtr<ID3D12InfoQueue> info;if(SUCCEEDED(device.As(&info)))info->ClearStoredMessages();
    ComPtr<ID3D12CommandQueue> queue;D3D12_COMMAND_QUEUE_DESC q{};check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list;check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
    ComPtr<ID3D12Fence> fence;check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));uint64_t serial=0;
    auto submit=[&]{check(list->Close());ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);check(queue->Signal(fence.Get(),++serial));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event)throw std::runtime_error("Test fence event failed");
        auto registered=fence->SetEventOnCompletion(serial,event);auto waited=SUCCEEDED(registered)?WaitForSingleObject(event,30000):WAIT_FAILED;CloseHandle(event);check(registered);
        if(waited!=WAIT_OBJECT_0)throw std::runtime_error("GPU transport test fence timeout");check(device->GetDeviceRemovedReason());check(allocator->Reset());check(list->Reset(allocator.Get(),nullptr));};
    unsigned checks=0;
    for(unsigned width:{1u,17u,256u,257u}) {
        const unsigned height=width==257?256:3;const size_t pixels=size_t(width)*height;
        phoenix_wow64::Rgba8GpuTransport transport(device.Get(),width,height);
        std::vector<uint8_t> input(pixels*4),output(pixels*4);
        for(size_t i=0;i<input.size();++i)input[i]=uint8_t(i*73);
        transport.record_input(list.Get(),input.data());submit();
        // Round-trip the GPU's input conversion before testing arbitrary model outputs.
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={transport.color(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE};list->ResourceBarrier(1,&b);
        b.Transition={transport.output(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST};list->ResourceBarrier(1,&b);
        list->CopyResource(transport.output(),transport.color());
        b.Transition={transport.color(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON};list->ResourceBarrier(1,&b);
        b.Transition={transport.output(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON};list->ResourceBarrier(1,&b);
        transport.record_output(list.Get());submit();transport.read_output(output.data());
        if(output!=input)throw std::runtime_error("GPU RGBA8 round trip differs");++checks;
        auto desc=transport.output()->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};uint64_t bytes=0;device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
        D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_UPLOAD;D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;bd.Width=bytes;bd.Height=bd.DepthOrArraySize=bd.MipLevels=bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> upload;check(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload)));
        hp.Type=D3D12_HEAP_TYPE_READBACK;ComPtr<ID3D12Resource> decoded;check(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&bd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&decoded)));
        b.Transition={transport.color(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE};list->ResourceBarrier(1,&b);
        D3D12_TEXTURE_COPY_LOCATION decoded_to{},decoded_from{};decoded_to.pResource=decoded.Get();decoded_to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;decoded_to.PlacedFootprint=fp;
        decoded_from.pResource=transport.color();decoded_from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;list->CopyTextureRegion(&decoded_to,0,0,0,&decoded_from,nullptr);
        std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(1,&b);submit();
        void* decoded_data=nullptr;D3D12_RANGE decoded_range{0,size_t(bytes)},decoded_empty{};check(decoded->Map(0,&decoded_range,&decoded_data));
        std::string mismatch;for(unsigned y=0;y<height;++y){const auto* row=reinterpret_cast<const uint16_t*>(static_cast<const char*>(decoded_data)+fp.Offset+size_t(y)*fp.Footprint.RowPitch);
            for(size_t c=0;c<size_t(width)*4;++c){const auto value=input[size_t(y)*width*4+c];const auto expected=DirectX::PackedVector::XMConvertFloatToHalf(float(value)/255.f);if(row[c]!=expected&&mismatch.empty())mismatch="byte="+std::to_string(value)+", CPU half="+std::to_string(expected)+", GPU half="+std::to_string(row[c]);++checks;}}
        decoded->Unmap(0,&decoded_empty);if(!mismatch.empty())throw std::runtime_error("GPU input half conversion differs: "+mismatch);
        std::vector<uint16_t> halves(pixels*4);
        for(size_t p=0;p<pixels;++p){auto value=uint16_t(p);if((value&0x7c00)==0x7c00)value=0;for(unsigned c=0;c<3;++c)halves[p*4+c]=value;halves[p*4+3]=0x7c01;}
        auto seed=[&]{void* data=nullptr;D3D12_RANGE empty{};check(upload->Map(0,&empty,&data));for(unsigned y=0;y<height;++y)std::memcpy(static_cast<char*>(data)+fp.Offset+size_t(y)*fp.Footprint.RowPitch,halves.data()+size_t(y)*width*4,size_t(width)*8);upload->Unmap(0,nullptr);
            b.Transition={transport.output(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST};list->ResourceBarrier(1,&b);
            D3D12_TEXTURE_COPY_LOCATION to{},from{};to.pResource=transport.output();to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;from.pResource=upload.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=fp;list->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(1,&b);transport.record_output(list.Get());submit();};
        seed();transport.read_output(output.data());
        for(size_t p=0;p<pixels;++p){for(unsigned c=0;c<3;++c){auto expected=phoenix_compat::cpu_color::unorm8(DirectX::PackedVector::XMConvertHalfToFloat(halves[p*4+c]));if(output[p*4+c]!=expected)throw std::runtime_error("GPU half quantization differs");++checks;}if(output[p*4+3]!=input[p*4+3])throw std::runtime_error("GPU alpha changed");++checks;}
        halves[0]=0x7c00;seed();std::fill(output.begin(),output.end(),77);bool rejected=false;try{transport.read_output(output.data());}catch(const std::runtime_error&){rejected=true;}
        if(!rejected||!std::all_of(output.begin(),output.end(),[](uint8_t b){return b==77;}))throw std::runtime_error("Nonfinite GPU RGB was published");++checks;
        halves[0]=0;seed();transport.read_output(output.data());++checks; // Error word must reset on the next frame.
    }
    if(info)for(UINT64 i=0;i<info->GetNumStoredMessages();++i){SIZE_T bytes=0;check(info->GetMessage(i,nullptr,&bytes));std::vector<uint8_t> storage(bytes);auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());check(info->GetMessage(i,message,&bytes));if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)throw std::runtime_error(message->pDescription);}
    std::cout<<"{\"passed\":true,\"gpu_used\":true,\"checks\":"<<checks<<"}\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
