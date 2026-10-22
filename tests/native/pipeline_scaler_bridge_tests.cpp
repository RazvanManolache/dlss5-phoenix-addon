#include "../../src/native/support/resolution_bridge.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <iostream>
using Microsoft::WRL::ComPtr;
using phoenix_native::hr;
using phoenix_bundle::require;
namespace spatial=phoenix_addon::phoenix::direct;
using Json=phoenix_bundle::Json;
int main(int argc,char** argv) {
    try {
        const bool tiny=argc==2;
        ComPtr<ID3D12Debug> debug;hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory4> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> adapter;hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        ComPtr<ID3D12Device> device;hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12CommandQueue> queue;D3D12_COMMAND_QUEUE_DESC q{};hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator;hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> command;hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&command)));
        constexpr unsigned width=7,height=5,output_width=13,output_height=9;
        auto low=phoenix_native::image(device.Get(),width,height,DXGI_FORMAT_R32G32B32A32_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto game=phoenix_native::image(device.Get(),output_width,output_height,DXGI_FORMAT_R32G32B32A32_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        std::vector<ComPtr<ID3D12Resource>> uploads;
        auto initialize=[&](ID3D12Resource* resource,bool original) {
            const auto description=resource->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};uint64_t bytes=0;
            device->GetCopyableFootprints(&description,0,1,0,&footprint,nullptr,nullptr,&bytes);
            auto upload=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped=nullptr;D3D12_RANGE empty{};hr(upload->Map(0,&empty,&mapped));
            for(unsigned y=0;y<description.Height;++y)for(unsigned x=0;x<description.Width;++x) {
                auto* pixel=reinterpret_cast<float*>(static_cast<uint8_t*>(mapped)+footprint.Footprint.RowPitch*y)+x*4;
                pixel[0]=.5f;pixel[1]=.5f;pixel[2]=.25f;pixel[3]=original?.5f+float(x)/64:.25f;
            }upload->Unmap(0,nullptr);
            D3D12_TEXTURE_COPY_LOCATION from{upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,footprint},to{resource,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            phoenix_native::transition(command.Get(),resource,D3D12_RESOURCE_STATE_COPY_DEST,original?D3D12_RESOURCE_STATE_COMMON:D3D12_RESOURCE_STATE_COPY_SOURCE);
            uploads.push_back(std::move(upload));
        };
        initialize(low.Get(),false);initialize(game.Get(),true);
        auto geometry=phoenix_addon::phoenix::fit_resolution(output_width,output_height,width,height);
        phoenix_addon::phoenix::Processing processing;processing.enabled=true;processing.encoding=2;
        spatial::ResolutionBridge bridge(device.Get(),geometry,DXGI_FORMAT_R32G32B32A32_FLOAT,{},processing);
        phoenix_pipeline::Options options;options.chain_enabled=true;options.scaler=tiny?3:1;
        bridge.additional_scaler(options,tiny?std::filesystem::absolute(argv[1]):std::filesystem::path{});
        // Three asynchronous recordings need independent descriptor/constants
        // while sharing the serialized workspaces and immutable coefficients.
        for(unsigned frame=1;frame<=3;++frame) {
            bridge.begin_frame(0,frame);
            require(bridge.present_additional(command.Get(),low.Get(),game.Get(),D3D12_RESOURCE_STATE_COMMON),"Additional scaler was bypassed");
        }
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};uint64_t bytes=0;const auto description=game->GetDesc();
        device->GetCopyableFootprints(&description,0,1,0,&footprint,nullptr,nullptr,&bytes);
        auto readback=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        phoenix_native::transition(command.Get(),game.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{game.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},to{readback.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,footprint};
        command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        hr(command->Close());ID3D12CommandList* lists[]={command.Get()};queue->ExecuteCommandLists(1,lists);
        ComPtr<ID3D12Fence> fence;hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));hr(queue->Signal(fence.Get(),1));
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"Fence event");hr(fence->SetEventOnCompletion(1,event));const auto waited=WaitForSingleObject(event,30000);CloseHandle(event);
        require(waited==WAIT_OBJECT_0,"Scaler bridge timed out");hr(device->GetDeviceRemovedReason());
        void* mapped=nullptr;D3D12_RANGE range{0,size_t(bytes)},empty{};hr(readback->Map(0,&range,&mapped));
        float maximum_alpha_error=0;
        for(unsigned y=0;y<output_height;++y)for(unsigned x=0;x<output_width;++x) {
            const auto* pixel=reinterpret_cast<const float*>(static_cast<const uint8_t*>(mapped)+footprint.Footprint.RowPitch*y)+x*4;
            maximum_alpha_error=std::max(maximum_alpha_error,std::abs(pixel[3]-(.5f+float(x)/64)));
            for(unsigned c=0;c<3;++c)require(std::isfinite(pixel[c]),"Nonfinite reconstruction output");
            if(!tiny){require(std::abs(pixel[0]-.5f)<.002f&&std::abs(pixel[1]-.5f)<.002f,"NIS constant color changed");}
        }readback->Unmap(0,&empty);require(maximum_alpha_error==0,"Original alpha not preserved");
        ComPtr<ID3D12InfoQueue> messages;hr(device.As(&messages));
        for(UINT64 i=0;i<messages->GetNumStoredMessages();++i) {
            SIZE_T size=0;hr(messages->GetMessage(i,nullptr,&size));std::vector<uint8_t> storage(size);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());hr(messages->GetMessage(i,message,&size));
            if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR)throw std::runtime_error(message->pDescription);
        }
        std::cout<<Json{{"passed",true},{"provider",tiny?"tiny":"nis"},{"asynchronous_recordings",3},
            {"alpha_exact",true},{"debug_errors",0},{"status",bridge.status()}}.dump()<<'\n';
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
