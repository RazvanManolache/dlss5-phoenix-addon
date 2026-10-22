#include "../../src/pipeline/d3d12_motion.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <iostream>
using Microsoft::WRL::ComPtr;
using phoenix_native::hr;
using phoenix_bundle::require;
using Json=phoenix_bundle::Json;
int main(){
    try{
        ComPtr<ID3D12Debug> debug;hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        for(unsigned i=0;;++i){ComPtr<IDXGIAdapter1> candidate;
            require(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&candidate))!=DXGI_ERROR_NOT_FOUND,"No NVIDIA test adapter");
            DXGI_ADAPTER_DESC1 desc{};hr(candidate->GetDesc1(&desc));if(desc.VendorId==0x10de){adapter=candidate;break;}}
        ComPtr<ID3D12Device> device;hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12CommandQueue> queue;D3D12_COMMAND_QUEUE_DESC q{};hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12Fence> fence;hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));uint64_t serial=0;
        ComPtr<ID3D12CommandAllocator> allocator;hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> command;hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&command)));hr(command->Close());
        auto begin=[&]{hr(allocator->Reset());hr(command->Reset(allocator.Get(),nullptr));};
        auto submit=[&]{hr(command->Close());ID3D12CommandList* lists[]={command.Get()};queue->ExecuteCommandLists(1,lists);hr(queue->Signal(fence.Get(),++serial));
            HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"Motion test event");hr(fence->SetEventOnCompletion(serial,event));
            const auto waited=WaitForSingleObject(event,30000);CloseHandle(event);require(waited==WAIT_OBJECT_0,"Motion test timeout");};
        constexpr unsigned w=320,h=180;constexpr int dx=8,dy=4;
        auto source=phoenix_native::image(device.Get(),w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COMMON);
        const auto shape=source->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT upload_fp{};uint64_t upload_bytes=0;
        device->GetCopyableFootprints(&shape,0,1,0,&upload_fp,nullptr,nullptr,&upload_bytes);
        auto upload=phoenix_native::buffer(device.Get(),upload_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto read=[&](ID3D12Resource* image){
            const auto desc=image->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};uint64_t bytes=0;device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
            auto buffer=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            begin();phoenix_native::transition(command.Get(),image,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{image,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},to{buffer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,fp};
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);phoenix_native::transition(command.Get(),image,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);submit();
            std::vector<uint8_t> result(size_t(w)*h*(desc.Format==DXGI_FORMAT_R8_UNORM?1:4));
            void* mapped=nullptr;D3D12_RANGE range{0,size_t(bytes)},empty{};hr(buffer->Map(0,&range,&mapped));
            for(unsigned y=0;y<h;++y)memcpy(result.data()+size_t(y)*result.size()/h,static_cast<uint8_t*>(mapped)+fp.Offset+y*fp.Footprint.RowPitch,result.size()/h);
            buffer->Unmap(0,&empty);return result;
        };
        Json cases=Json::array();
        for(int analysis_height:{180,90}){
            phoenix_pipeline::D3D12Motion provider(device.Get(),queue.Get(),fence.Get());
            phoenix_pipeline::Options options;options.motion_enabled=true;options.motion_source=2;options.flow_height=analysis_height;
            for(unsigned frame=0;frame<3;++frame){
                void* mapped=nullptr;D3D12_RANGE empty{};hr(upload->Map(0,&empty,&mapped));
                for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                    auto* pixel=reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(mapped)+upload_fp.Footprint.RowPitch*y)+x*4;
                    // A known rigid translation of a nonperiodic textured plane.
                    const uint32_t sx=uint32_t(int(x)-int(frame)*dx+1024),sy=uint32_t(int(y)-int(frame)*dy+1024);
                    uint32_t hash=(sx/3)*0x9e3779b9u+(sy/3)*0x85ebca6bu;hash^=hash>>16;hash*=0x7feb352du;hash^=hash>>15;
                    for(unsigned c=0;c<3;++c)pixel[c]=DirectX::PackedVector::XMConvertFloatToHalf(.1f+float((hash>>(c*8))&255)/320.f);pixel[3]=0x3c00;
                }upload->Unmap(0,nullptr);begin();
                phoenix_native::transition(command.Get(),source.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION from{upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,upload_fp},to{source.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
                command->CopyTextureRegion(&to,0,0,0,&from,nullptr);phoenix_native::transition(command.Get(),source.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);submit();
                const auto estimated=provider.prepare(source.Get(),D3D12_RESOURCE_STATE_COMMON,nullptr,options,frame==0,int64_t(frame)*16666667,1,frame);
                if(frame==0){require(estimated.reset&&estimated.motion==nullptr,"First frame did not prime history");continue;}
                require(estimated.source==phoenix_pipeline::MotionSource::Nvidia&&estimated.motion&&estimated.rejection,"Hardware optical flow unavailable");
                const auto vectors=read(provider.raw_motion()),gated=read(estimated.motion),rejection=read(estimated.rejection);provider.consumed(serial);
                std::vector<float> errors;double mean_x=0,mean_y=0,trust=0;unsigned outside=0;
                for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                    uint16_t bits[2];memcpy(bits,vectors.data()+(size_t(y)*w+x)*4,4);
                    const float vx=DirectX::PackedVector::XMConvertHalfToFloat(bits[0]),vy=DirectX::PackedVector::XMConvertHalfToFloat(bits[1]);
                    uint16_t gate_bits[2];memcpy(gate_bits,gated.data()+(size_t(y)*w+x)*4,4);
                    const float gx=DirectX::PackedVector::XMConvertHalfToFloat(gate_bits[0]),gy=DirectX::PackedVector::XMConvertHalfToFloat(gate_bits[1]);
                    if(rejection[size_t(y)*w+x]>=128)require(float(x)+gx>=w&&float(y)+gy>=h,"Rejected history still reaches neural reprojection");
                    else require(bits[0]==gate_bits[0]&&bits[1]==gate_bits[1],"Trusted motion changed at the history gate");
                    require(std::isfinite(vx)&&std::isfinite(vy),"Nonfinite optical flow");
                    if(float(x)+vx<0||float(y)+vy<0||float(x)+vx>=w||float(y)+vy>=h){require(rejection[size_t(y)*w+x]==255,"Out-of-image history was trusted");++outside;}
                    if(x>=32&&y>=32&&x<w-32&&y<h-32){errors.push_back(std::hypot(vx+dx,vy+dy));mean_x+=vx;mean_y+=vy;trust+=1.-rejection[size_t(y)*w+x]/255.;}
                }
                std::sort(errors.begin(),errors.end());const float median=errors[errors.size()/2],p95=errors[errors.size()*95/100];
                require(median<1.f&&p95<3.f,"Optical flow direction/scale does not match the known translation");
                require(outside>100,"No disoccluded border was identified");
                cases.push_back({{"analysis_height",analysis_height},{"frame",frame},{"expected_motion",{-dx,-dy}},
                    {"mean_motion",{mean_x/errors.size(),mean_y/errors.size()}},{"endpoint_error_median",median},
                    {"endpoint_error_p95",p95},{"interior_mean_confidence",trust/errors.size()},{"outside_rejected_pixels",outside},{"neural_history_rejection_verified",true}});
            }
        }
        ComPtr<ID3D12InfoQueue> messages;hr(device.As(&messages));
        for(UINT64 i=0;i<messages->GetNumStoredMessages();++i){SIZE_T bytes=0;hr(messages->GetMessage(i,nullptr,&bytes));std::vector<uint8_t> storage(bytes);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());hr(messages->GetMessage(i,message,&bytes));
            require(message->Severity>D3D12_MESSAGE_SEVERITY_ERROR,message->pDescription);}
        std::cout<<Json{{"passed",true},{"cases",cases},{"debug_errors",0},{"depth_supplied",false}}.dump()<<'\n';
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
