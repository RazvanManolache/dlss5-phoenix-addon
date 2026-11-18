// Generic native driver executor for an x86 Phoenix host. Deliberately contains
// no Phoenix graph/model loader, neural controls, plan generation or renderer.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <vector>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <source_location>
#include "../../../../third_party/nvapi/nvapi.h"
#include "driver_contract.h"
#include "umd_bridge.h"
#include "rgba8_gpu_transport.h"

using Microsoft::WRL::ComPtr;
namespace bridge=phoenix_driver_probe;
extern "C" __declspec(dllexport) void PhoenixSetGdiRpc64(PhoenixGdiRpc* rpc) { wow64_umd_probe::gdi=rpc; }
static void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
static void hr(HRESULT result,std::source_location where=std::source_location::current()) {
    if(FAILED(result)) throw std::runtime_error("Driver D3D12 failure "+std::to_string(uint32_t(result))+" at line "+std::to_string(where.line()));
}
static void nv(NvAPI_Status result) { require(result==NVAPI_OK,"Driver thunk NVAPI failure"); }
template<class T> static T function(HMODULE library,const char* name) {
    auto result=reinterpret_cast<T>(GetProcAddress(library,name)); require(result!=nullptr,name); return result;
}
struct Driver {
    bridge::Channel* channel=nullptr;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> views,samplers,clear_cpu;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> workspace,weights,color,motion,output[2],timestamps,upload,output_seed,gpu_seed,readback,aux;
    HANDLE completion_event=nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint64_t plane_bytes=0,serial=0,frequency=0;
    unsigned width=0,height=0,clear_words=0,caller_bits=0;
    bool input_uploaded=false;
    bool packed_negotiated=false,packed_next=false;
    std::unique_ptr<phoenix_wow64::Rgba8GpuTransport> packed_pixels;
    D3D12_GPU_DESCRIPTOR_HANDLE clear_gpu{};
    std::array<NVDX_ObjectHandle,41> modules{},functions{};
    decltype(&NvAPI_D3D12_CreateCuModule) create_module=nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) create_function=nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroy_module=nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroy_function=nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) dispatch=nullptr;
    decltype(&NvAPI_D3D12_GetCudaMergedTextureSamplerObject) texture=nullptr;
    decltype(&NvAPI_D3D12_GetCudaSurfaceObject) surface=nullptr;
    ~Driver() {
        if(completion_event)CloseHandle(completion_event);
        if(!device) return;
        for(unsigned i=0;i<41;++i) {
            if(functions[i]&&destroy_function) {
                if(destroy_function(device.Get(),functions[i])==NVAPI_OK) ++channel->functions_destroyed; else ++channel->cleanup_failures;
            }
            if(modules[i]&&destroy_module) {
                if(destroy_module(device.Get(),modules[i])==NVAPI_OK) ++channel->modules_destroyed; else ++channel->cleanup_failures;
            }
        }
    }
    void transition(ID3D12Resource* value,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={value,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; list->ResourceBarrier(1,&b);
    }
    void barrier() {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; list->ResourceBarrier(1,&b);
    }
    void submit() {
        // The model-free driver probe also submits through this path.
        if(!completion_event) {
            completion_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
            require(completion_event!=nullptr,"Driver fence event");
        }
        hr(list->Close()); ID3D12CommandList* lists[]{list.Get()}; queue->ExecuteCommandLists(1,lists);
        hr(queue->Signal(fence.Get(),++serial));
        hr(fence->SetEventOnCompletion(serial,completion_event));
        require(WaitForSingleObject(completion_event,10000)==WAIT_OBJECT_0,"Driver GPU fence timeout");
        hr(device->GetDeviceRemovedReason()); hr(allocator->Reset()); hr(list->Reset(allocator.Get(),nullptr));
    }
    ComPtr<ID3D12Resource> buffer(uint64_t bytes,D3D12_HEAP_TYPE heap,D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES properties{}; properties.Type=heap;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=bytes;
        desc.Height=1; desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(heap==D3D12_HEAP_TYPE_DEFAULT) desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result;
        const auto allocated=device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(&result));
        if(FAILED(allocated)) throw std::runtime_error("Driver D3D12 buffer allocation failed: "+std::to_string(bytes)+
            " bytes, heap type "+std::to_string(unsigned(heap))+", HRESULT "+std::to_string(uint32_t(allocated)));
        return result;
    }
    ComPtr<ID3D12Resource> image(unsigned width,unsigned height,DXGI_FORMAT format) {
        D3D12_HEAP_PROPERTIES properties{}; properties.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=width; desc.Height=height;
        desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1; desc.Format=format;
        desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result;
        hr(device->CreateCommittedResource(&properties,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            nullptr,IID_PPV_ARGS(&result))); return result;
    }
    uint64_t texture_handle(ID3D12Resource* image,unsigned slot,bool linear) {
        auto view=views->GetCPUDescriptorHandleForHeapStart();
        view.ptr+=slot*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto sampler=samplers->GetCPUDescriptorHandleForHeapStart();
        sampler.ptr+=slot*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format=image->GetDesc().Format;
        srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels=1; device->CreateShaderResourceView(image,&srv,view);
        D3D12_SAMPLER_DESC smp{}; smp.Filter=linear?D3D12_FILTER_MIN_MAG_MIP_LINEAR:D3D12_FILTER_MIN_MAG_MIP_POINT;
        smp.AddressU=smp.AddressV=smp.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        smp.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS; smp.MaxLOD=D3D12_FLOAT32_MAX; smp.MaxAnisotropy=1;
        device->CreateSampler(&smp,sampler);
        NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS params{};
        params.structSizeIn=params.structSizeOut=sizeof(params); params.pDevice=device.Get(); params.texDesc=view; params.smpDesc=sampler;
        nv(texture(&params)); return params.textureHandle;
    }
    void initialize(bridge::Channel& c) {
        require(bridge::valid_initialize(c),"Invalid driver init packet");
        packed_negotiated=(c.frame_flags&bridge::packed_rgba8)!=0;
        channel=&c;
        width=c.width; height=c.height; clear_words=c.clear_words; caller_bits=c.caller_bits;
        hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for(unsigned i=0;;++i) {
            hr(factory->EnumAdapters1(i,&adapter)); DXGI_ADAPTER_DESC1 desc{}; hr(adapter->GetDesc1(&desc));
            if(desc.VendorId==0x10de&&!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter.Reset();
        }
        hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC desc{}; hr(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)));
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        hr(queue->GetTimestampFrequency(&frequency)); require(frequency>0,"Driver timestamp frequency");
        D3D12_QUERY_HEAP_DESC q{}; q.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; q.Count=2;
        hr(device->CreateQueryHeap(&q,IID_PPV_ARGS(&queries)));
        timestamps=buffer(16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        auto library=LoadLibraryExW(L"nvapi64.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(library!=nullptr,"Native NVAPI load"); if(wow64_umd_probe::gdi) wow64_umd_probe::patch(library);
        const auto query=function<void*(__cdecl*)(unsigned)>(library,"nvapi_QueryInterface");
        const auto initialize=reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150e828)); require(initialize!=nullptr,"Native NVAPI init entry");
        if(wow64_umd_probe::gdi) wow64_umd_probe::patch(GetModuleHandleW(L"nvapi64_impl.dll")); nv(initialize());
#define PHX_DRIVER_ENTRY(member,id) member=reinterpret_cast<decltype(member)>(query(id)); require(member!=nullptr,#member)
        PHX_DRIVER_ENTRY(create_module,0xad1a677d); PHX_DRIVER_ENTRY(create_function,0xe2436e22);
        PHX_DRIVER_ENTRY(destroy_module,0x41c65285); PHX_DRIVER_ENTRY(destroy_function,0xdf295ea6);
        PHX_DRIVER_ENTRY(dispatch,0x24973538); PHX_DRIVER_ENTRY(texture,0x329fe6e0); PHX_DRIVER_ENTRY(surface,0x48f5b2ee);
#undef PHX_DRIVER_ENTRY
        for(unsigned i=0;i<41;++i) {
            const auto& k=c.kernels[i];
            require(bridge::valid_host_range(k.bytes,k.size,caller_bits)&&k.size<=16*1024*1024&&
                bridge::valid_host_range(k.entry,1,caller_bits),"Invalid kernel packet");
            nv(create_module(device.Get(),reinterpret_cast<const void*>(k.bytes),NvU32(k.size),&modules[i]));
            nv(create_function(device.Get(),modules[i],reinterpret_cast<const char*>(k.entry),&functions[i]));
        }
        workspace=buffer(c.workspace_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        weights=buffer(c.weight_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
        // Keep the temporary mapped upload small in a fragmented x86 process.
        // Each fence retires the copy before this same staging storage is reused.
        constexpr uint64_t upload_chunk_bytes=4ull*1024*1024;
        const auto staging_bytes=std::min(c.weight_bytes,upload_chunk_bytes);
        auto staging=buffer(staging_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        for(uint64_t offset=0;offset<c.weight_bytes;) {
            const auto bytes=std::min(staging_bytes,c.weight_bytes-offset);
            void* data=nullptr; D3D12_RANGE empty{}; hr(staging->Map(0,&empty,&data));
            memcpy(data,reinterpret_cast<const uint8_t*>(c.weights)+offset,size_t(bytes)); staging->Unmap(0,nullptr);
            list->CopyBufferRegion(weights.Get(),offset,staging.Get(),0,bytes);
            offset+=bytes;
            if(offset==c.weight_bytes)
                transition(weights.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            submit();
        }
        color=image(c.width,c.height,DXGI_FORMAT_R16G16B16A16_FLOAT); motion=image(c.width,c.height,DXGI_FORMAT_R16G16_FLOAT);
        for(auto& value:output) value=image(c.width,c.height,DXGI_FORMAT_R16G16B16A16_FLOAT);
        aux=image(c.width,c.height,DXGI_FORMAT_R16G16B16A16_FLOAT);
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,9,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        hr(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&views)));
        heap.NumDescriptors=2; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_NONE; hr(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&clear_cpu)));
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.Format=DXGI_FORMAT_R32_UINT; uav.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements=c.clear_words; auto handle=views->GetCPUDescriptorHandleForHeapStart();
        handle.ptr+=6*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateUnorderedAccessView(workspace.Get(),nullptr,&uav,handle);
        device->CreateUnorderedAccessView(workspace.Get(),nullptr,&uav,clear_cpu->GetCPUDescriptorHandleForHeapStart());
        clear_gpu=views->GetGPUDescriptorHandleForHeapStart();
        clear_gpu.ptr+=6*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER; heap.NumDescriptors=8; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&samplers)));
        c.color_texture=texture_handle(color.Get(),0,false); c.motion_texture=texture_handle(motion.Get(),1,false);
        c.aux_texture=texture_handle(aux.Get(),7,false);
        for(unsigned i=0;i<2;++i) {
            c.history_texture[i]=texture_handle(output[i].Get(),2+i,true);
            auto view=views->GetCPUDescriptorHandleForHeapStart();
            view.ptr+=(4+i)*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{}; desc.Format=DXGI_FORMAT_R16G16B16A16_FLOAT; desc.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(output[i].Get(),nullptr,&desc,view);
            NvU32 value=0; nv(surface(device.Get(),view,&value)); c.surface[i]=value;
        }
        c.workspace_address=workspace->GetGPUVirtualAddress(); c.weights_address=weights->GetGPUVirtualAddress();
        const auto color_desc=color->GetDesc(); device->GetCopyableFootprints(&color_desc,0,1,0,&footprint,nullptr,nullptr,&plane_bytes);
    }
    void prepare_gpu_seed() {
        if(gpu_seed)return;
        gpu_seed=buffer(plane_bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const auto stride=device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const D3D12_CPU_DESCRIPTOR_HANDLE shader_view{views->GetCPUDescriptorHandleForHeapStart().ptr+8*stride};
        const D3D12_CPU_DESCRIPTOR_HANDLE cpu_view{clear_cpu->GetCPUDescriptorHandleForHeapStart().ptr+stride};
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu_view{views->GetGPUDescriptorHandleForHeapStart().ptr+8*stride};
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{}; desc.Format=DXGI_FORMAT_R32_UINT;
        desc.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; desc.Buffer.NumElements=UINT(plane_bytes/4);
        device->CreateUnorderedAccessView(gpu_seed.Get(),nullptr,&desc,shader_view);
        device->CreateUnorderedAccessView(gpu_seed.Get(),nullptr,&desc,cpu_view);
        ID3D12DescriptorHeap* heaps[]{views.Get()}; list->SetDescriptorHeaps(1,heaps);
        constexpr UINT poison=0x7e007e00; // Two RGBA16F NaNs per word, including row padding.
        constexpr UINT words[]{poison,poison,poison,poison};
        list->ClearUnorderedAccessViewUint(gpu_view,cpu_view,gpu_seed.Get(),words,0,nullptr);
        transition(gpu_seed.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    void upload_half(const bridge::Channel& c) {
        if(!upload)upload=buffer(plane_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* data=nullptr; D3D12_RANGE empty{}; hr(upload->Map(0,&empty,&data));
        for(unsigned y=0;y<c.height;++y) memcpy(static_cast<char*>(data)+footprint.Offset+y*footprint.Footprint.RowPitch,
            reinterpret_cast<const char*>(c.color)+size_t(y)*c.width*8,size_t(c.width)*8);
        upload->Unmap(0,nullptr);
        transition(color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION destination{color.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION source{upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        source.PlacedFootprint=footprint;
        list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition(color.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        input_uploaded=true;
    }
    void seed_output(const bridge::Channel& c,ID3D12Resource* target) {
        const bool resident=(c.frame_flags&bridge::gpu_output_seed)!=0;
        if(resident)prepare_gpu_seed();
        else {
            if(!output_seed)output_seed=buffer(plane_bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void* data=nullptr; D3D12_RANGE empty{}; hr(output_seed->Map(0,&empty,&data));
            for(unsigned y=0;y<c.height;++y) memcpy(static_cast<char*>(data)+footprint.Offset+y*footprint.Footprint.RowPitch,
                reinterpret_cast<const char*>(c.output)+size_t(y)*c.width*8,size_t(c.width)*8);
            output_seed->Unmap(0,nullptr);
        }
        transition(target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION destination{target,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION source{resident?gpu_seed.Get():output_seed.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        source.PlacedFootprint=footprint;
        list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
        transition(target,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    void finish_packed(unsigned parity,bool read_output) {
        auto* target=output[parity].Get();
        transition(target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
        transition(color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        if(read_output)packed_pixels->record_output(list.Get(),parity);
        else packed_pixels->record_refinement(list.Get(),parity);
        transition(target,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(color.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    void dispatch_graph(bridge::Channel& c,const bridge::Launch* launches,unsigned count,bool timestamp) {
        if(timestamp)list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        ID3D12DescriptorHeap* heaps[]{views.Get(),samplers.Get()}; list->SetDescriptorHeaps(2,heaps);
        const UINT sentinel[4]{UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX}; barrier();
        list->ClearUnorderedAccessViewUint(clear_gpu,clear_cpu->GetCPUDescriptorHandleForHeapStart(),workspace.Get(),sentinel,0,nullptr); barrier();
        for(unsigned i=0;i<count;++i) {
            const auto& launch=launches[i];
            NVAPI_CU_KERNEL_LAUNCH_PARAMS packet{}; packet.hFunction=functions[launch.kernel]; packet.paramSize=launch.bytes;
            packet.gridDim={launch.grid[0],launch.grid[1],launch.grid[2]}; packet.blockDim={launch.block[0],launch.block[1],launch.block[2]};
            packet.pParams=const_cast<uint8_t*>(launch.arguments); nv(dispatch(list.Get(),&packet,1));
            if(launch.barrier_after) barrier(); ++c.dispatched_kernels;
        }
        if(timestamp) {
            list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
            list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,timestamps.Get(),0);
        }
    }
    void frame(bridge::Channel& c) {
        require(c.target<2&&c.width==width&&c.height==height&&c.clear_words==clear_words&&c.caller_bits==caller_bits&&
            bridge::valid_frame(c),"Invalid or changed frame resources");
        for(unsigned i=0;i<c.launch_total;++i) require(bridge::valid_launch(c.launches[i]),"Invalid dispatch packet");
        const bool packed=(c.frame_flags&bridge::packed_rgba8)!=0;
        const bool continuing=(c.frame_flags&bridge::reuse_refinement)!=0;
        const bool read_output=!(c.frame_flags&bridge::skip_readback);
        const unsigned batch=(c.frame_flags&bridge::packed_pass_mask)>>bridge::packed_pass_shift;
        const unsigned passes=batch?batch:1;
        const auto* alternate=batch?reinterpret_cast<const bridge::Launch*>(c.alternate_launches):nullptr;
        if(alternate)
            for(unsigned i=0;i<c.launch_total;++i)require(bridge::valid_launch(alternate[i]),"Invalid alternate dispatch packet");
        require(!packed||packed_negotiated,"Packed driver transport was not negotiated");
        require(!continuing||packed_next,"No resident refinement input for this pass");
        packed_next=false;
        if(packed&&!continuing) {
            if(!packed_pixels)packed_pixels=std::make_unique<phoenix_wow64::Rgba8GpuTransport>(
                device.Get(),width,height,color.Get(),output[0].Get(),output[1].Get());
            transition(color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
            packed_pixels->record_input(list.Get(),reinterpret_cast<const void*>(c.color),!read_output||batch!=0);
            transition(color.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        } else if(!packed) {
            if(!(c.frame_flags&1))upload_half(c);
            else require(input_uploaded,"No resident input uploaded");
        }

        for(unsigned pass=0;pass<passes;++pass) {
            const unsigned parity=c.target^(pass&1);
            auto* target=output[parity].Get();
            if(packed||read_output)seed_output(c,target);
            else transition(target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const bool last=pass+1==passes;
            dispatch_graph(c,pass&1?alternate:c.launches,c.launch_total,last);
            if(packed)finish_packed(parity,read_output&&last);
        }
        auto* target=output[c.target].Get();
        if(!packed&&read_output) {
            if(!readback)readback=buffer(plane_bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            transition(target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION destination{readback.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            destination.PlacedFootprint=footprint;
            D3D12_TEXTURE_COPY_LOCATION source{target,D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
            transition(target,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        } else if(!packed)transition(target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        submit();
        uint64_t* times=nullptr; D3D12_RANGE range{0,16},empty{};
        hr(timestamps->Map(0,&range,reinterpret_cast<void**>(&times)));
        const auto start=times[0], end=times[1];
        timestamps->Unmap(0,&empty);
        require(end>start,"Driver timestamp order"); c.gpu_ms=double(end-start)*1000.0/frequency;
        if(packed) {
            if(read_output)packed_pixels->read_output(reinterpret_cast<void*>(c.output));
            else packed_next=true;
        } else if(read_output) {
            void* data=nullptr;
            D3D12_RANGE all{0,size_t(plane_bytes)}; hr(readback->Map(0,&all,&data));
            for(unsigned y=0;y<c.height;++y) memcpy(reinterpret_cast<char*>(c.output)+size_t(y)*c.width*8,
                static_cast<const char*>(data)+footprint.Offset+y*footprint.Footprint.RowPitch,size_t(c.width)*8);
            readback->Unmap(0,&empty);
        }
        c.completed_frames+=passes;
    }
};
#include "driver_check64.h"
extern "C" __declspec(dllexport) int PhoenixDriverServe64(bridge::Channel* packet) {
    if(!packet||(!bridge::valid_initialize(*packet)&&!bridge::valid_compatibility_check(*packet))) return 2;
    auto& c=*packet; bool passed=false;
    try {
        c.driver_pid=GetCurrentProcessId(); c.driver_bits=64;
        c.phoenix64_loaded=GetModuleHandleW(L"phoenix_native.dll")?1:0;
        if(wow64_umd_probe::gdi) {
            const DWORD offsets[]{c.appcompat_offsets[0],c.appcompat_offsets[1],c.appcompat_offsets[2],c.appcompat_offsets[3]};
            require(wow64_umd_probe::install_native_compat_view(c.ntdll_timestamp,offsets),"Native compatibility view unavailable");
            wow64_runtime_diagnostics::retry_native_reservation=true;
            wow64_umd_probe::bridge_dc_creation=true; wow64_umd_probe::bridge_display_gdi=true; wow64_umd_probe::bridge_kmt_escape=true;
            require(wow64_umd_probe::install()!=0,"Native driver import adaptation unavailable");
            require(wow64_umd_probe::install_gdi_export(),"Native driver DeleteDC adaptation unavailable");
            require(wow64_umd_probe::install_display_exports(),"Native driver display adaptation unavailable");
        }
        auto cuda=LoadLibraryW(reinterpret_cast<const wchar_t*>(c.cuda_library)); require(cuda!=nullptr,"Native CUDA load");
        if(wow64_umd_probe::gdi) { wow64_umd_probe::owner=GetCurrentThreadId(); wow64_umd_probe::patch(cuda); }
        require(function<int(*)(unsigned)>(cuda,"cuInit")(0)==0,"Native CUDA initialization refused");
        {
            Driver driver;
            const bool compatibility_only=c.operation==bridge::check_compatibility;
            if(compatibility_only)check_installed_driver(cuda,c);else driver.initialize(c);
            SetEvent(reinterpret_cast<HANDLE>(c.response_event));
            for(;;) {
                require(WaitForSingleObject(reinterpret_cast<HANDLE>(c.request_event),INFINITE)==WAIT_OBJECT_0,"Driver request wait failed");
                if(c.operation==bridge::shutdown) break;
                require(!compatibility_only&&c.operation==bridge::frame,"Unknown driver operation"); driver.frame(c);
                ++c.response; SetEvent(reinterpret_cast<HANDLE>(c.response_event));
            }
        }
        passed=true;
    } catch(const std::exception& error) {
        c.status=-1; strncpy_s(c.error,error.what(),_TRUNCATE);
    }
    if(wow64_umd_probe::gdi) {
        wow64_umd_probe::restore(); wow64_umd_probe::restore_gdi_export(); wow64_umd_probe::restore_display_exports();
        c.native_compat_restored=wow64_umd_probe::restore_native_compat_view()?1:0;
    } else c.native_compat_restored=1;
    if(!c.native_compat_restored||c.cleanup_failures) { passed=false; c.status=-1; }
    ++c.response; SetEvent(reinterpret_cast<HANDLE>(c.response_event)); return passed?0:2;
}
