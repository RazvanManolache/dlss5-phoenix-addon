// Native same-process inference for APIs that need the existing fenced CPU
// capture/return path. This is deliberately distinct from GPU-only DX11 interop.
#pragma once
#include "d3d12_bridge.h"
#include "local_d3d12_identity.h"
#include <dxgi1_6.h>

namespace phoenix_compat {
class D3D12LocalPixels {
    template<class T> using Com = Microsoft::WRL::ComPtr<T>;
    struct Resources {
        Com<ID3D12Device> wrapper_device;
        Com<ID3D12CommandQueue> wrapper_queue;
        Com<ID3D12Device> device;
        Com<ID3D12CommandQueue> queue;
        Com<ID3D12CommandAllocator> allocator;
        Com<ID3D12GraphicsCommandList> list;
        Com<ID3D12Resource> color, upload, readback;
        Com<ID3D12Fence> fence;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        uint64_t bytes = 0, serial = 0;
        HANDLE event = nullptr;
        bool retired = true;
        bool queue_registered = false;
        ~Resources() {
            // The bridge is destroyed first, and every private submission has
            // retired before this owner releases the exact queue identity.
            if (queue_registered && in_process_api.local_queue_lifetime)
                in_process_api.local_queue_lifetime(queue.Get(),false);
            if (event) CloseHandle(event);
        }
    };
    std::unique_ptr<Resources> resources_ = std::make_unique<Resources>();
    std::unique_ptr<D3D12Bridge> bridge_;
    unsigned width_, height_;
    static void checked(HRESULT result) {
        if (FAILED(result)) throw std::runtime_error("Local native staging HRESULT " + std::to_string(uint32_t(result)));
    }
    void transfer(bool upload) {
        auto& r=*resources_;
        checked(r.allocator->Reset()); checked(r.list->Reset(r.allocator.Get(),nullptr));
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={r.color.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,
            upload?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_COPY_SOURCE};
        r.list->ResourceBarrier(1,&barrier);
        D3D12_TEXTURE_COPY_LOCATION image{}; image.pResource=r.color.Get(); image.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION buffer{}; buffer.pResource=upload?r.upload.Get():r.readback.Get();
        buffer.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; buffer.PlacedFootprint=r.footprint;
        if(upload)r.list->CopyTextureRegion(&image,0,0,0,&buffer,nullptr);
        else r.list->CopyTextureRegion(&buffer,0,0,0,&image,nullptr);
        std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);r.list->ResourceBarrier(1,&barrier);
        checked(r.list->Close()); ID3D12CommandList* lists[]{r.list.Get()};
        r.retired=false; r.queue->ExecuteCommandLists(1,lists);
        checked(r.queue->Signal(r.fence.Get(),++r.serial));
        checked(r.fence->SetEventOnCompletion(r.serial,r.event));
        if(WaitForSingleObject(r.event,5000)!=WAIT_OBJECT_0||r.fence->GetCompletedValue()==UINT64_MAX||
            r.fence->GetCompletedValue()<r.serial) {
            in_process_gpu_uncertain=true;
            throw std::runtime_error("Local native staging did not retire; helper fallback is unsafe");
        }
        r.retired=true;
    }
public:
    D3D12LocalPixels(reshade::api::effect_runtime* runtime,unsigned width,unsigned height,
                     ID3D12Device* software_test_device=nullptr) : width_(width),height_(height) {
        if (!width || !height || width > 8192 || height > 8192)
            throw std::runtime_error("Local native staging extent is outside the supported range");
        auto& r=*resources_;
        if(software_test_device)r.device=software_test_device;
        else {
            // A game-local DXGI proxy can enumerate Vulkan-backed adapters
            // that the real D3D12 runtime cannot open. The staged device only
            // exchanges pixels, so select an adapter through Windows DXGI.
            wchar_t system[MAX_PATH]{};
            const auto length = GetSystemDirectoryW(system, MAX_PATH);
            if (!length || length >= MAX_PATH) throw std::runtime_error("Windows system directory is unavailable");
            const auto path = std::filesystem::path(system) / L"dxgi.dll";
            const auto system_dxgi = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!system_dxgi) throw std::runtime_error("System DXGI is unavailable for local native inference");
            struct ReleaseModule { HMODULE value; ~ReleaseModule() { FreeLibrary(value); } } dxgi{system_dxgi};
            const auto create_factory = reinterpret_cast<decltype(&CreateDXGIFactory1)>(
                GetProcAddress(system_dxgi, "CreateDXGIFactory1"));
            if (!create_factory) throw std::runtime_error("System DXGI does not expose CreateDXGIFactory1");
            Com<IDXGIFactory6> factory;
            checked(create_factory(IID_PPV_ARGS(&factory)));
            for(UINT index=0;;++index) {
                Com<IDXGIAdapter1> adapter;
                const auto found = factory->EnumAdapterByGpuPreference(index,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    IID_PPV_ARGS(&adapter));
                if(found==DXGI_ERROR_NOT_FOUND)break;
                checked(found);
                DXGI_ADAPTER_DESC1 info{};checked(adapter->GetDesc1(&info));
                if(info.VendorId!=0x10de||(info.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))continue;
                if(SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&r.device))))break;
            }
            if(!r.device)throw std::runtime_error("No eligible NVIDIA adapter for local native inference");
        }
        D3D12_COMMAND_QUEUE_DESC queue{};checked(r.device->CreateCommandQueue(&queue,IID_PPV_ARGS(&r.queue)));
        resolve_local_d3d12_identity(r.device,r.queue,r.wrapper_device,r.wrapper_queue);
        if (in_process_api.local_queue_lifetime) {
            in_process_api.local_queue_lifetime(r.queue.Get(),true);
            r.queue_registered = true;
        }
        checked(r.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&r.allocator)));
        checked(r.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,r.allocator.Get(),nullptr,IID_PPV_ARGS(&r.list)));
        checked(r.list->Close());checked(r.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&r.fence)));
        r.event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!r.event)throw std::runtime_error("No local staging event");
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC color{};color.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        color.Width=width;color.Height=height;color.DepthOrArraySize=color.MipLevels=color.SampleDesc.Count=1;
        color.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
        checked(r.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&color,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r.color)));
        r.device->GetCopyableFootprints(&color,0,1,0,&r.footprint,nullptr,nullptr,&r.bytes);
        D3D12_RESOURCE_DESC buffer{};buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;buffer.Width=r.bytes;
        buffer.Height=buffer.DepthOrArraySize=buffer.MipLevels=buffer.SampleDesc.Count=1;buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        heap.Type=D3D12_HEAP_TYPE_UPLOAD;
        checked(r.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&r.upload)));
        heap.Type=D3D12_HEAP_TYPE_READBACK;
        checked(r.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r.readback)));
        const reshade::api::resource_desc desc(width,height,1,1,reshade::api::format::r8g8b8a8_unorm,1,
            reshade::api::memory_heap::gpu_only,reshade::api::resource_usage::render_target);
        bridge_=std::make_unique<D3D12Bridge>(runtime,desc,r.device.Get(),r.queue.Get());
    }
    ~D3D12LocalPixels() {
        if(!resources_->retired||(bridge_&&!bridge_->safe_to_fallback())) {
            in_process_gpu_uncertain = true;
            // The worker can outlive the window; do not release live allocations.
            bridge_.release();resources_.release();
        }
    }
    D3D12Bridge& bridge(){return *bridge_;}
    void upload(const std::vector<uint8_t>& pixels) {
        if(pixels.size()!=size_t(width_)*height_*4)throw std::runtime_error("Local native pixel extent mismatch");
        auto& r=*resources_;void* data=nullptr;D3D12_RANGE empty{};checked(r.upload->Map(0,&empty,&data));
        for(unsigned row=0;row<height_;++row)memcpy(static_cast<uint8_t*>(data)+r.footprint.Footprint.RowPitch*row,
            pixels.data()+size_t(row)*width_*4,size_t(width_)*4);
        r.upload->Unmap(0,nullptr);transfer(true);
    }
    bool frame(const Settings& settings,bool apply) {
        return bridge_->frame({uint64_t(uintptr_t(resources_->color.Get()))},settings,apply,false);
    }
    std::vector<uint8_t> download() {
        transfer(false);auto& r=*resources_;void* data=nullptr;D3D12_RANGE range{0,SIZE_T(r.bytes)},empty{};
        checked(r.readback->Map(0,&range,&data));std::vector<uint8_t> pixels(size_t(width_)*height_*4);
        for(unsigned row=0;row<height_;++row)memcpy(pixels.data()+size_t(row)*width_*4,
            static_cast<const uint8_t*>(data)+r.footprint.Footprint.RowPitch*row,size_t(width_)*4);
        r.readback->Unmap(0,&empty);return pixels;
    }
};
} // namespace phoenix_compat
