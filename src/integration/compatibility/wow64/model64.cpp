#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <nvsdk_ngx.h>
#include "../../../../native_runtime/phoenix_native.h"
#include "../../../../third_party/nlohmann/json.hpp"
#include "model_contract.h"
#include "umd_bridge.h"
#include "client32.h"
#include <chrono>
#include <source_location>
using Json=nlohmann::json;
using Microsoft::WRL::ComPtr;
namespace fs=std::filesystem;
extern "C" BOOL WINAPI PhoenixNrLoad(const wchar_t*);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrInit(unsigned long long,const wchar_t*,ID3D12Device*,NVSDK_NGX_Version,const NVSDK_NGX_Parameter*);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrCreate(ID3D12GraphicsCommandList*,NVSDK_NGX_Feature,NVSDK_NGX_Parameter*,NVSDK_NGX_Handle**);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrEvaluate(ID3D12GraphicsCommandList*,const NVSDK_NGX_Handle*,const NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrRelease(NVSDK_NGX_Handle*);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrShutdown(ID3D12Device*);
extern "C" HMODULE WINAPI PhoenixNrModule();
extern "C" __declspec(dllexport) void PhoenixSetGdiRpc64(PhoenixGdiRpc* rpc) {wow64_umd_probe::gdi=rpc;}
static void require(bool valid,const char* message) {if(!valid) throw std::runtime_error(message);}
static std::string host_allocation_context() {
    std::string detail="; host heap calls="+std::to_string(wow64_runtime_diagnostics::heap_calls)+
        ", failures="+std::to_string(wow64_runtime_diagnostics::failure_count);
    const auto failures=std::min<LONG>(LONG(wow64_runtime_diagnostics::failure_count),32);
    if(failures>0) {
        const auto& failure=wow64_runtime_diagnostics::failures[failures-1];
        detail+="; last host operation="+std::to_string(failure.operation)+
            ", bytes="+std::to_string(failure.bytes)+", flags="+std::to_string(failure.flags)+
            ", Windows error="+std::to_string(failure.error);
    }
    HEAP_SUMMARY summary{}; summary.cb=sizeof(summary);
    if(HeapSummary(GetProcessHeap(),0,&summary))
        detail+="; heap allocated="+std::to_string(summary.cbAllocated)+
            ", committed="+std::to_string(summary.cbCommitted)+
            ", reserved="+std::to_string(summary.cbReserved)+
            ", maximum reserve="+std::to_string(summary.cbMaxReserve);
    return detail;
}
static std::string low_address_space_context() {
    constexpr uint64_t limit = uint64_t(1) << 32;
    uint64_t cursor = 0, free_bytes = 0, largest = 0;
    while (cursor < limit) {
        MEMORY_BASIC_INFORMATION region{};
        if (!VirtualQuery(reinterpret_cast<void*>(cursor), &region, sizeof(region))) break;
        const auto begin = uint64_t(region.BaseAddress);
        const auto end = std::min(limit, begin + uint64_t(region.RegionSize));
        if (end <= cursor) break;
        if (region.State == MEM_FREE) {
            const auto bytes = end - std::max(cursor, begin);
            free_bytes += bytes;
            largest = std::max(largest, bytes);
        }
        cursor = end;
    }
    return "; before model load, free low address bytes=" + std::to_string(free_bytes) +
        ", largest low region=" + std::to_string(largest);
}
static void hr(HRESULT result,std::source_location where=std::source_location::current()) {
    if(FAILED(result)) throw std::runtime_error("Native model D3D12 failure "+std::to_string(uint32_t(result))+" at line "+std::to_string(where.line()));
}
template<class T> static T symbol(HMODULE module,const char* name) {
    auto value=reinterpret_cast<T>(GetProcAddress(module,name)); require(value!=nullptr,name); return value;
}
struct Device {
    ComPtr<IDXGIFactory4> factory; ComPtr<IDXGIAdapter1> adapter; ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list; ComPtr<ID3D12Fence> fence; uint64_t serial=0;
    void create() {
        hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for(unsigned i=0;;++i) {
            hr(factory->EnumAdapters1(i,&adapter)); DXGI_ADAPTER_DESC1 desc{}; hr(adapter->GetDesc1(&desc));
            if(desc.VendorId==0x10de&&!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)) break; adapter.Reset();
        }
        const auto created=D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device));
        if(FAILED(created)) {
            auto detail="Native D3D12 device HRESULT="+std::to_string(uint32_t(created));
            const auto failures=std::min<LONG>(LONG(wow64_runtime_diagnostics::failure_count),32);
            if(failures>0) {
                const auto& failure=wow64_runtime_diagnostics::failures[failures-1];
                detail+="; last allocation operation="+std::to_string(failure.operation)+
                    ", bytes="+std::to_string(failure.bytes)+", Windows error="+std::to_string(failure.error);
            }
            SYSTEM_INFO system{};GetNativeSystemInfo(&system);
            detail+="; native maximum address="+std::to_string(uint64_t(system.lpMaximumApplicationAddress));
            const auto reservations=std::min<LONG>(LONG(wow64_runtime_diagnostics::reservation_count),32);
            unsigned shown=0;
            for(LONG i=0;i<reservations&&shown<1;++i) {
                const auto& value=wow64_runtime_diagnostics::reservations[i];
                if(!value.initial) {
                    ++shown;
                    detail+="; reserve="+std::to_string(value.bytes)+
                        ", err="+std::to_string(value.error)+", retry="+std::to_string(value.retry_error)+
                        ", method="+std::to_string(value.method)+", params="+std::to_string(value.parameters)+
                        ", flags="+std::to_string(value.flags)+", protect="+std::to_string(value.protect)+
                        ", result="+std::to_string(value.retried);
                }
            }
            detail+=host_allocation_context();
            ComPtr<IDXGIAdapter3> memory;
            if(SUCCEEDED(adapter.As(&memory))) {
                DXGI_QUERY_VIDEO_MEMORY_INFO budget{};
                if(SUCCEEDED(memory->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&budget)))
                    detail+="; GPU local usage="+std::to_string(budget.CurrentUsage)+", budget="+std::to_string(budget.Budget);
            }
            throw std::runtime_error(detail);
        }
        D3D12_COMMAND_QUEUE_DESC desc{}; hr(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)));
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
        hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    }
    void submit() {
        hr(list->Close()); ID3D12CommandList* lists[]{list.Get()}; queue->ExecuteCommandLists(1,lists);
        hr(queue->Signal(fence.Get(),++serial)); auto event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        require(event!=nullptr,"Native model fence event"); auto result=fence->SetEventOnCompletion(serial,event);
        auto waited=SUCCEEDED(result)?WaitForSingleObject(event,10000):WAIT_FAILED; CloseHandle(event);
        hr(result); require(waited==WAIT_OBJECT_0,"Native model fence timeout"); hr(device->GetDeviceRemovedReason());
        hr(allocator->Reset()); hr(list->Reset(allocator.Get(),nullptr));
    }
    void barrier(ID3D12Resource* value,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={value,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; list->ResourceBarrier(1,&b);
    }
    ComPtr<ID3D12Resource> buffer(uint64_t bytes,D3D12_HEAP_TYPE heap) {
        D3D12_HEAP_PROPERTIES p{}; p.Type=heap; D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width=bytes; d.Height=1; d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> result; hr(device->CreateCommittedResource(&p,D3D12_HEAP_FLAG_NONE,&d,
            heap==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&result))); return result;
    }
    ComPtr<ID3D12Resource> image(unsigned w,unsigned h,DXGI_FORMAT format,bool output=false) {
        D3D12_HEAP_PROPERTIES p{}; p.Type=D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d{};
        d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width=w; d.Height=h; d.Format=format;
        d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1; if(output) d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result; hr(device->CreateCommittedResource(&p,D3D12_HEAP_FLAG_NONE,&d,
            output?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&result))); return result;
    }
};
class Models {
    Device gpu_;
    Json config_;
    unsigned width_,height_;
    HMODULE native_=nullptr,core_=nullptr;
    PhoenixNativeSession* phoenix_=nullptr;
    NVSDK_NGX_Parameter* params_=nullptr; NVSDK_NGX_Handle* nvidia_=nullptr;
    ComPtr<ID3D12Resource> color_,motion_,output_,upload_,motion_upload_,readback_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{},motion_fp_{}; uint64_t bytes_=0,motion_bytes_=0;
    bool color_uploaded_=false;
    bool nvidia_initialized_=false;
    phoenix_wow64::ModelChannel* packet_=nullptr;
    unsigned nvidia_imports_=0;
    inline static std::string loaded_nvidia_sha_;
    void initialize_phoenix() {
        if(phoenix_) return;
        native_=LoadLibraryExW(fs::u8path(config_.at("native_library").get<std::string>()).c_str(),nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(native_!=nullptr,"Native Phoenix DLSS5 library unavailable");
        // The standalone library is loaded directly, just like the NGX core.
        // Adapt its native imports before it caches loader/allocator entry points.
        wow64_umd_probe::patch(native_);
        auto create=symbol<decltype(&phoenix_native_create)>(native_,"phoenix_native_create");
        const auto text=config_.dump();
        phoenix_=create(text.c_str(),gpu_.device.Get(),gpu_.queue.Get());
        if (!phoenix_) {
            std::string detail=symbol<decltype(&phoenix_native_error)>(native_,"phoenix_native_error")();
            if (detail.find("allocation")!=std::string::npos) {
                const auto count=wow64_runtime_diagnostics::failure_count;
                if (count>0) {
                    const auto& failure=wow64_runtime_diagnostics::failures[std::min<LONG>(count-1,31)];
                    detail+="; last host allocation bytes="+std::to_string(failure.bytes)+
                        ", Windows error="+std::to_string(failure.error);
                }
                detail+=host_allocation_context();
            }
            throw std::runtime_error(detail);
        }
    }
    void initialize_nvidia(const wchar_t* cuda_path,phoenix_wow64::ModelChannel& packet) {
        if(nvidia_) return;
        const auto source=phoenix_model::WeightSource::config(config_.at("weight_source"));
        require(!source.dll.empty(),"NVIDIA DLSS5 mode requires a selected NVIDIA DLSS5 model DLL, not a disk model");
        require(phoenix_wow64::file_digest(source.dll)==source.dll_sha256,"NVIDIA DLSS5 model digest mismatch");
        require(loaded_nvidia_sha_.empty()||loaded_nvidia_sha_==source.dll_sha256,
            "Changing the NVIDIA DLSS5 library in the native x86 route requires the helper or a game restart; the previous library stays loaded.");
        core_=LoadLibraryW((fs::path(cuda_path).parent_path()/"_nvngx.dll").c_str()); require(core_!=nullptr,"Native NGX core unavailable");
        // These libraries are loaded here, rather than through the driver's
        // adapted loader. Apply the same native import/allocation handling
        // before they initialize or cache dependency entry points.
        wow64_umd_probe::patch(core_);
        auto allocate=symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**)>(core_,"NVSDK_NGX_D3D12_AllocateParameters");
        require(NVSDK_NGX_SUCCEED(allocate(&params_))&&params_,"NVIDIA DLSS5 parameter allocation failed");
        const auto address_space = low_address_space_context();
        const auto last_status = reinterpret_cast<LONG(NTAPI*)()>(GetProcAddress(
            GetModuleHandleW(L"ntdll.dll"), "RtlGetLastNtStatus"));
        if (!PhoenixNrLoad(source.dll.c_str())) {
            const auto error = GetLastError();
            const auto status = last_status ? uint32_t(last_status()) : 0;
            throw std::runtime_error("NVIDIA DLSS5 model load failed; Windows error=" +
                std::to_string(error) + ", NT status=" + std::to_string(status) +
                address_space + host_allocation_context());
        }
        const auto model_module=PhoenixNrModule();
        require(model_module!=nullptr,"Loaded NVIDIA model module unavailable");
        const auto before=wow64_umd_probe::slot_count;
        wow64_umd_probe::patch(model_module);
        nvidia_imports_=wow64_umd_probe::slot_count-before;
        loaded_nvidia_sha_=source.dll_sha256;
        if(config_.value("nvidia_preset",1u)==2) {
            const auto mapping=LoadLibraryExW(source.dll.c_str(),nullptr,LOAD_LIBRARY_AS_DATAFILE|LOAD_LIBRARY_AS_IMAGE_RESOURCE);
            require(mapping!=nullptr,"Cannot inspect selected NVIDIA model resources");
            const bool available=FindResourceW(mapping,L"WEIGHTS_CC_SMALL_FP8",MAKEINTRESOURCEW(10))!=nullptr;
            FreeLibrary(mapping);
            require(available,"Selected NVIDIA DLL has no Lightweight FP8 model; choose 310.9.16");
        }
        // No game SR hook, no upscaling at 1:1; this is direct NR evaluation.
        const auto cache=fs::temp_directory_path()/"PhoenixAddon"/"ngx";
        fs::create_directories(cache);
        const auto initialized=PhoenixNrInit(0x1000000ull,cache.c_str(),gpu_.device.Get(),NVSDK_NGX_Version_API,params_);
        if(!NVSDK_NGX_SUCCEED(initialized)) throw std::runtime_error("NVIDIA DLSS5 initialization failed: "+std::to_string(uint32_t(initialized)));
        nvidia_initialized_=true;
        params_->Reset();
        for(auto key:{"DLSSNR.Width","DLSSNR.InputWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"}) params_->Set(key,width_);
        for(auto key:{"DLSSNR.Height","DLSSNR.InputHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"}) params_->Set(key,height_);
        for(auto key:{"CreationNodeMask","VisibilityNodeMask","DLSSNR.SDR","DLSSNR.AutoExposure"}) params_->Set(key,1u);
        for(auto key:{"DLSSNR.Upscaling","PerfQualityValue","DLSSNR.Hdr","DLSS.Feature.Create.Flags"}) params_->Set(key,0u);
        params_->Set("DLSSNR.Hint.Render.Preset",config_.value("nvidia_preset",1u));
        for(auto key:{"DLSSNR.Scale","DLSSNR.ScalingRatio","MV.Scale.X","MV.Scale.Y","InPreExposure","InExposureScale"}) params_->Set(key,1.f);
        try {
            packet.nvidia_create_result=uint32_t(PhoenixNrCreate(gpu_.list.Get(),static_cast<NVSDK_NGX_Feature>(18),params_,&nvidia_));
        } catch(const std::bad_alloc&) {
            throw std::runtime_error("NVIDIA DLSS5 host allocation failed"+host_allocation_context());
        }
        if(!NVSDK_NGX_SUCCEED(static_cast<NVSDK_NGX_Result>(packet.nvidia_create_result))||!nvidia_) {
            std::string detail;
            if(wow64_runtime_diagnostics::failure_count>0) {
                const auto index=std::min<LONG>(wow64_runtime_diagnostics::failure_count-1,31);
                const auto& failure=wow64_runtime_diagnostics::failures[index];
                detail=" (host allocation: "+std::to_string(failure.bytes)+" bytes, error "+std::to_string(failure.error)+")";
            }
            detail+="; adapted imports="+std::to_string(wow64_umd_probe::slot_count);
            detail+="; model imports="+std::to_string(nvidia_imports_);
            detail+=host_allocation_context();
            throw std::runtime_error("NVIDIA DLSS5 feature creation failed: "+std::to_string(packet.nvidia_create_result)+detail);
        }
        gpu_.submit();++packet.nvidia_creates;
        color_=gpu_.image(width_,height_,DXGI_FORMAT_R16G16B16A16_FLOAT); motion_=gpu_.image(width_,height_,DXGI_FORMAT_R16G16_FLOAT);
        output_=gpu_.image(width_,height_,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
        auto desc=color_->GetDesc(); gpu_.device->GetCopyableFootprints(&desc,0,1,0,&footprint_,nullptr,nullptr,&bytes_);
        desc=motion_->GetDesc(); gpu_.device->GetCopyableFootprints(&desc,0,1,0,&motion_fp_,nullptr,nullptr,&motion_bytes_);
        upload_=gpu_.buffer(bytes_,D3D12_HEAP_TYPE_UPLOAD); readback_=gpu_.buffer(bytes_,D3D12_HEAP_TYPE_READBACK);
        motion_upload_=gpu_.buffer(motion_bytes_,D3D12_HEAP_TYPE_UPLOAD); void* data=nullptr; D3D12_RANGE empty{};
        hr(motion_upload_->Map(0,&empty,&data)); memset(data,0,size_t(motion_bytes_)); motion_upload_->Unmap(0,nullptr);
        D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=motion_.Get(); a.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        b.pResource=motion_upload_.Get(); b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; b.PlacedFootprint=motion_fp_;
        gpu_.list->CopyTextureRegion(&a,0,0,0,&b,nullptr); gpu_.barrier(motion_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu_.submit();
    }
  public:
    Models(const Json& config,unsigned width,unsigned height,phoenix_wow64::ModelChannel& packet)
        :config_(config),width_(width),height_(height),packet_(&packet) {gpu_.create();gpu_.submit();}
    ~Models() {
        if(phoenix_) {
            const auto destroy=reinterpret_cast<decltype(&phoenix_native_destroy)>(GetProcAddress(native_,"phoenix_native_destroy"));
            if(destroy) destroy(phoenix_);
        }
        if(nvidia_) PhoenixNrRelease(nvidia_);
        if(params_) {
            const auto destroy=reinterpret_cast<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*)>(GetProcAddress(core_,"NVSDK_NGX_D3D12_DestroyParameters"));
            if(destroy) destroy(params_);
        }
        // Releasing a feature alone leaves NGX's device context alive. Every
        // worker recreation owns a different D3D12 device; retire that context
        // as well, before the device is released, including failed creates.
        if(nvidia_initialized_) {
            const auto result=PhoenixNrShutdown(gpu_.device.Get());
            if(!NVSDK_NGX_SUCCEED(result)) {
                ++packet_->base.cleanup_failures;
                packet_->base.status=-1;
                strncpy_s(packet_->base.error,"NVIDIA DLSS5 device shutdown failed",_TRUNCATE);
            }
        }
    }
    void frame(phoenix_wow64::ModelChannel& packet) {
        auto& c=packet.base; const auto controls=Json::parse(reinterpret_cast<const char*>(packet.parameters_json));
        if(packet.backend==3) {
            initialize_phoenix(); PhoenixNativeControls options{};
            symbol<decltype(&phoenix_native_default_controls)>(native_,"phoenix_native_default_controls")(&options);
            options.reset=1; options.local_tone=controls.at("tone"); options.local_structure=controls.at("structure");
            options.skin_structure=controls.at("skin"); options.auto_mask=controls.at("auto_mask").get<bool>();
            auto error=symbol<decltype(&phoenix_native_error)>(native_,"phoenix_native_error");
            require(symbol<decltype(&phoenix_native_frame_host)>(native_,"phoenix_native_frame_host")(phoenix_,
                reinterpret_cast<const void*>(c.color),size_t(c.plane_bytes),width_*8,nullptr,0,0,
                reinterpret_cast<void*>(c.output),size_t(c.plane_bytes),width_*8,&options)==0,error());
            const auto status=symbol<decltype(&phoenix_native_status)>(native_,"phoenix_native_status");
            std::vector<char> text(status(phoenix_,nullptr,0)); status(phoenix_,text.data(),text.size());
            c.gpu_ms=Json::parse(text.data()).at("gpu_processing_ms");
        } else {
            require(packet.backend==1,"Unknown native model backend"); initialize_nvidia(reinterpret_cast<const wchar_t*>(c.cuda_library),packet);
            void* data=nullptr; D3D12_RANGE empty{}; hr(upload_->Map(0,&empty,&data));
            for(unsigned y=0;y<height_;++y) memcpy(static_cast<char*>(data)+footprint_.Offset+y*footprint_.Footprint.RowPitch,
                reinterpret_cast<const char*>(c.color)+size_t(y)*width_*8,width_*8);
            upload_->Unmap(0,nullptr);
            if(color_uploaded_) gpu_.barrier(color_.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=color_.Get(); a.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            b.pResource=upload_.Get(); b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; b.PlacedFootprint=footprint_;
            gpu_.list->CopyTextureRegion(&a,0,0,0,&b,nullptr); gpu_.barrier(color_.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); gpu_.submit();color_uploaded_=true;
            params_->Reset();
            params_->Set("DLSSNR.Intensity",controls.value("intensity",1.f));
            params_->Set("DLSSNR.Style",unsigned(controls.value("style",0)));
            params_->Set("DLSSNR.UICorrection",controls.value("ui_correction",0)); params_->Set("DLSSNR.Color",color_.Get()); params_->Set("Color",color_.Get()); params_->Set("DLSSNR.Output",output_.Get()); params_->Set("Output",output_.Get());
            for(auto key:{"DLSSNR.MVec","MVec","MotionVectors"}) params_->Set(key,motion_.Get());
            for(auto key:{"DLSSNR.ColorSubrectWidth","DLSSNR.MVecSubrectWidth","DLSSNR.OutputSubrectWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"}) params_->Set(key,width_);
            for(auto key:{"DLSSNR.ColorSubrectHeight","DLSSNR.MVecSubrectHeight","DLSSNR.OutputSubrectHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"}) params_->Set(key,height_);
            for(auto key:{"DLSSNR.MVecScaleX","DLSSNR.MVecScaleY","DLSSNR.ScalingRatio","DLSSNR.Scale","DLSSNR.GlobalToneStrength","DLSS.Pre.Exposure","DLSS.Exposure.Scale"}) params_->Set(key,1.f);
            params_->Set("DLSSNR.LocalToneStrength",controls.at("tone").get<float>()); params_->Set("DLSSNR.LocalStructureStrength",controls.at("structure").get<float>());
            params_->Set("DLSSNR.SkinStructureStrength",controls.at("skin").get<float>()); params_->Set("DLSSNR.UseAutoMask",unsigned(controls.at("auto_mask").get<bool>()));
            params_->Set("DLSSNR.SDR",1u); params_->Set("DLSSNR.Enabled",1u); params_->Set("DLSSNR.Reset",1u);
            packet.nvidia_evaluate_result=uint32_t(PhoenixNrEvaluate(gpu_.list.Get(),nvidia_,params_,nullptr));
            require(NVSDK_NGX_SUCCEED(static_cast<NVSDK_NGX_Result>(packet.nvidia_evaluate_result)),"NVIDIA DLSS5 evaluation failed");
            gpu_.submit();++packet.nvidia_evaluates;
            gpu_.barrier(output_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            a={};b={};a.pResource=readback_.Get();a.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;a.PlacedFootprint=footprint_;
            b.pResource=output_.Get();b.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;gpu_.list->CopyTextureRegion(&a,0,0,0,&b,nullptr);
            gpu_.barrier(output_.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);gpu_.submit();
            D3D12_RANGE all{0,size_t(bytes_)};hr(readback_->Map(0,&all,&data));
            for(unsigned y=0;y<height_;++y) memcpy(reinterpret_cast<char*>(c.output)+size_t(y)*width_*8,
                static_cast<const char*>(data)+footprint_.Offset+y*footprint_.Footprint.RowPitch,width_*8);
            readback_->Unmap(0,&empty); c.gpu_ms=0; // No wall clock masquerading as GPU timestamps.
        }
        ++c.completed_frames;
    }
};
extern "C" __declspec(dllexport) int PhoenixModelServe64(phoenix_wow64::ModelChannel* packet) {
    if(!packet) return 2; auto& c=packet->base; bool passed=false;
    try {
        require(packet->bytes==sizeof(*packet),"Native model packet size mismatch; update the matching addon and runtime together");
        require(c.abi==phoenix_driver_probe::version&&c.struct_bytes==sizeof(c)&&c.caller_bits==32&&c.width&&c.height&&c.width<=8192&&c.height<=8192,"Invalid native model packet");
        require(packet->config_json&&c.color&&c.output&&c.plane_bytes==uint64_t(c.width)*c.height*8,"Invalid native model buffers");
        c.driver_pid=GetCurrentProcessId(); c.driver_bits=64;
        const DWORD offsets[]{c.appcompat_offsets[0],c.appcompat_offsets[1],c.appcompat_offsets[2],c.appcompat_offsets[3]};
        require(wow64_umd_probe::gdi&&wow64_umd_probe::install_native_compat_view(c.ntdll_timestamp,offsets),"Native model compatibility view unavailable");
        wow64_runtime_diagnostics::retry_native_reservation=true;wow64_umd_probe::bridge_dc_creation=true;wow64_umd_probe::bridge_display_gdi=true;
        require(wow64_umd_probe::install()!=0,"Native model import adaptation unavailable");
        require(wow64_umd_probe::install_gdi_export(),"Native model DeleteDC adaptation unavailable");
        require(wow64_umd_probe::install_display_exports(),"Native model display adaptation unavailable");
        auto cuda=LoadLibraryW(reinterpret_cast<const wchar_t*>(c.cuda_library));require(cuda!=nullptr,"Native CUDA load failed");
        wow64_umd_probe::owner=GetCurrentThreadId();wow64_umd_probe::patch(cuda);
        require(symbol<int(*)(unsigned)>(cuda,"cuInit")(0)==0,"Native CUDA initialization failed");
        {
            Models models(Json::parse(reinterpret_cast<const char*>(packet->config_json)),c.width,c.height,*packet);
            SetEvent(reinterpret_cast<HANDLE>(c.response_event));
            for(;;) {
                require(WaitForSingleObject(reinterpret_cast<HANDLE>(c.request_event),INFINITE)==WAIT_OBJECT_0,"Native model wait failed");
                if(c.operation==phoenix_driver_probe::shutdown) break;
                require(c.operation==phoenix_driver_probe::frame,"Unknown native model request"); models.frame(*packet);
                SetEvent(reinterpret_cast<HANDLE>(c.response_event));
            }
        }
        passed=true;
    } catch(const std::exception& error) {c.status=-1;strncpy_s(c.error,error.what(),_TRUNCATE);}
    wow64_umd_probe::restore();wow64_umd_probe::restore_gdi_export();wow64_umd_probe::restore_display_exports();
    c.native_compat_restored=wow64_umd_probe::restore_native_compat_view();
    if(!c.native_compat_restored||c.cleanup_failures) {c.status=-1;passed=false;}
    SetEvent(reinterpret_cast<HANDLE>(c.response_event));return passed?0:2;
}
