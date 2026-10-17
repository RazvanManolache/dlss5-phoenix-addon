// Disposable native-thread research probe. Not linked into any shipped addon.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <nvsdk_ngx.h>
#include "../../third_party/nvapi/nvapi.h"
#include "../../native_runtime/phoenix_native.h"
#include "../../third_party/nlohmann/json.hpp"
#include "wow64_umd_probe.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <bcrypt.h>
#pragma comment(lib,"bcrypt.lib")
using Json = nlohmann::json;
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
extern "C" BOOL WINAPI PhoenixNrLoad(const wchar_t*);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrInit(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrCreate(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrEvaluate(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
extern "C" NVSDK_NGX_Result NVSDK_CONV PhoenixNrRelease(NVSDK_NGX_Handle*);
extern "C" __declspec(dllexport) void PhoenixSetGdiRpc64(PhoenixGdiRpc* rpc) { wow64_umd_probe::gdi = rpc; }
static void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
static void hr(HRESULT value) { if (FAILED(value)) throw std::runtime_error("D3D12 HRESULT "+std::to_string(uint32_t(value))); }
static std::string output_sha256(const std::vector<uint16_t>& pixels) {
    unsigned char digest[32]{}; char hex[65]{};
    require(BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,reinterpret_cast<PUCHAR>(const_cast<uint16_t*>(pixels.data())),
        ULONG(pixels.size()*2),digest,sizeof(digest))>=0,"Output SHA256 failed");
    for(unsigned i=0;i<32;++i) sprintf_s(hex+i*2,3,"%02x",unsigned(digest[i]));
    return hex;
}
template<class T> static T symbol(HMODULE library, const char* name) {
    auto value = reinterpret_cast<T>(GetProcAddress(library, name));
    require(value != nullptr, name); return value;
}
struct Device {
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t serial = 0;
    void create() {
        hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for (unsigned i=0;;++i) {
            hr(factory->EnumAdapters1(i, &adapter));
            DXGI_ADAPTER_DESC1 desc{}; hr(adapter->GetDesc1(&desc));
            if (desc.VendorId == 0x10de && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter.Reset();
        }
        hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC desc{};
        hr(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
    void submit() {
        hr(list->Close()); ID3D12CommandList* lists[]{list.Get()}; queue->ExecuteCommandLists(1, lists);
        hr(queue->Signal(fence.Get(), ++serial));
        const auto event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        require(event != nullptr, "Fence event");
        const auto registered = fence->SetEventOnCompletion(serial, event);
        const auto waited = SUCCEEDED(registered) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
        CloseHandle(event); hr(registered); require(waited == WAIT_OBJECT_0, "Fence timeout");
        hr(device->GetDeviceRemovedReason());
        hr(allocator->Reset()); hr(list->Reset(allocator.Get(), nullptr));
    }
    ComPtr<ID3D12Resource> buffer(uint64_t bytes, D3D12_HEAP_TYPE heap) {
        D3D12_HEAP_PROPERTIES props{}; props.Type=heap;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=bytes;
        desc.Height=1; desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
        desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> value;
        hr(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
            heap==D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&value))); return value;
    }
    ComPtr<ID3D12Resource> texture(unsigned width, unsigned height, DXGI_FORMAT format, bool output=false) {
        D3D12_HEAP_PROPERTIES props{}; props.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width=width; desc.Height=height; desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
        desc.Format=format; desc.Flags=output ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> value;
        hr(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
            output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&value))); return value;
    }
    void barrier(ID3D12Resource* value, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={value, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after}; list->ResourceBarrier(1,&b);
    }
    void copy_control() {
        auto a=buffer(4096,D3D12_HEAP_TYPE_UPLOAD), b=buffer(4096,D3D12_HEAP_TYPE_READBACK);
        void* data=nullptr; D3D12_RANGE empty{}; hr(a->Map(0,&empty,&data));
        for (unsigned i=0;i<1024;++i) static_cast<unsigned*>(data)[i]=i^0x71895a32;
        a->Unmap(0,nullptr); list->CopyBufferRegion(b.Get(),0,a.Get(),0,4096); submit();
        D3D12_RANGE all{0,4096}; hr(b->Map(0,&all,&data)); bool ok=true;
        for (unsigned i=0;i<1024;++i) ok &= static_cast<unsigned*>(data)[i]==(i^0x71895a32);
        b->Unmap(0,&empty); require(ok,"GPU copy mismatch");
    }
};
using Clock = std::chrono::steady_clock;
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
static Json timing_stats(std::vector<double> values) {
    require(!values.empty(),"Empty benchmark timing samples");
    for(auto value:values) require(std::isfinite(value)&&value>0,"Invalid benchmark timing");
    const auto mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    std::sort(values.begin(),values.end());
    return {{"samples",values.size()},{"mean_ms",mean},{"median_ms",values[values.size()/2]},
        {"p95_ms",values[size_t(std::ceil(values.size()*.95))-1]}};
}
template<class Step> static Json benchmark_frames(Step step) {
    const auto warm=Clock::now();
    do { step(); } while(milliseconds(warm)<5000);
    std::vector<double> gpu,wall;
    const auto begin=Clock::now();
    do {
        const auto sample=step(); gpu.push_back(sample.first); wall.push_back(sample.second);
    } while(milliseconds(begin)<20000);
    const auto elapsed=milliseconds(begin);
    require(gpu.size()>=16,"Too few completed benchmark frames");
    return {{"width",1920},{"height",1080},{"scale",1},{"warmup_seconds",5},
        {"requested_seconds",20},{"measured_seconds",elapsed/1000},{"completed_frames",gpu.size()},
        {"completed_output_fps",gpu.size()*1000.0/elapsed},{"gpu_processing",timing_stats(gpu)},
        {"synchronized_call",timing_stats(wall)},
        {"scope","GPU-resident textures, serial fence-completed calls; excludes initialization, image validation, host transfers and game presentation. Native worker entered once, not per-frame x86 caller handoff."}};
}
static Json benchmark_phoenix(Device& gpu, PhoenixNativeSession* session, HMODULE library,
    unsigned width, unsigned height, PhoenixNativeControls controls, bool reset_each_frame) {
    auto frame=symbol<decltype(&phoenix_native_frame_d3d12)>(library,"phoenix_native_frame_d3d12");
    auto status=symbol<decltype(&phoenix_native_status)>(library,"phoenix_native_status");
    auto error=symbol<decltype(&phoenix_native_error)>(library,"phoenix_native_error");
    auto color=gpu.texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto motion=gpu.texture(width,height,DXGI_FORMAT_R16G16_FLOAT);
    auto output=gpu.texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{},mf{}; uint64_t bytes=0,mbytes=0;
    auto desc=color->GetDesc(); gpu.device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
    desc=motion->GetDesc(); gpu.device->GetCopyableFootprints(&desc,0,1,0,&mf,nullptr,nullptr,&mbytes);
    auto upload=gpu.buffer(bytes,D3D12_HEAP_TYPE_UPLOAD), mv=gpu.buffer(mbytes,D3D12_HEAP_TYPE_UPLOAD);
    auto poison=gpu.buffer(bytes,D3D12_HEAP_TYPE_UPLOAD);
    void* data=nullptr; D3D12_RANGE empty{}; hr(upload->Map(0,&empty,&data));
    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<4;++c) {
        const auto i=(size_t(y)*width+x)*4+c;
        reinterpret_cast<uint16_t*>(static_cast<char*>(data)+fp.Offset+y*fp.Footprint.RowPitch+x*8)[c]=
            c==3?0x3c00:uint16_t(0x3000+(i*19)%0x800);
    }
    upload->Unmap(0,nullptr); hr(mv->Map(0,&empty,&data)); memset(data,0,size_t(mbytes)); mv->Unmap(0,nullptr);
    hr(poison->Map(0,&empty,&data));
    for(size_t i=0;i<size_t(bytes)/2;++i) static_cast<uint16_t*>(data)[i]=0x7e00;
    poison->Unmap(0,nullptr);
    auto copy=[&](ID3D12Resource* destination,ID3D12Resource* source,D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint) {
        D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=destination; a.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        b.pResource=source; b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; b.PlacedFootprint=footprint;
        gpu.list->CopyTextureRegion(&a,0,0,0,&b,nullptr);
    };
    copy(color.Get(),upload.Get(),fp); copy(motion.Get(),mv.Get(),mf);
    gpu.barrier(color.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.barrier(motion.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
    copy(output.Get(),poison.Get(),fp);
    gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); gpu.submit();
    controls.reset=1;
    auto result=benchmark_frames([&]() {
        const auto started=Clock::now();
        require(frame(session,color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,motion.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,&controls)==0,error());
        const auto wall=milliseconds(started); controls.reset=reset_each_frame?1:0;
        const auto size=status(session,nullptr,0); require(size>0&&size<1024*1024,"Native status size");
        std::vector<char> text(size); require(status(session,text.data(),text.size())==size,"Native status read");
        const auto state=Json::parse(text.data());
        require(state.at("backend")=="native_rtx"&&!state.at("failed").get<bool>(),"Non-native benchmark backend");
        return std::make_pair(state.at("gpu_processing_ms").get<double>(),wall);
    });
    auto readback=gpu.buffer(bytes,D3D12_HEAP_TYPE_READBACK);
    gpu.barrier(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=readback.Get(); a.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; a.PlacedFootprint=fp;
    b.pResource=output.Get(); b.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; gpu.list->CopyTextureRegion(&a,0,0,0,&b,nullptr); gpu.submit();
    D3D12_RANGE all{0,size_t(bytes)}; hr(readback->Map(0,&all,&data));
    std::vector<uint16_t> pixels(size_t(width)*height*3); size_t changed=0;
    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<3;++c) {
        const auto value=reinterpret_cast<uint16_t*>(static_cast<char*>(data)+fp.Offset+y*fp.Footprint.RowPitch+x*8)[c];
        require((value&0x7c00)!=0x7c00,"Nonfinite Phoenix benchmark output");
        const auto i=(size_t(y)*width+x)*4+c; changed+=value!=uint16_t(0x3000+(i*19)%0x800);
        pixels[(size_t(y)*width+x)*3+c]=value;
    }
    readback->Unmap(0,&empty); require(changed>0,"Unchanged Phoenix benchmark output");
    result["output_sha256"]=output_sha256(pixels); result["changed_rgb_samples"]=changed;
    result["gpu_timing_scope"]="native API GPU timestamps: model and output copy";
    result["reset_each_frame"]=reset_each_frame;
    return result;
}
extern "C" __declspec(dllexport) int PhoenixPortableFrameProbe64(const wchar_t* path) {
    Json report{{"passed",false},{"frames",Json::array()},{"pointer_bits",64}};
    BOOL wow64=FALSE;
    require(IsWow64Process(GetCurrentProcess(),&wow64)!=0,"Process architecture query failed");
    report["process_pid"]=GetCurrentProcessId(); report["process_bits"]=wow64?32:64; report["helper_pid"]=0;
    Json job; fs::path output;
    uint64_t* native_image_slot=nullptr;
    uint64_t original_native_image=0;
    auto save=[&](const char* stage) {
        report["stage"]=stage;
        if (!output.empty()) { std::ofstream file(output); file<<report.dump(2)<<'\n'; require(bool(file),"Receipt write"); }
    };
    try {
        std::ifstream file{fs::path(path)}; file>>job;
        output=fs::u8path(job.at("output").get<std::string>());
        const auto backend=job.at("backend").get<std::string>(); report["backend"]=backend;
        if(job.value("native_appcompat_view",false) && wow64_umd_probe::gdi) {
            const auto fields=job.at("appcompat_offsets").get<std::vector<DWORD>>();
            require(fields.size()==4,"Native compatibility symbols unavailable");
            const DWORD offsets[]{fields[0],fields[1],fields[2],fields[3]};
            require(wow64_umd_probe::install_native_compat_view(job.at("ntdll_timestamp"),offsets),"Native compatibility view unavailable");
            report["native_appcompat_before"]=std::vector<uint64_t>(wow64_umd_probe::compat_before,wow64_umd_probe::compat_before+4);
            report["native_appcompat_view"]=true;
        }
        wow64_runtime_diagnostics::inspect_threads=job.value("inspect_threads",false);
        wow64_runtime_diagnostics::wrap_threads=job.value("wrap_driver_threads",false);
        if(job.value("repair_driver_tls",false) && wow64_umd_probe::gdi) {
            auto ntdll=GetModuleHandleW(L"ntdll.dll");
            auto base=reinterpret_cast<const unsigned char*>(ntdll);
            auto dos=reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
            auto pe=reinterpret_cast<const IMAGE_NT_HEADERS64*>(base+dos->e_lfanew);
            const auto allocate=job.at("tls_allocate_rva").get<unsigned>();
            const auto release=job.at("tls_free_rva").get<unsigned>();
            require(pe->FileHeader.Machine==IMAGE_FILE_MACHINE_AMD64 && pe->FileHeader.TimeDateStamp==job.at("ntdll_timestamp").get<unsigned>() &&
                allocate>0 && release>0 && allocate<pe->OptionalHeader.SizeOfImage && release<pe->OptionalHeader.SizeOfImage,"TLS allocator image mismatch");
            wow64_runtime_diagnostics::allocate_tls=reinterpret_cast<wow64_runtime_diagnostics::AllocateTls>(uint64_t(ntdll)+allocate);
            wow64_runtime_diagnostics::free_tls=reinterpret_cast<wow64_runtime_diagnostics::FreeTls>(uint64_t(ntdll)+release);
            wow64_runtime_diagnostics::repair_thread_tls=true;
            wow64_runtime_diagnostics::wrap_threads=true;
        }
        report["repair_driver_tls"]=wow64_runtime_diagnostics::repair_thread_tls;
        wow64_runtime_diagnostics::retry_native_reservation=job.value("retry_native_reservation",false) && wow64_umd_probe::gdi;
        report["retry_native_reservation"]=wow64_runtime_diagnostics::retry_native_reservation;
        wow64_runtime_diagnostics::native_registry=job.value("native_registry",false);
        wow64_umd_probe::force_native_identity=job.value("force_native_identity",false);
        report["native_registry"]=wow64_runtime_diagnostics::native_registry;
        wow64_umd_probe::bridge_dc_creation=job.value("bridge_create_dc",false);
        report["bridge_create_dc"]=wow64_umd_probe::bridge_dc_creation;
        wow64_umd_probe::bridge_display_gdi=job.value("bridge_display_gdi",false);
        report["bridge_display_gdi"]=wow64_umd_probe::bridge_display_gdi;
        wow64_umd_probe::bridge_kmt_escape=job.value("bridge_kmt_escape",false);
        report["bridge_kmt_escape"]=wow64_umd_probe::bridge_kmt_escape;
        if(wow64_umd_probe::bridge_display_gdi && wow64_umd_probe::gdi)
            require(wow64_umd_probe::install_display_exports(),"Native display export bridge unavailable");
        if(job.value("early_native_identity",false) && wow64_umd_probe::gdi) {
            require(wow64_umd_probe::install_identity_exports(),"Native identity export unavailable");
            report["early_native_identity"]=true;
        }
        if(job.value("native_main_image",false) || job.value("native_peb_image",false)) {
            auto image=fs::u8path(job.at("native_executable").get<std::string>());
            require(wow64_umd_probe::x64_image(image.c_str()),"Native control image machine mismatch");
            wow64_umd_probe::native_main_image=LoadLibraryExW(image.c_str(),nullptr,DONT_RESOLVE_DLL_REFERENCES);
            require(wow64_umd_probe::native_main_image!=nullptr,"Native control image mapping failed");
            report["native_main_image"]=true;
            if(job.value("native_peb_image",false) && wow64_umd_probe::gdi) {
                const auto peb=__readgsqword(0x60);
                require(peb!=0 && job.value("peb_image_offset",0u)==0x10u,"Native PEB image offset not verified");
                native_image_slot=reinterpret_cast<uint64_t*>(peb+0x10);
                original_native_image=*native_image_slot;
                report["original_native_image"]=original_native_image;
                *native_image_slot=uint64_t(wow64_umd_probe::native_main_image);
                report["native_peb_image"]=true;
            }
        }
        if(job.value("cuda_no_preload",false)) {
            require(SetEnvironmentVariableW(L"CUDA_FORCE_PRELOAD_LIBRARIES",L"0")!=0,"CUDA environment");
            report["cuda_no_preload"]=true;
        }
        if(job.value("cuda_no_jit",false)) {
            require(SetEnvironmentVariableW(L"CUDA_DISABLE_JIT",L"1")!=0,"CUDA environment");
            report["cuda_no_jit"]=true;
        }
        const unsigned width=job.value("width",640u), height=job.value("height",384u);
        require(width && height && width<=1920 && height<=1080,"Probe extent refused");
        const bool benchmark=job.value("benchmark",false);
        require(!benchmark||(width==1920&&height==1080),"Benchmark extent must be 1080p");
        save("driver_setup");
        if (wow64_umd_probe::gdi) {
            require(wow64_umd_probe::install()!=0,"UMD shim unavailable");
            require(wow64_umd_probe::install_gdi_export(),"Native GDI export bridge unavailable");
        }
        auto initialize_cuda=[&]() {
            if(job.value("preload_driver_object_loader",false)) {
                const auto object_path=fs::u8path(job.at("cuda_library").get<std::string>()).parent_path()/L"nvobjectloader64.dll";
                auto object=LoadLibraryExW(object_path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
                report["object_loader_preloaded"]=object!=nullptr;
                report["object_loader_load_error"]=object?0:GetLastError();
                require(object!=nullptr,"Native driver object loader preload failed");
            }
            save("cuda_load");
            auto cuda=LoadLibraryW(fs::u8path(job.at("cuda_library").get<std::string>()).c_str());
            require(cuda!=nullptr,"CUDA library load");
            wow64_umd_probe::owner=GetCurrentThreadId(); wow64_umd_probe::patch(cuda);
            auto init=symbol<int(*)(unsigned)>(cuda,"cuInit");
            save("cuda_initialize"); const auto status=init(0); report["cuda_status"]=status;
            using DumpLogs=int(*)(unsigned*,char*,size_t*,unsigned);
            if(auto logs=reinterpret_cast<DumpLogs>(GetProcAddress(cuda,"cuLogsDumpToMemory"))) {
                char buffer[25600]{}; size_t bytes=sizeof(buffer);
                report["cuda_logs_status"]=logs(nullptr,buffer,&bytes,0);
                if(bytes<sizeof(buffer)) report["cuda_logs"]=std::string(buffer,bytes);
            }
            auto name=symbol<int(*)(int,const char**)>(cuda,"cuGetErrorName"); const char* text=nullptr;
            if(name(status,&text)==0 && text) report["cuda_error_name"]=text;
            save("cuda_result"); require(status==0,"CUDA initialization refused");
        };
        if(job.value("cuda_first",false)) { report["cuda_before_d3d12"]=true; initialize_cuda(); }
        {
            Device gpu; save("device_create"); gpu.create(); report["device_and_queue"]=true;
            save("gpu_copy"); gpu.copy_control(); report["gpu_copy_verified"]=true;
            // Direct CUDA initialization exposes the error hidden by NVAPI_ERROR.
            if (job.value("cuda_diagnostic",false)) {
                SYSTEM_INFO system{}, native{}; GetSystemInfo(&system); GetNativeSystemInfo(&native);
                report["system_maximum_address"]=uint64_t(system.lpMaximumApplicationAddress);
                report["native_maximum_address"]=uint64_t(native.lpMaximumApplicationAddress);
                auto high=VirtualAlloc(reinterpret_cast<void*>(0x1000000000ull),65536,MEM_RESERVE,PAGE_NOACCESS);
                report["high_address_reservation"]=uint64_t(high);
                report["high_address_error"]=high?0u:GetLastError();
                if(high) VirtualFree(high,0,MEM_RELEASE);
                if(!job.value("cuda_first",false)) initialize_cuda();
            } else if (backend=="legacy") {
                save("legacy_nvapi");
                auto library=LoadLibraryExW(L"nvapi64.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
                require(library!=nullptr,"NVAPI library load"); wow64_umd_probe::patch(library);
                auto query=symbol<void*(__cdecl*)(unsigned)>(library,"nvapi_QueryInterface");
                auto initialize=reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150e828));
                auto create=reinterpret_cast<decltype(&NvAPI_D3D12_CreateCubinComputeShaderWithName)>(query(0x1dc7261f));
                auto destroy=reinterpret_cast<decltype(&NvAPI_D3D12_DestroyCubinComputeShader)>(query(0x7fb785ba));
                require(initialize && create && destroy,"Legacy shader API unavailable");
                wow64_umd_probe::patch(GetModuleHandleW(L"nvapi64_impl.dll"));
                const auto status=initialize(); report["nvapi_initialize_status"]=int(status);
                require(status==NVAPI_OK,"NVAPI initialization refused");
                std::ifstream kernel(fs::u8path(job.at("kernel_library").get<std::string>()),std::ios::binary|std::ios::ate);
                require(bool(kernel)&&kernel.tellg()>0&&kernel.tellg()<16*1024*1024,"Kernel file size");
                std::vector<char> blob(size_t(kernel.tellg()),0); kernel.seekg(0);
                require(bool(kernel.read(blob.data(),blob.size())),"Kernel file read");
                const auto block=job.at("kernel_block").get<std::vector<unsigned>>();
                require(block.size()==3,"Kernel block shape");
                const auto entry=job.at("kernel_entry").get<std::string>();
                NVDX_ObjectHandle shader=nullptr; save("legacy_shader_create");
                const auto created=create(gpu.device.Get(),blob.data(),NvU32(blob.size()),block[0],block[1],block[2],entry.c_str(),&shader);
                report["shader_create_status"]=int(created); report["shader_created"]=shader!=nullptr;
                save("legacy_shader_result"); require(created==NVAPI_OK && shader,"Legacy shader creation refused");
                require(destroy(gpu.device.Get(),shader)==NVAPI_OK,"Legacy shader destruction refused");
            } else if (backend=="phoenix") {
                save("phoenix_load");
                auto library=LoadLibraryW(fs::u8path(job.at("native_library").get<std::string>()).c_str());
                require(library!=nullptr,"Phoenix library load");
                auto create=symbol<decltype(&phoenix_native_create)>(library,"phoenix_native_create");
                auto destroy=symbol<decltype(&phoenix_native_destroy)>(library,"phoenix_native_destroy");
                auto frame=symbol<decltype(&phoenix_native_frame_host)>(library,"phoenix_native_frame_host");
                auto error=symbol<decltype(&phoenix_native_error)>(library,"phoenix_native_error");
                auto defaults=symbol<decltype(&phoenix_native_default_controls)>(library,"phoenix_native_default_controls");
                const auto config=job.at("config").dump();
                save("phoenix_create"); auto session=create(config.c_str(),gpu.device.Get(),gpu.queue.Get());
                require(session!=nullptr,error());
                try {
                    std::vector<uint16_t> color(size_t(width)*height*4), enhanced(color.size(),0x7e00);
                    for (size_t i=0;i<color.size();++i) color[i]=(i%4==3)?0x3c00:uint16_t(0x3000+(i*19)%0x800);
                    PhoenixNativeControls controls{}; defaults(&controls); controls.reset=1;
                    for(unsigned iteration=0;iteration<3;++iteration) {
                    std::fill(enhanced.begin(),enhanced.end(),uint16_t(0x7e00));
                    save("phoenix_frame");
                    require(frame(session,color.data(),color.size()*2,width*8,nullptr,0,0,
                        enhanced.data(),enhanced.size()*2,width*8,&controls)==0,error());
                    size_t changed=0;
                    for (size_t i=0;i<enhanced.size();++i) {
                        require((enhanced[i]&0x7c00)!=0x7c00,"Nonfinite Phoenix output");
                        if (i%4!=3) changed+=enhanced[i]!=color[i];
                    }
                    require(changed!=0,"No changed Phoenix output");
                    report["frames"].push_back({{"changed_rgb_samples",changed},{"output_sha256",output_sha256(enhanced)}});
                    }
                    if(benchmark) { save("phoenix_benchmark"); report["benchmark"]=benchmark_phoenix(gpu,session,library,width,height,controls,job.value("benchmark_reset",false)); }
                } catch (...) { destroy(session); throw; }
                require(destroy(session)==0,error());
            } else if (backend=="nvidia") {
                save("ngx_load");
                auto core=LoadLibraryW(fs::u8path(job.at("ngx_core").get<std::string>()).c_str());
                require(core!=nullptr,"NGX core load");
                auto alloc=symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**)>(core,"NVSDK_NGX_D3D12_AllocateParameters");
                auto free_params=symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*)>(core,"NVSDK_NGX_D3D12_DestroyParameters");
                NVSDK_NGX_Parameter* params=nullptr;
                report["allocate_status"]=uint32_t(alloc(&params)); require(params!=nullptr,"NGX parameters");
                require(PhoenixNrLoad(fs::u8path(job.at("model_library").get<std::string>()).c_str())!=0,"NVIDIA library load");
                save("ngx_initialize");
                const auto init=PhoenixNrInit(0x1000000ull,output.parent_path().c_str(),gpu.device.Get(),NVSDK_NGX_Version_API,params);
                report["initialize_status"]=uint32_t(init); save("ngx_initialize_result");
                require(NVSDK_NGX_SUCCEED(init),"NVIDIA initialization refused");
                params->Reset();
                for (const auto* key:{"DLSSNR.Width","DLSSNR.InputWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"}) params->Set(key,width);
                for (const auto* key:{"DLSSNR.Height","DLSSNR.InputHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"}) params->Set(key,height);
                for (const auto* key:{"CreationNodeMask","VisibilityNodeMask","DLSSNR.SDR","DLSSNR.AutoExposure"}) params->Set(key,1u);
                for (const auto* key:{"DLSSNR.Upscaling","PerfQualityValue","DLSSNR.Hdr","DLSS.Feature.Create.Flags","DLSSNR.Hint.Render.Preset"}) params->Set(key,0u);
                for (const auto* key:{"DLSSNR.Scale","DLSSNR.ScalingRatio","MV.Scale.X","MV.Scale.Y","InPreExposure","InExposureScale"}) params->Set(key,1.f);
                NVSDK_NGX_Handle* handle=nullptr; save("ngx_create");
                auto created=PhoenixNrCreate(gpu.list.Get(),static_cast<NVSDK_NGX_Feature>(18),params,&handle);
                report["create_status"]=uint32_t(created); save("ngx_create_result");
                require(NVSDK_NGX_SUCCEED(created)&&handle,"NVIDIA feature creation refused"); gpu.submit();
                // Real evaluation follows the same post-render recipe as the addon.
                auto color=gpu.texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
                auto motion=gpu.texture(width,height,DXGI_FORMAT_R16G16_FLOAT);
                auto enhanced=gpu.texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{},mf{}; uint64_t bytes=0,mbytes=0;
                auto desc=color->GetDesc(); gpu.device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
                desc=motion->GetDesc(); gpu.device->GetCopyableFootprints(&desc,0,1,0,&mf,nullptr,nullptr,&mbytes);
                auto upload=gpu.buffer(bytes,D3D12_HEAP_TYPE_UPLOAD), mv=gpu.buffer(mbytes,D3D12_HEAP_TYPE_UPLOAD);
                void* data=nullptr; D3D12_RANGE empty{}; hr(upload->Map(0,&empty,&data));
                for (unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
                    auto pixel=reinterpret_cast<uint16_t*>(static_cast<char*>(data)+fp.Offset+y*fp.Footprint.RowPitch+x*8);
                    for(unsigned c=0;c<4;++c) pixel[c]=c==3?0x3c00:uint16_t(0x3000+(x*11+y*7+c*17)%0x800);
                }
                upload->Unmap(0,nullptr); hr(mv->Map(0,&empty,&data)); memset(data,0,size_t(mbytes)); mv->Unmap(0,nullptr);
                auto copy=[&](ID3D12Resource* texture,ID3D12Resource* source,D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint) {
                    D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=texture; a.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    b.pResource=source; b.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; b.PlacedFootprint=footprint;
                    gpu.list->CopyTextureRegion(&a,0,0,0,&b,nullptr);
                    gpu.barrier(texture,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                };
                copy(color.Get(),upload.Get(),fp); copy(motion.Get(),mv.Get(),mf); gpu.submit();
                // Seed with half NaNs so a successful API call without GPU writes cannot pass.
                auto poison=gpu.buffer(bytes,D3D12_HEAP_TYPE_UPLOAD); hr(poison->Map(0,&empty,&data));
                for(size_t i=0;i<size_t(bytes)/2;++i) static_cast<uint16_t*>(data)[i]=0x7e00;
                poison->Unmap(0,nullptr);
                gpu.barrier(enhanced.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
                copy(enhanced.Get(),poison.Get(),fp);
                gpu.barrier(enhanced.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                gpu.submit();
                ComPtr<ID3D12QueryHeap> queries; ComPtr<ID3D12Resource> timestamps; uint64_t frequency=0;
                if(benchmark) {
                    D3D12_QUERY_HEAP_DESC query_desc{}; query_desc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; query_desc.Count=2;
                    hr(gpu.device->CreateQueryHeap(&query_desc,IID_PPV_ARGS(&queries)));
                    timestamps=gpu.buffer(16,D3D12_HEAP_TYPE_READBACK); hr(gpu.queue->GetTimestampFrequency(&frequency));
                    require(frequency>0,"Invalid GPU timestamp frequency");
                }
                unsigned evaluations=0;
                auto evaluate=[&]() {
                    const auto started=Clock::now(); const auto f=evaluations++;
                    params->Reset(); params->Set("DLSSNR.Color",color.Get()); params->Set("Color",color.Get());
                    params->Set("DLSSNR.Output",enhanced.Get()); params->Set("Output",enhanced.Get());
                    for(const auto* key:{"DLSSNR.MVec","MVec","MotionVectors"}) params->Set(key,motion.Get());
                    for(const auto* key:{"DLSSNR.ColorSubrectWidth","DLSSNR.MVecSubrectWidth","DLSSNR.OutputSubrectWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"}) params->Set(key,width);
                    for(const auto* key:{"DLSSNR.ColorSubrectHeight","DLSSNR.MVecSubrectHeight","DLSSNR.OutputSubrectHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"}) params->Set(key,height);
                    for(const auto* key:{"DLSSNR.MVecScaleX","DLSSNR.MVecScaleY","DLSSNR.ScalingRatio","DLSSNR.Scale","DLSSNR.Intensity","DLSSNR.LocalToneStrength","DLSSNR.LocalStructureStrength","DLSSNR.GlobalToneStrength","DLSS.Pre.Exposure","DLSS.Exposure.Scale"}) params->Set(key,1.f);
                    params->Set("DLSSNR.SkinStructureStrength",-1.f); params->Set("DLSSNR.SDR",1u);
                    params->Set("DLSSNR.Enabled",1u); params->Set("DLSSNR.Reset",unsigned(f==0||job.value("benchmark_reset",false)));
                    if(benchmark) gpu.list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
                    const auto evaluated=PhoenixNrEvaluate(gpu.list.Get(),handle,params,nullptr);
                    if(benchmark) {
                        gpu.list->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
                        gpu.list->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,timestamps.Get(),0);
                    }
                    report["evaluate_status"]=uint32_t(evaluated);
                    require(NVSDK_NGX_SUCCEED(evaluated),"NVIDIA evaluation refused"); gpu.submit();
                    const auto wall=milliseconds(started); double gpu_ms=0;
                    if(benchmark) {
                        uint64_t* values=nullptr; D3D12_RANGE range{0,16}; hr(timestamps->Map(0,&range,reinterpret_cast<void**>(&values)));
                        require(values[1]>values[0],"Invalid GPU timestamp order"); gpu_ms=double(values[1]-values[0])*1000.0/frequency;
                        timestamps->Unmap(0,&empty);
                    }
                    return std::make_pair(gpu_ms,wall);
                };
                save("ngx_evaluate"); for(unsigned f=0;f<3;++f) evaluate();
                if(benchmark) { save("ngx_benchmark"); report["benchmark"]=benchmark_frames(evaluate); }
                auto readback=gpu.buffer(bytes,D3D12_HEAP_TYPE_READBACK);
                gpu.barrier(enhanced.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION a{},b{}; a.pResource=readback.Get(); a.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; a.PlacedFootprint=fp;
                b.pResource=enhanced.Get(); b.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                gpu.list->CopyTextureRegion(&a,0,0,0,&b,nullptr); gpu.submit();
                D3D12_RANGE all{0,size_t(bytes)}; hr(readback->Map(0,&all,&data)); size_t changed=0;
                std::vector<uint16_t> pixel_output(size_t(width)*height*3);
                for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
                    auto pixel=reinterpret_cast<const uint16_t*>(static_cast<char*>(data)+fp.Offset+y*fp.Footprint.RowPitch+x*8);
                    for(unsigned c=0;c<3;++c) { require((pixel[c]&0x7c00)!=0x7c00,"Nonfinite NVIDIA output"); changed+=pixel[c]!=uint16_t(0x3000+(x*11+y*7+c*17)%0x800); pixel_output[(size_t(y)*width+x)*3+c]=pixel[c]; }
                }
                readback->Unmap(0,&empty); require(changed!=0,"No changed NVIDIA output");
                report["frames"].push_back({{"changed_rgb_samples",changed},{"evaluations",evaluations},{"output_sha256",output_sha256(pixel_output)}});
                if(benchmark) {
                    report["benchmark"]["output_sha256"]=output_sha256(pixel_output);
                    report["benchmark"]["changed_rgb_samples"]=changed;
                    report["benchmark"]["gpu_timing_scope"]="D3D12 GPU timestamps around NVIDIA evaluation";
                    report["benchmark"]["reset_each_frame"]=job.value("benchmark_reset",false);
                }
                PhoenixNrRelease(handle); free_params(params);
            } else throw std::runtime_error("Unknown backend");
        }
        report["passed"]=true; save("complete");
    } catch(const std::exception& error) {
        report["failed_stage"]=report.value("stage",""); report["error"]=error.what(); save("failed");
    }
    report["umd_redirects"]=wow64_umd_probe::redirects;
    report["shim_slots"]=wow64_umd_probe::slot_count;
    report["identity_queries"]=wow64_umd_probe::identity_queries;
    report["main_image_queries"]=LONG(wow64_umd_probe::main_image_queries);
    report["driver_packet_headers"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_umd_probe::packet_count) && i<16;++i) {
        const auto& packet=wow64_umd_probe::packets[i];
        Json before=Json::array(),after=Json::array();
        for(unsigned w=0;w<packet.words;++w) { before.push_back(packet.before[w]); after.push_back(packet.after[w]); }
        report["driver_packet_headers"].push_back({{"type",packet.code},{"bytes",packet.bytes},
            {"adapter",packet.adapter},{"device",packet.device},{"flags",packet.flags},
            {"before",before},{"after",after}});
    }
    report["heap_calls"]=LONG(wow64_runtime_diagnostics::heap_calls);
    report["registry_calls"]=LONG(wow64_runtime_diagnostics::registry_calls);
    report["file_mappings"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::mapping_count) && i<32;++i) {
        const auto& value=wow64_runtime_diagnostics::mappings[i];
        report["file_mappings"].push_back({{"bytes",value.bytes},{"requested",value.requested},
            {"returned",value.returned},{"error",value.error}});
    }
    report["gdi_calls"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::gdi_count) && i<64;++i) {
        const auto& value=wow64_runtime_diagnostics::gdi_calls[i];
        report["gdi_calls"].push_back({{"operation",value.operation},{"handle",value.handle},
            {"argument",value.argument},{"result",value.result},{"error",value.error}});
    }
    report["heap_failures"]=Json::array();
    report["dynamic_tls"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::slot_count) && i<64;++i) {
        const auto& value=wow64_runtime_diagnostics::slot_calls[i];
        report["dynamic_tls"].push_back({{"operation",value.operation},{"slot",value.slot},
            {"success",value.success!=0},{"error",value.error}});
    }
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::failure_count) && i<32;++i) {
        const auto& value=wow64_runtime_diagnostics::failures[i];
        report["heap_failures"].push_back({{"operation",value.operation},{"bytes",value.bytes},{"flags",value.flags},{"error",value.error}});
    }
    report["threads"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::thread_count) && i<32;++i) {
        const auto& value=wow64_runtime_diagnostics::threads[i];
        report["threads"].push_back({{"start",value.start},{"stack",value.stack},{"handle",value.handle},
            {"flags",value.flags},{"error",value.error},{"instruction",value.instruction},
            {"entered",LONG(value.entered)},{"entered_teb",value.entered_teb},{"tls_vector",value.tls_vector},
            {"tls_after",value.tls_after},{"tls_status",value.tls_status},
            {"exit_code",value.exit_code},
            {"code_segment",value.segment},{"context_error",value.context_error}});
    }
    report["kmt_calls"]=Json::array();
    report["native_reservations"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_runtime_diagnostics::reservation_count) && i<32;++i) {
        const auto& value=wow64_runtime_diagnostics::reservations[i];
        report["native_reservations"].push_back({{"bytes",value.bytes},{"requested",value.requested},
            {"initial",value.initial},{"retried",value.retried},{"flags",value.flags},{"protect",value.protect},
            {"error",value.error},{"retry_error",value.retry_error},{"parameters",value.parameters},
            {"retry_method",value.method},{"reservation_verified",value.verified}});
    }
    report["driver_library_loads"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_umd_probe::library_load_count) && i<96;++i) {
        const auto& value=wow64_umd_probe::library_loads[i];
        report["driver_library_loads"].push_back({{"path",fs::path(value.path).generic_string()},
            {"loaded",value.module!=0},{"flags",value.flags},{"error",value.error}});
    }
    report["adapter_queries"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_umd_probe::adapter_query_count) && i<128;++i) {
        const auto& value=wow64_umd_probe::adapter_queries[i];
        report["adapter_queries"].push_back({{"type",value.type},{"bytes",value.bytes},{"status",uint32_t(value.status)},
            {"words",std::vector<DWORD>(value.words,value.words+16)}});
    }
    report["kmt_observed"]=Json::object();
    report["bridged_escape_count"]=LONG(wow64_umd_probe::bridged_escape_count);
    for(unsigned i=0;i<11;++i) if(wow64_umd_probe::kmt_observed[i])
        report["kmt_observed"][wow64_umd_probe::kmt_names[i]]=LONG(wow64_umd_probe::kmt_observed[i]);
    for(unsigned i=0;i<unsigned(wow64_umd_probe::kmt_count) && i<128;++i) {
        const auto& call=wow64_umd_probe::kmt_calls[i];
        report["kmt_calls"].push_back({{"name",wow64_umd_probe::kmt_names[call.function]},
            {"status",uint32_t(call.status)}});
    }
    report["ioctls"]=Json::array();
    for(unsigned i=0;i<wow64_umd_probe::ioctl_count;++i) {
        const auto& call=wow64_umd_probe::ioctls[i];
        report["ioctls"].push_back({{"code",call.code},{"input_bytes",call.input_bytes},
            {"output_bytes",call.output_bytes},{"success",call.result!=0},{"error",call.error}});
    }
    report["cuda_allocations"]=Json::array();
    for(unsigned i=0;i<wow64_umd_probe::allocation_count;++i) {
        const auto& a=wow64_umd_probe::allocations[i];
        report["cuda_allocations"].push_back({{"requested",a.requested},{"bytes",a.bytes},
            {"returned",a.returned},{"flags",a.flags},{"error",a.error}});
    }
    wow64_umd_probe::restore();
    wow64_umd_probe::restore_gdi_export();
    wow64_umd_probe::restore_identity_exports();
    wow64_umd_probe::restore_display_exports();
    const auto compat_restored=wow64_umd_probe::restore_native_compat_view();
    if(report.value("native_appcompat_view",false)) report["native_appcompat_restored"]=compat_restored;
    if(native_image_slot) *native_image_slot=original_native_image;
    save(report.value("passed",false)?"complete":"failed");
    return report.value("passed",false)?0:2;
}
