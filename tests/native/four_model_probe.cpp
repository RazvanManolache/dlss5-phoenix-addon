// Explicit offline GPU comparison. Same queue, inputs and timestamp boundaries
// for Phoenix's production recorder and NVIDIA NGX feature 18 presets 1 and 2.
#include "../../src/native/support/dispatch.h"
#include "../../src/native/runtime/texture_binding.h"
#include "../../native_runtime/network_schedule.h"
#include <nvsdk_ngx.h>
#include <MinHook.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <psapi.h>

using Json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static void require(bool good, const std::string& message) { if (!good) throw std::runtime_error(message); }
static double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
static std::string digest(const fs::path& path) {
    const auto bytes=read(path); return phoenix_model::weight_digest(bytes.data(),bytes.size());
}
template<class T> static T symbol(HMODULE module,const char* name) {
    auto function=reinterpret_cast<T>(GetProcAddress(module,name));require(function!=nullptr,name);return function;
}
using LaunchFn=decltype(&NvAPI_D3D12_LaunchCuKernelChain);
static LaunchFn real_launch=nullptr;
static thread_local bool counting=false;
static thread_local unsigned launch_count=0;
// Offline-only deep markers: the production observer remains untouched.
static thread_local ID3D12QueryHeap* diagnostic_heap=nullptr;
static thread_local ID3D12GraphicsCommandList* diagnostic_command=nullptr;
static thread_local unsigned diagnostic_last_group=0;
static void diagnostic_progress(const char* stage){
    if(diagnostic_heap && std::string_view(stage)=="network_first_dispatch")
        diagnostic_command->EndQuery(diagnostic_heap,D3D12_QUERY_TYPE_TIMESTAMP,6);
}
static void diagnostic_group(ID3D12GraphicsCommandList* command,unsigned group){
    if(diagnostic_heap && group==diagnostic_last_group)
        command->EndQuery(diagnostic_heap,D3D12_QUERY_TYPE_TIMESTAMP,5);
}
static NvAPI_Status __cdecl count_launch(ID3D12GraphicsCommandList* cmd,const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches,NvU32 count) {
    if(counting && !launch_count && count && diagnostic_heap)
        cmd->EndQuery(diagnostic_heap,D3D12_QUERY_TYPE_TIMESTAMP,6);
    if(counting) launch_count+=count;
    return real_launch(cmd,launches,count);
}
static Json stats(std::vector<double> values) {
    require(!values.empty(),"No measured frames");std::sort(values.begin(),values.end());
    return {{"samples",values.size()},{"mean_ms",std::accumulate(values.begin(),values.end(),0.)/values.size()},
        {"median_ms",values[values.size()/2]},{"p95_ms",values[size_t(.95*(values.size()-1))]},
        {"min_ms",values.front()},{"max_ms",values.back()}};
}
struct Gpu {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device; ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> command;
    ComPtr<ID3D12Fence> fence;ComPtr<ID3D12QueryHeap> queries;ComPtr<ID3D12Resource> stamps;
    uint64_t serial=0,frequency=0;
    unsigned query_count=2;
    std::array<unsigned,7> query_order{{0,2,3,4,1,0,0}};
    std::array<double,6> last_segments{};
    explicit Gpu(bool debug,bool profile=false,bool deep=false) : query_count(deep?7:profile?5:2) {
        if(deep)query_order={{0,6,5,2,3,4,1}};
        if(debug){ComPtr<ID3D12Debug> layer;check(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)));layer->EnableDebugLayer();}
        ComPtr<IDXGIFactory6> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for(unsigned index=0;;++index){ComPtr<IDXGIAdapter1> item;
            if(factory->EnumAdapterByGpuPreference(index,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&item))==DXGI_ERROR_NOT_FOUND)break;
            DXGI_ADAPTER_DESC1 desc{};check(item->GetDesc1(&desc));if(desc.VendorId==0x10de){adapter=item;break;}}
        require(adapter!=nullptr,"NVIDIA adapter required");check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC desc{};check(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)));
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&command)));check(command->Close());
        check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));check(queue->GetTimestampFrequency(&frequency));
        D3D12_QUERY_HEAP_DESC query{D3D12_QUERY_HEAP_TYPE_TIMESTAMP,query_count,0};check(device->CreateQueryHeap(&query,IID_PPV_ARGS(&queries)));
        stamps=phoenix_native::buffer(device.Get(),query_count*sizeof(uint64_t),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void begin(){check(allocator->Reset());check(command->Reset(allocator.Get(),nullptr));}
    void submit(){check(command->Close());ID3D12CommandList* lists[]{command.Get()};queue->ExecuteCommandLists(1,lists);
        check(queue->Signal(fence.Get(),++serial));HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);require(event!=nullptr,"Fence event");
        check(fence->SetEventOnCompletion(serial,event));const auto result=WaitForSingleObject(event,30000);CloseHandle(event);
        require(result==WAIT_OBJECT_0,"GPU fence timeout");require(fence->GetCompletedValue()!=UINT64_MAX,"GPU device removed");}
    void mark(unsigned index){command->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,index);}
    double timing(){
        uint64_t* data=nullptr;check(stamps->Map(0,nullptr,reinterpret_cast<void**>(&data)));
        const auto ticks=data[1]-data[0];
        if(query_count>2){
            // Heap slots preserve the original 0/1 total-time ABI; chronological
            // order is explicit; deep mode subdivides engine work before slot 2.
            uint64_t sum=0;
            for(unsigned i=0;i+1<query_count;++i){require(data[query_order[i+1]]>=data[query_order[i]],"Nonmonotonic GPU timestamp");
                const auto elapsed=data[query_order[i+1]]-data[query_order[i]];sum+=elapsed;
                last_segments[i]=double(elapsed)*1000/frequency;}
            require(sum==ticks,"GPU segment accounting mismatch");
        }
        stamps->Unmap(0,nullptr);return double(ticks)*1000/frequency;
    }
    Json calibrate_timestamps(){
        require(query_count>2,"Calibration requires diagnostic timestamps");
        std::vector<double> totals;
        for(unsigned i=0;i<80;++i){begin();for(unsigned index=0;index<query_count;++index)mark(query_order[index]);
            command->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,query_count,stamps.Get(),0);
            submit();const auto value=timing();if(i>=16)totals.push_back(value);}
        return {{"gpu",stats(totals)},{"scope","Adjacent diagnostic timestamps only, after neural warmup; not model or copy work"},
                {"timestamp_count",query_count},
                {"timestamp_frequency",frequency}};
    }
    Json memory() {
        ComPtr<IDXGIAdapter3> measured;check(adapter.As(&measured));DXGI_QUERY_VIDEO_MEMORY_INFO local{},nonlocal{};
        check(measured->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local));
        check(measured->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nonlocal));
        PROCESS_MEMORY_COUNTERS_EX process{};process.cb=sizeof(process);
        require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process),sizeof(process)),"Process memory query");
        return {{"device_local_bytes",local.CurrentUsage},{"device_nonlocal_bytes",nonlocal.CurrentUsage},
                {"host_private_bytes",process.PrivateUsage},{"host_working_set_bytes",process.WorkingSetSize},
                {"host_peak_working_set_bytes",process.PeakWorkingSetSize},
                {"scope","Current-process DXGI residency and process RAM; includes harness/driver caches, not vendor internal-buffer accounting"}};
    }
    unsigned errors(){ComPtr<ID3D12InfoQueue> info;if(FAILED(device.As(&info)))return 0;unsigned count=0;
        for(uint64_t i=0;i<info->GetNumStoredMessages();++i){SIZE_T size=0;info->GetMessage(i,nullptr,&size);std::vector<uint8_t> bytes(size);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());check(info->GetMessage(i,message,&size));
            if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR){++count;std::cerr<<message->pDescription<<'\n';}}
        return count;}
};
struct Nvidia {
    using Init=NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long,const wchar_t*,ID3D12Device*,NVSDK_NGX_Version,const NVSDK_NGX_Parameter*);
    using Create=NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,NVSDK_NGX_Feature,NVSDK_NGX_Parameter*,NVSDK_NGX_Handle**);
    using Evaluate=NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,const NVSDK_NGX_Handle*,const NVSDK_NGX_Parameter*,PFN_NVSDK_NGX_ProgressCallback);
    HMODULE core=nullptr,forwarder=nullptr;NVSDK_NGX_Parameter* params=nullptr;NVSDK_NGX_Handle* handle=nullptr;
    Evaluate evaluate=nullptr;
    static void success(NVSDK_NGX_Result code,const char* operation){if(!NVSDK_NGX_SUCCEED(code))throw std::runtime_error(std::string(operation)+" NGX result "+std::to_string(uint32_t(code)));}
    Nvidia(Gpu& gpu,const Json& config,unsigned w,unsigned h,unsigned preset){
        core=LoadLibraryExW(fs::u8path(config.at("ngx_core").get<std::string>()).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);require(core!=nullptr,"NGX core load");
        auto allocate=symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**)>(core,"NVSDK_NGX_D3D12_AllocateParameters");success(allocate(&params),"Allocate parameters");
        forwarder=LoadLibraryExW(fs::u8path(config.at("forwarder").get<std::string>()).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);require(forwarder!=nullptr,"NR forwarder load");
        auto load=symbol<BOOL(WINAPI*)(const wchar_t*)>(forwarder,"PhoenixNrLoad");require(load(fs::u8path(config.at("weight_source").at("path").get<std::string>()).c_str()),"NR DLL load");
        const auto cache=fs::u8path(config.at("cache").get<std::string>());fs::create_directories(cache);
        success(symbol<Init>(forwarder,"PhoenixNrInit")(0x1000000ull,cache.c_str(),gpu.device.Get(),NVSDK_NGX_Version_API,params),"Init NR");
        params->Reset();
        for(auto key:{"DLSSNR.Width","DLSSNR.InputWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"})params->Set(key,w);
        for(auto key:{"DLSSNR.Height","DLSSNR.InputHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"})params->Set(key,h);
        for(auto key:{"CreationNodeMask","VisibilityNodeMask","DLSSNR.SDR","DLSSNR.AutoExposure"})params->Set(key,1u);
        for(auto key:{"DLSSNR.Upscaling","PerfQualityValue","DLSSNR.Hdr","DLSS.Feature.Create.Flags","DLSSNR.Style","DLSSNR.UICorrection","DLSSNR.DepthInverted"})params->Set(key,0u);
        params->Set("DLSSNR.Hint.Render.Preset",preset);
        for(auto key:{"DLSSNR.Scale","DLSSNR.ScalingRatio","MV.Scale.X","MV.Scale.Y","InPreExposure","InExposureScale","DLSSNR.Intensity","DLSSNR.GlobalToneStrength","DLSSNR.LocalToneStrength","DLSSNR.LocalStructureStrength"})params->Set(key,1.f);
        params->Set("DLSSNR.UseAutoMask",1u);params->Set("DLSSNR.SkinStructureStrength",-1.f);
        gpu.begin();success(symbol<Create>(forwarder,"PhoenixNrCreate")(gpu.command.Get(),static_cast<NVSDK_NGX_Feature>(18),params,&handle),"Create NR");gpu.submit();require(handle!=nullptr,"NR feature null");
        evaluate=symbol<Evaluate>(forwarder,"PhoenixNrEvaluate");
        // Observe the driver's existing launch entry without changing packets.
        const auto nvapi=LoadLibraryExW(L"nvapi64.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);require(nvapi!=nullptr,"NVAPI load");
        auto query=symbol<void*(__cdecl*)(unsigned)>(nvapi,"nvapi_QueryInterface");auto* target=query(0x24973538);require(target!=nullptr,"NVAPI launch entry");
        require(MH_Initialize()==MH_OK,"Launch observer initialize");require(MH_CreateHook(target,reinterpret_cast<void*>(&count_launch),reinterpret_cast<void**>(&real_launch))==MH_OK,"Launch observer hook");require(MH_EnableHook(target)==MH_OK,"Launch observer enable");
    }
    void record(Gpu& gpu,Texture& color,Texture& motion,Texture& output,bool reset,unsigned w,unsigned h){
        params->Reset();
        for(auto key:{"DLSSNR.Color","Color"})params->Set(key,color.resource.Get());
        for(auto key:{"DLSSNR.Output","Output"})params->Set(key,output.resource.Get());
        for(auto key:{"DLSSNR.MVec","MVec","MotionVectors"})params->Set(key,motion.resource.Get());
        for(auto key:{"DLSSNR.ColorSubrectWidth","DLSSNR.MVecSubrectWidth","DLSSNR.OutputSubrectWidth","DLSSNR.OutputWidth","DLSSNR.Output.Width","Width"})params->Set(key,w);
        for(auto key:{"DLSSNR.ColorSubrectHeight","DLSSNR.MVecSubrectHeight","DLSSNR.OutputSubrectHeight","DLSSNR.OutputHeight","DLSSNR.Output.Height","Height"})params->Set(key,h);
        for(auto key:{"DLSSNR.MVecScaleX","DLSSNR.MVecScaleY","DLSSNR.ScalingRatio","DLSSNR.Scale","DLSSNR.Intensity","DLSSNR.GlobalToneStrength","DLSSNR.LocalToneStrength","DLSSNR.LocalStructureStrength","DLSS.Pre.Exposure","DLSS.Exposure.Scale"})params->Set(key,1.f);
        params->Set("DLSSNR.SkinStructureStrength",-1.f);params->Set("DLSSNR.UseAutoMask",1u);
        params->Set("DLSSNR.SDR",1u);params->Set("DLSSNR.Enabled",1u);params->Set("DLSSNR.Reset",unsigned(reset));params->Set("DLSSNR.Style",0u);params->Set("DLSSNR.UICorrection",0u);
        params->Set("DLSSNR.DepthInverted",0u);
        counting=true;launch_count=0;const auto result=evaluate(gpu.command.Get(),handle,params,nullptr);counting=false;success(result,"Evaluate NR");
    }
    ~Nvidia(){if(handle)symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*)>(forwarder,"PhoenixNrRelease")(handle);
        if(params)symbol<NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*)>(core,"NVSDK_NGX_D3D12_DestroyParameters")(params);
        if(real_launch){MH_DisableHook(MH_ALL_HOOKS);MH_Uninitialize();}}
};
int main(int argc,char** argv){
    Json report{{"passed",false}};fs::path destination;
    try{
        require(argc==3,"JOB_JSON OUTPUT_JSON");const auto config=phoenix_bundle::json_file(argv[1]);destination=argv[2];
        const unsigned w=config.at("width"),h=config.at("height");const bool vendor=config.at("engine")=="nvidia";const auto source=phoenix_model::WeightSource::config(config.at("weight_source"));
        if(source.disk_model())require(!vendor,"NVIDIA reference requires its DLL, not a disk-model export");
        else require(digest(source.dll)==source.dll_sha256,"Model DLL digest mismatch");
        report.update({{"engine",config.at("engine")},{"model_id",source.model_id},{"width",w},{"height",h},{"debug",config.value("debug",false)},
            {"input_sha256",digest(fs::u8path(config.at("color_file").get<std::string>()))},{"motion_sha256",digest(fs::u8path(config.at("motion_file").get<std::string>()))},
            {"model_dll_sha256",source.dll_sha256},{"captures",Json::object()}, {"gpu_scope","Complete engine recording, barriers, and matching output copy; excludes upload/readback/CPU gaps"},
            {"effective_controls",{{"auto_mask",true},{"inverted_depth",false},{"tone",1},{"structure",1},{"skin",-1},{"style",0}}},
            {"wall_scope","Command recording, execute, signal and synchronous fence wait; excludes upload/readback/status parsing"}});
        phoenix_cpp::NetworkSchedule schedule;schedule.prepare(config,source.model_id,true);
        require(!vendor || (schedule.passes==1 && schedule.depth==8 && schedule.repeats==std::array<unsigned,8>{1,1,1,1,1,1,1,1}),"Vendor comparison requires its full trained schedule");
        const bool profile_gpu=config.value("profile_gpu_stages",false);
        const bool deep_gpu=config.value("deep_profile_gpu_stages",false);
        require(!deep_gpu || profile_gpu,"Deep diagnostics require stage profiling");
        require(!profile_gpu || schedule.passes==1,"Stage diagnostic requires one pass");
        Gpu gpu(config.value("debug",false),profile_gpu,deep_gpu);report["memory_before"]=gpu.memory();auto color=texture(gpu.device.Get(),w,h,16,false);auto motion=texture(gpu.device.Get(),w,h,16,false,2);auto output=texture(gpu.device.Get(),w,h,16,true);
        auto shared=phoenix_native::image(gpu.device.Get(),w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        const auto input=read(fs::u8path(config.at("color_file").get<std::string>())),vectors=read(fs::u8path(config.at("motion_file").get<std::string>()));
        require(input.size()==size_t(w)*h*8&&vectors.size()==size_t(w)*h*4,"Input geometry");
        gpu.begin();for(unsigned kind=0;kind<2;++kind){auto& t=kind?motion:color;const auto& bytes=kind?vectors:input;const auto row=size_t(w)*(kind?4:8);
            void* map=nullptr;check(t.transfer->Map(0,nullptr,&map));for(unsigned y=0;y<h;++y)memcpy(static_cast<uint8_t*>(map)+t.footprint.Offset+y*t.footprint.Footprint.RowPitch,bytes.data()+y*row,row);t.transfer->Unmap(0,nullptr);
            D3D12_TEXTURE_COPY_LOCATION from{t.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,t.footprint},to{t.resource.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
            gpu.command->CopyTextureRegion(&to,0,0,0,&from,nullptr);barrier(gpu.command.Get(),t.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);}
        gpu.submit();
        ComPtr<ID3D12Resource> original_color;
        if(schedule.passes>1){original_color=phoenix_native::image(gpu.device.Get(),w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.begin();barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            gpu.command->CopyResource(original_color.Get(),color.resource.Get());barrier(gpu.command.Get(),original_color.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);gpu.submit();}
        phoenix_native::memory::Ledger ledger;auto accounting=ledger.begin(w,h);
        std::unique_ptr<phoenix_bundle::Bundle> bundle;std::unique_ptr<phoenix_addon::FrameDispatch> dispatch;std::unique_ptr<phoenix_native::Network> network;
        std::vector<std::unique_ptr<phoenix_native::Network>> refinement;
        std::unique_ptr<Nvidia> nvidia;auto started=Clock::now();
        if(vendor){const unsigned preset=source.model_id=="original"?1:2;report["render_preset"]=preset;nvidia=std::make_unique<Nvidia>(gpu,config,w,h,preset);}
        else{bundle=std::make_unique<phoenix_bundle::Bundle>(fs::u8path(config.at("bundle_path").get<std::string>()),digest,source);auto weights=bundle->load_weights();
            auto kernels=phoenix_kernel::load_catalog(phoenix_native::catalog(*bundle));const auto& plan=bundle->select_frame(w,h,[&](const Json& frame){phoenix_native::preflight(frame,kernels);});
            if (config.value("report_kernel_selection", false)) {
                // Use the production selector and generated plan, not filename
                // guesses. Handles are created for all modules; launches_per_pass
                // distinguishes modules actually used by this model's full plan.
                report["kernel_selections"] = Json::array();
                unsigned planned_launches = 0;
                for (const auto& binary : kernels) {
                    const auto& spec = binary.spec;
                    unsigned uses = 0;
                    for (const auto& launch : plan.at("launches"))
                        uses += launch.at("name") == spec.name;
                    planned_launches += uses;
                    report["kernel_selections"].push_back({
                        {"name", spec.name},
                        {"entry", phoenix_kernel::entry_for_resolution(spec.entry,
                            spec.large_input_entry, spec.large_input_min_pixels, w, h)},
                        {"launches_per_pass", uses}});
                }
                require(planned_launches == plan.at("launches").size() - 2,
                        "Kernel-selection telemetry omitted neural launches");
            }
            const auto counts=config.value("repeats",std::array<int,8>{1,1,1,1,1,1,1,1});
            dispatch=std::make_unique<phoenix_addon::FrameDispatch>();dispatch->configure_repeats(gpu.device.Get(),plan,counts);
            network=std::make_unique<phoenix_native::Network>(gpu.device.Get(),dispatch->api(),plan,weights,kernels,color.resource.Get(),motion.resource.Get(),accounting);
            for(unsigned pass=1;pass<schedule.passes;++pass)refinement.push_back(std::make_unique<phoenix_native::Network>(gpu.device.Get(),dispatch->api(),plan,weights,kernels,color.resource.Get(),motion.resource.Get(),accounting,true,network.get()));
            report["neural_launches"]=network->program_steps();report["model_resource"]=bundle->model_resource.name;}
        report["create_ms"]=ms(started);
        if(deep_gpu){
            diagnostic_heap=gpu.queries.Get();diagnostic_command=gpu.command.Get();
            if(network){require(!network->group_ends().empty(),"No native dispatch groups");
                diagnostic_last_group=unsigned(network->group_ends().size()-1);
                phoenix_native::record_progress=&diagnostic_progress;
                phoenix_native::group_marker=&diagnostic_group;}
        }
        auto frame=[&](bool reset){auto tick=Clock::now();schedule.prepare(config,source.model_id,true);gpu.begin();
            if(original_color){barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);gpu.command->CopyResource(color.resource.Get(),original_color.Get());barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);}
            gpu.command->EndQuery(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);ID3D12Resource* result=nullptr;
            if(vendor){nvidia->record(gpu,color,motion,output,reset,w,h);result=output.resource.Get();if(deep_gpu)gpu.mark(5);if(profile_gpu)gpu.mark(2);barrier(gpu.command.Get(),result,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);}
            else{phoenix_addon::Controls controls;controls.reset=reset;controls.auto_mask=1;controls.inverted=0;
                for(unsigned pass=0;pass<schedule.passes;++pass){auto* net=pass?refinement.at(pass-1).get():network.get();
                    if(pass){barrier(gpu.command.Get(),result,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);gpu.command->CopyResource(color.resource.Get(),result);barrier(gpu.command.Get(),color.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);barrier(gpu.command.Get(),result,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);}
                    const auto resolved=phoenix_addon::resolve_controls(controls,w,h,true,net->frames()!=0,reset?0:net->next_sampling_index());
                    dispatch->prepare_repeats(gpu.command.Get());phoenix_addon::FrameDispatch::Scope scope(*dispatch,gpu.command.Get(),resolved,schedule.depth);result=net->record(gpu.command.Get(),reset,0,gpu.queue.Get());scope.finish();}
                if(profile_gpu)gpu.mark(2);barrier(gpu.command.Get(),result,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);}
            if(profile_gpu)gpu.mark(3);
            gpu.command->CopyResource(shared.Get(),result);if(profile_gpu)gpu.mark(4);
            barrier(gpu.command.Get(),result,D3D12_RESOURCE_STATE_COPY_SOURCE,vendor?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.command->EndQuery(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);gpu.command->ResolveQueryData(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,gpu.query_count,gpu.stamps.Get(),0);gpu.submit();
            if(network)network->commit(gpu.command.Get());for(auto& net:refinement)net->commit(gpu.command.Get());
            const auto pair=std::pair{ms(tick),gpu.timing()};schedule.complete({{UINT32_MAX,0,pair.second,true,UINT32_MAX}});return pair;};
        auto capture=[&](const std::string& name){gpu.begin();barrier(gpu.command.Get(),shared.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{shared.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0},to{output.transfer.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,output.footprint};gpu.command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            barrier(gpu.command.Get(),shared.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST);gpu.submit();void* mapped=nullptr;check(output.transfer->Map(0,nullptr,&mapped));std::vector<uint8_t> pixels(size_t(w)*h*8);
            for(unsigned y=0;y<h;++y)memcpy(pixels.data()+size_t(y)*w*8,static_cast<uint8_t*>(mapped)+output.footprint.Offset+y*output.footprint.Footprint.RowPitch,size_t(w)*8);output.transfer->Unmap(0,nullptr);
            for(size_t i=0;i<pixels.size();i+=2){uint16_t value;memcpy(&value,pixels.data()+i,2);require((value&0x7c00)!=0x7c00,"Nonfinite model output");}
            Json row{{"sha256",phoenix_model::weight_digest(pixels.data(),pixels.size())},{"finite",true}};
            if(config.value("capture",false)&&name!="reset_b"){auto path=destination.parent_path()/(destination.stem().string()+"-"+name+".rgba16f");std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());row["file"]=path.filename().string();}
            report["captures"][name]=row;};
        const auto cold=frame(true);report["first_frame_wall_ms"]=cold.first;report["first_frame_gpu_ms"]=cold.second;
        const double warm=config.value("warmup_seconds",.5),measure=config.value("measure_seconds",1.5);unsigned warmed=0;started=Clock::now();while(warmed<4||ms(started)<warm*1000){frame(false);++warmed;}
        report["warmup_frames"]=warmed;
        if(profile_gpu)report["gpu_timestamp_calibration"]=gpu.calibrate_timestamps();
        const unsigned segment_count=deep_gpu?6:4;
        std::vector<double> walls,gpus;std::array<std::vector<double>,6> segments;
        started=Clock::now();while((walls.size()<30||ms(started)<measure*1000)&&ms(started)<15000){
            const auto pair=frame(false);walls.push_back(pair.first);gpus.push_back(pair.second);
            if(profile_gpu)for(unsigned i=0;i<segment_count;++i)segments[i].push_back(gpu.last_segments[i]);}
        report["wall"]=stats(walls);report["gpu"]=stats(gpus);report["wall_samples_ms"]=walls;report["gpu_samples_ms"]=gpus;
        if(profile_gpu){
            const std::vector<const char*> names=deep_gpu?std::vector<const char*>{"engine_prepare","dispatch_region","engine_output_transition","to_copy_transition","output_copy","restore_transition"}:
                std::vector<const char*>{"engine","to_copy_transition","output_copy","restore_transition"};
            for(unsigned i=0;i<segment_count;++i){report["gpu_stages"][names[i]]=stats(segments[i]);
                report["gpu_stage_samples_ms"][names[i]]=segments[i];}
            report["gpu_stage_scope"]="Engine includes recorded preparation/network GPU work, not CPU recording; optional extra timestamps perturb boundaries. Use the uninstrumented control. Segment sums checked per frame.";
        }
        frame(true);capture("reset_a");frame(false);capture("history");frame(true);capture("reset_b");
        report["reset_repeatable"]=report["captures"]["reset_a"]["sha256"]==report["captures"]["reset_b"]["sha256"];
        if(vendor)report["observed_cuda_launches"]=launch_count;
        report["debug_errors"]=gpu.errors();report["passed"]=report["reset_repeatable"]==true&&report["debug_errors"]==0;
        report["memory"]=gpu.memory();if(!vendor)report["owned_memory"]=ledger.snapshot();
        report["network_schedule"]=schedule.status();report["passes"]=schedule.passes;report["repeats"]=schedule.repeats;
        std::ofstream(destination)<<report.dump(2)<<'\n';std::cout<<config.at("engine")<<' '<<source.model_id<<' '<<w<<'x'<<h<<" GPU="<<report["gpu"]["median_ms"]<<" ms\n";
        return report["passed"]==true?0:2;
    }catch(const std::exception& error){report["error"]=error.what();if(!destination.empty())std::ofstream(destination)<<report.dump(2)<<'\n';std::cerr<<error.what()<<'\n';return 1;}
}
