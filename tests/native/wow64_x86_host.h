#pragma once
#include "wow64_driver_contract.h"
#include "../../src/native/support/network_policy.h"
#include "../../src/native/support/controls.h"
#include <bcrypt.h>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <numeric>
#pragma comment(lib,"bcrypt.lib")

// Compiled into the x86 coordinator. This owns model authentication, graph
// generation/preflight, relocation, controls, sampling and output validation.
// The native DLL receives only generic resources and raw dispatch packets.
class PhoenixX86HostProbe {
    using Json=nlohmann::json;
    using Channel=phoenix_driver_probe::Channel;
    std::unique_ptr<phoenix_bundle::Bundle> bundle_;
    std::vector<phoenix_kernel::Binary> kernels_;
    std::unordered_map<std::string,unsigned> index_;
    std::vector<uint8_t> weights_;
    std::vector<uint16_t> color_, output_;
    Json plan_, job_, report_{{"passed",false},{"frames",Json::array()}};
    phoenix_native::PreflightInfo policy_;
    phoenix_addon::Sampling sampling_;
    std::wstring cuda_;
    std::filesystem::path receipt_;
    unsigned next_=0;
    bool done_=false;
    bool benchmark_=false;
    unsigned benchmark_phase_=0;
    uint64_t benchmark_iterations_=0;
    using Clock=std::chrono::steady_clock;
    Clock::time_point phase_started_,call_started_;
    std::vector<double> gpu_samples_,wall_samples_;
    std::array<std::array<phoenix_driver_probe::Launch,154>,2> cached_{};
    static double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
    static Json statistics(std::vector<double> values) {
        phoenix_bundle::require(!values.empty(),"No timed host frames");
        for(auto value:values) phoenix_bundle::require(std::isfinite(value)&&value>0,"Invalid timed host sample");
        const auto mean=std::accumulate(values.begin(),values.end(),0.0)/values.size(); std::sort(values.begin(),values.end());
        return {{"samples",values.size()},{"mean_ms",mean},{"median_ms",values[values.size()/2]},
            {"p95_ms",values[size_t(std::ceil(values.size()*.95))-1]}};
    }
    static uint64_t pointer(const void* value) { return uint64_t(uintptr_t(value)); }
    static std::string digest(const std::filesystem::path& path) {
        std::ifstream file(path,std::ios::binary);
        phoenix_bundle::require(bool(file),"Host cannot open authenticated file");
        BCRYPT_HASH_HANDLE hash=nullptr;
        phoenix_bundle::require(BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE,&hash,nullptr,0,nullptr,0,0)>=0,"Host SHA256 create");
        unsigned char result[32]{}; char text[65]{}; std::vector<char> buffer(1024*1024);
        try {
            while(file) {
                file.read(buffer.data(),buffer.size()); const auto bytes=file.gcount();
                if(bytes) phoenix_bundle::require(BCryptHashData(hash,reinterpret_cast<PUCHAR>(buffer.data()),ULONG(bytes),0)>=0,"Host SHA256 update");
            }
            phoenix_bundle::require(file.eof()&&!file.bad(),"Host short file read");
            phoenix_bundle::require(BCryptFinishHash(hash,result,sizeof(result),0)>=0,"Host SHA256 finish");
        } catch(...) { BCryptDestroyHash(hash); throw; }
        BCryptDestroyHash(hash);
        for(unsigned i=0;i<32;++i) sprintf_s(text+i*2,3,"%02x",unsigned(result[i]));
        return text;
    }
    static std::string pixels_digest(const std::vector<uint16_t>& pixels) {
        unsigned char result[32]{}; char text[65]{};
        phoenix_bundle::require(BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,reinterpret_cast<PUCHAR>(const_cast<uint16_t*>(pixels.data())),
            ULONG(pixels.size()*2),result,sizeof(result))>=0,"Host output digest");
        for(unsigned i=0;i<32;++i) sprintf_s(text+i*2,3,"%02x",unsigned(result[i]));
        return text;
    }
    void save(const char* stage) {
        report_["stage"]=stage;
        std::ofstream file(receipt_); file<<report_.dump(2)<<'\n';
        phoenix_bundle::require(bool(file),"Host receipt write");
    }
    void send(int32_t operation) {
        channel.operation=operation;
        phoenix_bundle::require(SetEvent(reinterpret_cast<HANDLE>(uintptr_t(channel.request_event)))!=0,"Host request event");
    }
    void prepare_frame() {
        namespace bridge=phoenix_driver_probe;
        const auto width=channel.width,height=channel.height;
        const bool alternate=next_==3||next_==4;
        for(size_t i=0;i<color_.size();++i) color_[i]=i%4==3?0x3c00:uint16_t(0x3000+(i*19+(alternate?256:0))%0x800);
        std::fill(output_.begin(),output_.end(),uint16_t(0x7e00));
        phoenix_addon::Controls controls; controls.reset=1;
        if(next_==4) { controls.tone=0; controls.structure=0; controls.skin=0; controls.auto_mask=1; }
        const auto resolved=phoenix_addon::resolve_controls(controls,width,height,false,false,sampling_.begin(true));
        channel.target=next_%2;
        build_program(resolved);
        save("frame_dispatch"); send(bridge::frame);
    }
    void build_program(const phoenix_addon::ResolvedControls& resolved) {
        namespace bridge=phoenix_driver_probe;
        const std::array<phoenix_launch_plan::Arena,2> arenas{{
            {channel.workspace_address,channel.workspace_bytes},{channel.weights_address,channel.weight_bytes}}};
        for(unsigned i=0;i<bridge::launch_count;++i) {
            const auto& source=plan_.at("launches").at(i+1);
            auto launch=phoenix_bundle::launch(source,[&](const Json& binding)->uint64_t {
                const auto role=binding.at("role").get<std::string>();
                if(role=="color") return channel.color_texture;
                if(role=="history_previous"||role=="motion") return 0;
                if(role=="neural_output_surface") return channel.surface[channel.target];
                throw std::runtime_error("Unexpected host descriptor role");
            });
            auto bytes=phoenix_launch_plan::materialize(launch,arenas);
            if(i==0) phoenix_addon::patch_stem(bytes.data(),bytes.size(),resolved);
            if(i==bridge::launch_count-1) phoenix_addon::patch_head(bytes.data(),bytes.size(),resolved);
            auto& target=channel.launches[i]; target={}; target.kernel=index_.at(source.at("name").get<std::string>());
            target.bytes=uint32_t(bytes.size()); target.barrier_after=policy_.barrier_after[i];
            for(unsigned axis=0;axis<3;++axis) { target.grid[axis]=launch.grid[axis]; target.block[axis]=launch.block[axis]; }
            memcpy(target.arguments,bytes.data(),bytes.size());
            phoenix_bundle::require(bridge::valid_launch(target),"Host packet validation");
        }
    }
    void prepare_benchmark_frame() {
        call_started_=Clock::now();
        channel.target=unsigned(benchmark_iterations_%2); channel.frame_flags=3;
        memcpy(channel.launches,cached_[channel.target].data(),sizeof(channel.launches));
        send(phoenix_driver_probe::frame);
    }
    void start_benchmark() {
        phoenix_addon::Controls controls; controls.reset=1;
        const auto resolved=phoenix_addon::resolve_controls(controls,channel.width,channel.height,false,false,0);
        for(unsigned target=0;target<2;++target) {
            channel.target=target; build_program(resolved);
            memcpy(cached_[target].data(),channel.launches,sizeof(channel.launches));
        }
        save("benchmark_warmup"); benchmark_phase_=1; phase_started_=Clock::now(); prepare_benchmark_frame();
    }
    void advance_benchmark() {
        const auto wall=elapsed(call_started_); ++benchmark_iterations_;
        if(benchmark_phase_==1&&elapsed(phase_started_)>=5000) {
            benchmark_phase_=2; phase_started_=Clock::now();
        } else if(benchmark_phase_==2) {
            gpu_samples_.push_back(channel.gpu_ms); wall_samples_.push_back(wall);
            const auto measured=elapsed(phase_started_);
            if(measured>=20000) {
                phoenix_bundle::require(gpu_samples_.size()>=16,"Insufficient timed host frames");
                report_["benchmark"]={{"width",channel.width},{"height",channel.height},{"scale",1},{"warmup_seconds",5},
                    {"requested_seconds",20},{"measured_seconds",measured/1000},{"completed_frames",gpu_samples_.size()},
                    {"total_iterations",benchmark_iterations_},{"completed_output_fps",gpu_samples_.size()*1000.0/measured},
                    {"gpu_processing",statistics(gpu_samples_)},{"synchronized_call",statistics(wall_samples_)},
                    {"reset_each_frame",true},{"scope","GPU-resident input/output, cached x86 dispatch packets; synchronized call includes per-frame x86/native event handoff. Excludes host image transfers, hashing, initialization and game presentation."}};
                benchmark_phase_=3; channel.frame_flags=0; prepare_frame(); return;
            }
        }
        prepare_benchmark_frame();
    }
  public:
    Channel channel{};
    explicit PhoenixX86HostProbe(const wchar_t* path) {
        job_=phoenix_bundle::json_file(path); receipt_=std::filesystem::u8path(job_.at("output").get<std::string>());
        benchmark_=job_.value("benchmark_x86_host",false);
        channel.caller_bits=sizeof(void*)*8; channel.width=job_.at("width"); channel.height=job_.at("height");
        report_["host_bits"]=sizeof(void*)*8; report_["process_pid"]=GetCurrentProcessId(); report_["helper_pid"]=0;
        report_["host_owns_graph"]=true; report_["host_owns_controls"]=true;
        cuda_=std::filesystem::u8path(job_.at("cuda_library").get<std::string>()).wstring(); channel.cuda_library=pointer(cuda_.c_str());
        channel.ntdll_timestamp=job_.at("ntdll_timestamp");
        const auto offsets=job_.at("appcompat_offsets").get<std::array<uint32_t,4>>();
        std::copy(offsets.begin(),offsets.end(),channel.appcompat_offsets);
        save("host_authenticate_model");
        bundle_=std::make_unique<phoenix_bundle::Bundle>(std::filesystem::u8path(job_.at("config").at("bundle_path").get<std::string>()),
            digest,phoenix_model::WeightSource::config(job_.at("config").at("weight_source")));
        std::vector<phoenix_kernel::Spec> specs;
        for(const auto& item:bundle_->kernels.items()) {
            if(item.key()=="cc_cb_clear"||item.key()=="cg2r_copy_kernel") continue;
            const auto& row=item.value(); const auto block=row.at("block").get<std::array<unsigned,3>>();
            specs.push_back({item.key(),row.at("binary"),row.at("binary_sha256"),row.at("entry"),row.at("parameter_bytes"),
                block[0],block[1],block[2],row.value("large_input_entry",std::string()),row.value("large_input_min_pixels",uint64_t(0))});
        }
        kernels_=phoenix_kernel::load_catalog(specs); weights_=bundle_->load_weights();
        plan_=bundle_->select_frame(channel.width,channel.height,[&](const Json& frame) { phoenix_native::preflight(frame,kernels_); });
        policy_=phoenix_native::preflight(plan_,kernels_);
        channel.workspace_bytes=plan_.at("arenas").at("workspace").at("bytes");
        channel.weight_bytes=weights_.size(); channel.weights=pointer(weights_.data()); channel.clear_words=policy_.clear_words;
        for(unsigned i=0;i<kernels_.size();++i) {
            auto& kernel=kernels_[i]; index_.emplace(kernel.spec.name,i);
            const auto& entry=phoenix_kernel::entry_for_resolution(kernel.spec.entry,kernel.spec.large_input_entry,
                kernel.spec.large_input_min_pixels,channel.width,channel.height);
            channel.kernels[i]={pointer(kernel.bytes.data()),kernel.bytes.size(),pointer(entry.c_str())};
        }
        color_.resize(size_t(channel.width)*channel.height*4); output_.resize(color_.size());
        channel.color=pointer(color_.data()); channel.output=pointer(output_.data()); channel.plane_bytes=color_.size()*2;
        channel.request_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        channel.response_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        phoenix_bundle::require(phoenix_driver_probe::valid_initialize(channel),"Host init contract");
        report_["authenticated_kernels"]=kernels_.size(); report_["launches_per_frame"]=154;
        save("driver_initialize");
    }
    ~PhoenixX86HostProbe() {
        if(channel.request_event) CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(channel.request_event)));
        if(channel.response_event) CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(channel.response_event)));
    }
    void fail(const char* message) { report_["error"]=message; report_["passed"]=false; save("failed"); }
    void advance() {
        phoenix_bundle::require(channel.status==0,channel.error);
        if(channel.operation==phoenix_driver_probe::initialize) { prepare_frame(); return; }
        if(channel.operation==phoenix_driver_probe::shutdown) { done_=true; return; }
        if(benchmark_phase_==1||benchmark_phase_==2) { advance_benchmark(); return; }
        size_t changed=0;
        for(size_t i=0;i<output_.size();++i) {
            phoenix_bundle::require((output_[i]&0x7c00)!=0x7c00,"Nonfinite x86-host output");
            if(i%4!=3) changed+=output_[i]!=color_[i];
        }
        phoenix_bundle::require(changed>0,"Unchanged x86-host output");
        if(benchmark_phase_==3) {
            const auto hash=pixels_digest(output_);
            phoenix_bundle::require(hash==report_["frames"][0]["output_sha256"].get<std::string>(),"Timed host changed reset-frame result");
            report_["benchmark"]["output_sha256"]=hash; report_["benchmark"]["changed_rgb_samples"]=changed;
            sampling_.commit(); save("driver_shutdown"); send(phoenix_driver_probe::shutdown); return;
        }
        report_["frames"].push_back({{"index",next_},{"output_sha256",pixels_digest(output_)},
            {"changed_rgb_samples",changed},{"gpu_ms",channel.gpu_ms},{"reset",true},
            {"input_changed",next_==3||next_==4},{"controls_changed",next_==4}});
        sampling_.commit(); ++next_;
        if(next_<6) prepare_frame(); else if(benchmark_) start_benchmark(); else { save("driver_shutdown"); send(phoenix_driver_probe::shutdown); }
    }
    bool finish() {
        const auto& frames=report_["frames"];
        const bool output_ok=frames.size()==6 && frames[0]["output_sha256"]==frames[1]["output_sha256"] &&
            frames[0]["output_sha256"]==frames[2]["output_sha256"] && frames[0]["output_sha256"]==frames[5]["output_sha256"] &&
            frames[0]["output_sha256"]!=frames[3]["output_sha256"] && frames[3]["output_sha256"]!=frames[4]["output_sha256"];
        report_["driver_bits"]=channel.driver_bits; report_["driver_pid"]=channel.driver_pid;
        report_["phoenix64_loaded"]=channel.phoenix64_loaded!=0; report_["helper_pid"]=channel.helper_pid;
        report_["completed_frames"]=channel.completed_frames; report_["dispatched_kernels"]=channel.dispatched_kernels;
        report_["functions_destroyed"]=channel.functions_destroyed; report_["modules_destroyed"]=channel.modules_destroyed;
        report_["cleanup_failures"]=channel.cleanup_failures; report_["native_compat_restored"]=channel.native_compat_restored!=0;
        if(channel.status) report_["error"]=channel.error;
        report_["controls_and_input_verified"]=output_ok;
        const auto expected_frames=benchmark_?7+benchmark_iterations_:6;
        const bool benchmark_ok=!benchmark_||(report_.contains("benchmark")&&report_["benchmark"].contains("output_sha256"));
        const bool passed=done_&&output_ok&&benchmark_ok&&channel.driver_bits==64&&channel.driver_pid==GetCurrentProcessId()&&
            !channel.phoenix64_loaded&&!channel.helper_pid&&channel.completed_frames==expected_frames&&channel.dispatched_kernels==expected_frames*154&&
            channel.functions_destroyed==41&&channel.modules_destroyed==41&&!channel.cleanup_failures&&channel.native_compat_restored;
        report_["passed"]=passed; save(passed?"complete":"failed"); return passed;
    }
};

#ifdef _WIN64
inline int PhoenixRunNativeHostControl(const wchar_t* library_path,const wchar_t* job) {
    PhoenixX86HostProbe host(job);
    auto library=LoadLibraryW(library_path); if(!library) return 2;
    using Serve=int(*)(phoenix_driver_probe::Channel*);
    auto serve=reinterpret_cast<Serve>(GetProcAddress(library,"PhoenixDriverServe64")); if(!serve) return 2;
    struct Context { Serve serve; phoenix_driver_probe::Channel* channel; } context{serve,&host.channel};
    const auto thread=CreateThread(nullptr,0,[](void* pointer)->DWORD {
        auto& context=*static_cast<Context*>(pointer); return context.serve(context.channel);
    },&context,0,nullptr);
    if(!thread) return 2;
    const auto deadline=GetTickCount64()+60000;
    for(;;) {
        HANDLE handles[]{thread,reinterpret_cast<HANDLE>(host.channel.response_event)};
        const auto waited=WaitForMultipleObjects(2,handles,FALSE,100);
        if(waited==WAIT_OBJECT_0) break;
        if(waited==WAIT_OBJECT_0+1) try { host.advance(); } catch(const std::exception& error) { host.fail(error.what()); TerminateProcess(GetCurrentProcess(),2); }
        if(waited==WAIT_FAILED||GetTickCount64()>=deadline) { host.fail("Native host control timed out"); TerminateProcess(GetCurrentProcess(),3); }
    }
    // The worker can exit immediately after publishing its shutdown response.
    if(host.channel.operation==phoenix_driver_probe::shutdown&&host.channel.status==0) host.advance();
    DWORD code=2; GetExitCodeThread(thread,&code); CloseHandle(thread);
    return !code&&host.finish()?0:2;
}
#endif
