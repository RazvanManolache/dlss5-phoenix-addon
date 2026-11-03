#pragma once
#include "client32.h"
#include "model_contract.h"
#include "bridge_policy.h"
#include "../../../native/support/network_policy.h"
#include "../../../native/support/controls.h"
#include "../host/phoenix-host-pixels.h"
#include "../cpu_color_transfer.h"
#include "../process_address_space.h"
#include "../../../pipeline/json.h"
#include <chrono>
#ifndef _WIN64
namespace phoenix_wow64 {
struct NativeFrameContext {
    uint64_t sequence=0;
    int64_t timestamp_ns=0;
    bool reset=false;
};
class FrameSession {
    using Json=nlohmann::json;
    std::unique_ptr<phoenix_bundle::Bundle> bundle_;
    std::vector<phoenix_kernel::Binary> kernels_;
    std::unordered_map<std::string,unsigned> indices_;
    Json plan_;
    phoenix_native::PreflightInfo policy_;
    std::vector<uint8_t> weights_;
    std::vector<uint16_t> color_,output_;
    std::string config_,parameters_;
    std::optional<phoenix_pipeline::Options> parameter_pipeline_;
    Json parameter_pipeline_json_;
    ModelChannel packet_;
    bool driver_;
    bool frame_services_=false;
    bool rgba8_gpu_transport_=false;
    std::vector<char> status_buffer_;
    Json runtime_status_=Json::object();
    uint64_t frames_=0;
    phoenix_compat::ProcessAddressSpace address_space_;
    uint64_t next_address_check_=0;
    NativeThread thread_; // Retires before all packet storage and model buffers.
    void ensure_half_buffers() {
        const size_t components=size_t(packet_.base.width)*packet_.base.height*4;
        color_.resize(components);output_.resize(components);
    }

  public:
    FrameSession(const std::filesystem::path& root,const std::filesystem::path& bundle,
        const phoenix_model::WeightSource& source,unsigned width,unsigned height,
        const phoenix_compat::Settings& settings,bool driver):driver_(driver) {
        const char* stage="address-space guard";
        try {
            // The driver ABI still materializes Original's 141 MiB packed image.
            // The streamed x64-library path needs only small bounded CPU chunks.
            phoenix_compat::require_in_process_address_space((driver ? 160ull : 64ull) * 1024 * 1024);
            const auto profile=phoenix_bundle::json_file(root/"bootstrap.json");
            frame_services_=!driver&&declared_library_frame_services(
                profile.value("library_frame_services",0u),profile.value("model_channel_bytes",0u));
            rgba8_gpu_transport_=!driver&&declared_rgba8_gpu_transport(profile.value("library_rgba8_transport",0u),
                profile.value("library_frame_services",0u),profile.value("model_channel_bytes",0u));
            phoenix_compat::validate_native_settings(settings,driver,frame_services_,
                frame_services_ && profile.value("library_appearance_controls",0u)==1);
            if(driver) {
                phoenix_bundle::require(source.model().id=="original",
                    "Experimental x86 driver host requires Original; use the x64-library or helper for Lightweight FP8");
                const auto image=reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
                const auto dos=reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
                const auto pe=reinterpret_cast<const IMAGE_NT_HEADERS32*>(image+dos->e_lfanew);
                phoenix_bundle::require((pe->FileHeader.Characteristics&IMAGE_FILE_LARGE_ADDRESS_AWARE)!=0,
                    "Native x86 driver processing requires a large-address-aware game; its executable is not modified. Use the helper.");
            }
            packet_.bytes=frame_services_?sizeof(packet_):model_channel_legacy_bytes;
            if(frame_services_) {
                status_buffer_.resize(65536);
                packet_.status_json=uint64_t(uintptr_t(status_buffer_.data()));
                packet_.status_capacity=uint32_t(status_buffer_.size());
            }
            auto& c=packet_.base; c.caller_bits=32; c.width=width; c.height=height;
            stage="half-precision host image buffers";
            const bool deferred=rgba8_gpu_transport_&&profile.value("library_deferred_host_buffers",0u)==1;
            if(deferred) {
                packet_.pixel_transport=rgba8_gpu_transport_version;
                c.plane_bytes=uint64_t(width)*height*4;
            } else {
                ensure_half_buffers();
                c.color=uint64_t(uintptr_t(color_.data()));c.output=uint64_t(uintptr_t(output_.data()));c.plane_bytes=color_.size()*2;
            }
            if(driver) {
                bundle_=std::make_unique<phoenix_bundle::Bundle>(bundle,file_digest,source);
                std::vector<phoenix_kernel::Spec> specs;
                for(auto item:bundle_->kernels.items()) {
                    if(item.key()=="cc_cb_clear"||item.key()=="cg2r_copy_kernel") continue;
                    const auto& row=item.value(); const auto block=row.at("block").get<std::array<unsigned,3>>();
                    specs.push_back({item.key(),row.at("binary"),row.at("binary_sha256"),row.at("entry"),row.at("parameter_bytes"),
                        block[0],block[1],block[2],row.value("large_input_entry",std::string()),row.value("large_input_min_pixels",uint64_t(0))});
                }
                kernels_=phoenix_kernel::load_catalog(specs); weights_=bundle_->load_weights();
                plan_=bundle_->select_frame(width,height,[&](const Json& frame){phoenix_native::preflight(frame,kernels_);});
                policy_=phoenix_native::preflight(plan_,kernels_);
                c.weights=uint64_t(uintptr_t(weights_.data())); c.weight_bytes=weights_.size();
                c.workspace_bytes=plan_.at("arenas").at("workspace").at("bytes"); c.clear_words=policy_.clear_words;
                for(unsigned i=0;i<kernels_.size();++i) {
                    const auto& k=kernels_[i]; indices_[k.spec.name]=i;
                    const auto& entry=phoenix_kernel::entry_for_resolution(k.spec.entry,k.spec.large_input_entry,k.spec.large_input_min_pixels,width,height);
                    c.kernels[i]={uint64_t(uintptr_t(k.bytes.data())),k.bytes.size(),uint64_t(uintptr_t(entry.c_str()))};
                }
            } else {
                auto repeats=settings.block_repeats;
                for(unsigned i=settings.blocks_for(settings.mode);i<8;++i) repeats[i]=0;
                config_=Json{{"width",width},{"height",height},{"bundle_path",phoenix_model::source_path_string(bundle)},
                    {"weight_source",source.configuration()},{"native_library",phoenix_model::source_path_string(root/"phoenix_native.dll")},
                    {"initial_backend",phoenix_compat::needs_nvidia_dll(settings)?1:3},
                    {"nvidia_preset",phoenix_addon::phoenix::nvidia_preset(settings.nvidia_model, source.model().id)},
                    {"repeats",repeats}}.dump();
                {
                    auto config=Json::parse(config_);
                    auto pipeline=settings.pipeline;
                    if(!frame_services_) {
                        // Preserve the legacy spatial/always-reset contract and
                        // historical GPU timing when using an older packet ABI.
                        pipeline.motion_enabled=false;pipeline.scene_detection=false;
                        pipeline.monitoring=int(phoenix_pipeline::Monitoring::Detailed);
                    }
                    config["pipeline"]=phoenix_pipeline::options_json(pipeline);
                    // CPU color transport carries no engine vectors. Never mark a
                    // zero-filled placeholder as valid engine motion.
                    config["engine_motion_valid"]=false;
                    config_=config.dump();
                }
                packet_.config_json=uint64_t(uintptr_t(config_.c_str()));
            }
            stage="native thread/bootstrap";
            thread_.start(root,c,driver?4:6,driver?static_cast<void*>(&c):static_cast<void*>(&packet_));
            phoenix_bundle::require(c.driver_pid==GetCurrentProcessId()&&c.driver_bits==64&&c.helper_pid==0,
                "Native bridge did not start in the caller process");
            // Driver/model startup can consume low addresses after the initial
            // routing check. Reject before serving frames if the game lost its
            // reserve; NativeThread retires before packet/model storage is freed.
            stage="post-initialization game address-space reserve";
            address_space_=phoenix_compat::require_in_process_address_space();
            next_address_check_=GetTickCount64()+1000;
        } catch(const std::exception& error) {
            throw std::runtime_error(std::string("FrameSession [")+stage+"]: "+error.what());
        }
    }
    std::vector<float> evaluate(const std::vector<uint8_t>& rgba,const phoenix_compat::Settings& settings,unsigned backend,
        const std::vector<float>* previous=nullptr,const NativeFrameContext* context=nullptr,
        std::vector<uint8_t>* packed_output=nullptr) {
        auto& c=packet_.base;
        const auto address_tick=GetTickCount64();
        if(frames_==0 || address_tick>=next_address_check_) {
            // Level loads can exhaust an initially healthy x86 process. This
            // bounded scan runs on the processing worker, never on Present.
            address_space_=phoenix_compat::require_in_process_address_space();
            next_address_check_=address_tick+1000;
        }
        // Profiling is opt-in. Default frames pay no extra clock queries.
        const bool profiling=settings.pipeline.monitoring!=0;
        using Clock=std::chrono::steady_clock;
        const auto pack_started=profiling?Clock::now():Clock::time_point{};
        phoenix_bundle::require(rgba.size()==size_t(c.width)*c.height*4,"Native bridge pixel extent mismatch");
        const bool gpu_bytes=rgba8_gpu_transport_&&packed_output&&previous==nullptr&&backend==3;
        packet_.pixel_transport=gpu_bytes?rgba8_gpu_transport_version:0;
        if(gpu_bytes) {
            // A compact GPU frame never reads these legacy half planes. Free
            // them after a float-view transition rather than retaining tens of
            // MiB in the 32-bit game's fragmented address space.
            if(!color_.empty()){std::vector<uint16_t>().swap(color_);std::vector<uint16_t>().swap(output_);}
            packed_output->resize(rgba.size());
            c.color=uint64_t(uintptr_t(rgba.data()));c.output=uint64_t(uintptr_t(packed_output->data()));c.plane_bytes=rgba.size();
        } else {
            ensure_half_buffers();
            c.color=uint64_t(uintptr_t(color_.data()));c.output=uint64_t(uintptr_t(output_.data()));c.plane_bytes=color_.size()*2;
            phoenix_compat::cpu_color::rgba8_to_half(rgba,color_);
        }
        if(previous) {
            phoenix_bundle::require(previous->size()==size_t(c.width)*c.height*3,"Native refinement pixel extent mismatch");
            for(size_t pixel=0;pixel<previous->size()/3;++pixel) for(unsigned channel=0;channel<3;++channel) {
                const float value=previous->at(pixel*3+channel);
                phoenix_bundle::require(std::isfinite(value),"Native refinement contains nonfinite pixels");
                color_[pixel*4+channel]=DirectX::PackedVector::XMConvertFloatToHalf(std::clamp(value,-65504.f,65504.f));
            }
        }
        if(!gpu_bytes)std::fill(output_.begin(),output_.end(),uint16_t(0x7e00));
        auto controls=settings.controls.apply(phoenix_addon::Controls{});
        if(settings.processing.enabled && !controls.auto_mask)controls.skin=0;
        if(driver_) {
            phoenix_bundle::require(backend==3,"Native x86 driver host is Phoenix DLSS5-only");
            const auto resolved=phoenix_addon::resolve_controls(controls,c.width,c.height,false,false,0);
            c.target=unsigned(frames_%2); c.frame_flags=0;
            const std::array<phoenix_launch_plan::Arena,2> arenas{{{c.workspace_address,c.workspace_bytes},{c.weights_address,c.weight_bytes}}};
            for(unsigned i=0;i<154;++i) {
                const auto& source=plan_.at("launches").at(i+1);
                auto launch=phoenix_bundle::launch(source,[&](const Json& binding)->uint64_t {
                    const auto role=binding.at("role").get<std::string>();
                    if(role=="color") return c.color_texture;
                    if(role=="motion"||role=="history_previous") return 0;
                    if(role=="neural_output_surface") return c.surface[c.target];
                    throw std::runtime_error("Unsupported native descriptor binding");
                });
                auto arguments=phoenix_launch_plan::materialize(launch,arenas);
                if(i==0) phoenix_addon::patch_stem(arguments.data(),arguments.size(),resolved);
                if(i==153) phoenix_addon::patch_head(arguments.data(),arguments.size(),resolved);
                auto& target=c.launches[i]; target={}; target.kernel=indices_.at(source.at("name").get<std::string>());
                target.bytes=uint32_t(arguments.size()); target.barrier_after=policy_.barrier_after[i];
                for(unsigned axis=0;axis<3;++axis) {target.grid[axis]=launch.grid[axis];target.block[axis]=launch.block[axis];}
                memcpy(target.arguments,arguments.data(),arguments.size());
            }
        } else {
            packet_.backend=backend;
            Json parameters={{"tone",controls.tone},{"structure",controls.structure},{"skin",controls.skin},
                {"auto_mask",controls.auto_mask!=0},
                {"motion_x",controls.motion_x},{"motion_y",controls.motion_y},
                {"intensity",settings.controls.appearance?controls.intensity:(settings.native_options?settings.intensity:1.f)},
                {"style",settings.controls.appearance?controls.style:(settings.native_options?settings.style:0)},
                {"ui_correction",settings.processing.enabled?int(settings.processing.correction(false)):
                    (settings.native_options?settings.ui_correction:0)}};
            if(frame_services_) {
                // Pipeline settings normally stay unchanged for thousands of
                // frames. Rebuild their JSON only on an actual live edit.
                if(!parameter_pipeline_||*parameter_pipeline_!=settings.pipeline) {
                    parameter_pipeline_json_=phoenix_pipeline::options_json(settings.pipeline);
                    parameter_pipeline_=settings.pipeline;
                }
                parameters["pipeline"]=parameter_pipeline_json_;
                parameters["reset"]=frames_<unsigned(settings.reset_after_switch)||settings.force_reset||
                    (settings.reset_every>0&&(frames_+1)%unsigned(settings.reset_every)==0)||(context&&context->reset)||
                    (!settings.pipeline.motion_enabled&&!settings.pipeline.scene_detection);
                parameters["sequence"]=context?context->sequence:frames_;
                parameters["timestamp_ns"]=context?context->timestamp_ns:
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                status_buffer_[0]=0; // A failed frame cannot expose stale motion status.
            }
            parameters_=parameters.dump();
            packet_.parameters_json=uint64_t(uintptr_t(parameters_.c_str()));
        }
        const auto dispatch_started=profiling?Clock::now():Clock::time_point{};
        thread_.frame(); ++frames_;
        if(frames_==1) {
            // The motion provider and format resources may be lazy even after
            // model creation. Do not wait a full second to account for them.
            address_space_=phoenix_compat::require_in_process_address_space();
            next_address_check_=GetTickCount64()+1000;
        }
        const auto dispatch_finished=profiling?Clock::now():Clock::time_point{};
        if(frame_services_) {
            const auto end=std::find(status_buffer_.begin(),status_buffer_.end(),'\0');
            phoenix_bundle::require(end!=status_buffer_.begin()&&end!=status_buffer_.end(),"Invalid native motion telemetry extent");
            runtime_status_=Json::parse(status_buffer_.begin(),end);
        }
        const auto unpack_started=profiling?Clock::now():Clock::time_point{};
        std::vector<float> output;
        if(packed_output&&!gpu_bytes) {
            phoenix_bundle::require(previous==nullptr,"Packed output does not support float refinements");
            phoenix_compat::cpu_color::half_rgba_to_rgba8(output_,rgba,*packed_output);
        } else if(!packed_output) {
            output.resize(size_t(c.width)*c.height*3);
            phoenix_compat::cpu_color::half_rgba_to_rgb(output_,output);
        }
        if(profiling) {
            const auto milliseconds=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
            runtime_status_["frame_stages_ms"]={
                {"input_pack_and_controls",milliseconds(pack_started,dispatch_started)},
                {"native_call_and_wait",milliseconds(dispatch_started,dispatch_finished)},
                {"status_decode",milliseconds(dispatch_finished,unpack_started)},
                {"output_decode",milliseconds(unpack_started,Clock::now())}};
            runtime_status_["output_format"]=gpu_bytes?"RGBA8-GPU":packed_output?"RGBA8":"RGB32F";
        } else runtime_status_.erase("frame_stages_ms");
        return output;
    }
    uint64_t host_half_buffer_bytes()const{return uint64_t(color_.capacity()+output_.capacity())*sizeof(uint16_t);}
    double gpu_ms() const {return packet_.base.gpu_ms;}
    const Json& runtime_status() const {return runtime_status_;}
    bool frame_services() const {return frame_services_;}
    Json nvidia_status() const {
        return Json{{"ready",!driver_&&packet_.nvidia_creates>0&&packet_.nvidia_evaluates>0&&
            packet_.nvidia_create_result==1&&packet_.nvidia_evaluate_result==1},
            {"creates",packet_.nvidia_creates},{"evaluates",packet_.nvidia_evaluates},
            {"create_result",packet_.nvidia_create_result},{"evaluate_result",packet_.nvidia_evaluate_result},
            {"feature",18}};
    }
    void retire() {
        thread_.close();
        const auto& c=packet_.base;
        phoenix_bundle::require(c.status==0&&c.native_compat_restored&&c.cleanup_failures==0,
            "Native session did not retire cleanly");
        if(driver_) phoenix_bundle::require(c.functions_destroyed==41&&c.modules_destroyed==41,
            "Native driver session did not release all kernel objects");
    }
};
}
#endif
