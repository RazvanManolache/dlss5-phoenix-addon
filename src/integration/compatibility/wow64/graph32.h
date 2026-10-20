#pragma once
#include "client32.h"
#include "model_contract.h"
#include "bridge_policy.h"
#include "../../../native/support/network_policy.h"
#include "../../../native/support/controls.h"
#include "../host/phoenix-host-pixels.h"
#ifndef _WIN64
namespace phoenix_wow64 {
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
    ModelChannel packet_;
    bool driver_;
    uint64_t frames_=0;
    NativeThread thread_; // Retires before all packet storage and model buffers.
  public:
    FrameSession(const std::filesystem::path& root,const std::filesystem::path& bundle,
        const phoenix_model::WeightSource& source,unsigned width,unsigned height,
        const phoenix_compat::Settings& settings,bool driver):driver_(driver) {
        phoenix_compat::validate_native_settings(settings,driver);
        if(driver) {
            phoenix_bundle::require(source.model().id=="original",
                "Experimental x86 driver host requires Original; use the x64-library or helper for Lightweight FP8");
            const auto image=reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
            const auto dos=reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
            const auto pe=reinterpret_cast<const IMAGE_NT_HEADERS32*>(image+dos->e_lfanew);
            phoenix_bundle::require((pe->FileHeader.Characteristics&IMAGE_FILE_LARGE_ADDRESS_AWARE)!=0,
                "Native x86 driver processing requires a large-address-aware game; its executable is not modified. Use the helper.");
        }
        packet_.bytes=sizeof(packet_);
        auto& c=packet_.base; c.caller_bits=32; c.width=width; c.height=height;
        color_.resize(size_t(width)*height*4); output_.resize(color_.size());
        c.color=uint64_t(uintptr_t(color_.data())); c.output=uint64_t(uintptr_t(output_.data())); c.plane_bytes=color_.size()*2;
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
            packet_.config_json=uint64_t(uintptr_t(config_.c_str()));
        }
        thread_.start(root,c,driver?4:6,driver?static_cast<void*>(&c):static_cast<void*>(&packet_));
        phoenix_bundle::require(c.driver_pid==GetCurrentProcessId()&&c.driver_bits==64&&c.helper_pid==0,
            "Native bridge did not start in the caller process");
    }
    std::vector<float> evaluate(const std::vector<uint8_t>& rgba,const phoenix_compat::Settings& settings,unsigned backend,
        const std::vector<float>* previous=nullptr) {
        auto& c=packet_.base;
        phoenix_bundle::require(rgba.size()*2==c.plane_bytes,"Native bridge pixel extent mismatch");
        for(size_t i=0;i<rgba.size();++i) color_[i]=DirectX::PackedVector::XMConvertFloatToHalf(rgba[i]/255.f);
        if(previous) {
            phoenix_bundle::require(previous->size()==size_t(c.width)*c.height*3,"Native refinement pixel extent mismatch");
            for(size_t pixel=0;pixel<previous->size()/3;++pixel) for(unsigned channel=0;channel<3;++channel) {
                const float value=previous->at(pixel*3+channel);
                phoenix_bundle::require(std::isfinite(value),"Native refinement contains nonfinite pixels");
                color_[pixel*4+channel]=DirectX::PackedVector::XMConvertFloatToHalf(std::clamp(value,-65504.f,65504.f));
            }
        }
        std::fill(output_.begin(),output_.end(),uint16_t(0x7e00));
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
            parameters_=Json{{"tone",controls.tone},{"structure",controls.structure},{"skin",controls.skin},
                {"auto_mask",controls.auto_mask!=0},
                {"intensity",settings.native_options?settings.intensity:1.f},
                {"style",settings.native_options?settings.style:0},
                {"ui_correction",settings.processing.enabled?int(settings.processing.correction(false)):
                    (settings.native_options?settings.ui_correction:0)}}.dump();
            packet_.parameters_json=uint64_t(uintptr_t(parameters_.c_str()));
        }
        thread_.frame(); ++frames_;
        std::vector<float> output(size_t(c.width)*c.height*3);
        for(size_t pixel=0;pixel<output.size()/3;++pixel) for(unsigned channel=0;channel<3;++channel) {
            const auto value=DirectX::PackedVector::XMConvertHalfToFloat(output_[pixel*4+channel]);
            phoenix_bundle::require(std::isfinite(value),"Native bridge returned nonfinite pixels"); output[pixel*3+channel]=value;
        }
        return output;
    }
    double gpu_ms() const {return packet_.base.gpu_ms;}
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
