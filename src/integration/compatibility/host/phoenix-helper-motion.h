#pragma once
#include "../helper_motion_policy.h"
#include "../protocol.h"
#include "../../../pipeline/d3d12_motion.h"
#include <array>

// Own one flow/history identity per mono/eye stream. Preparation happens once
// for a captured input, outside the feeder's initialization/deferred retry loop.
class PhoenixHelperMotion {
    struct Stream {
        std::unique_ptr<phoenix_pipeline::D3D12Motion> provider;
        uint64_t capture=0, sequence=0, resets=0, estimated=0;
        phoenix_pipeline::D3D12Motion::Result result;
        bool reset_requested=true;
        std::array<unsigned,2> analysis_extent{};
        std::array<double, 64> motion_ms{};
        unsigned motion_samples = 0, motion_cursor = 0;
        std::string status="Motion estimation disabled; spatial history reset";
    };
    std::array<Stream,3> streams_;
    phoenix_pipeline::Options options_;
    bool supported_=true;
public:
    void configure(const phoenix_pipeline::Options& next,bool supported=true) {
        if(!next.valid())throw std::invalid_argument("Invalid helper motion configuration");
        auto copy=options_;
        if(phoenix_compat::copy_helper_motion(copy,next)||supported_!=supported)request_reset();
        if (options_.monitoring != next.monitoring)
            for (auto& stream : streams_) stream.motion_samples = stream.motion_cursor = 0;
        options_=next;supported_=supported;
    }
    const phoenix_pipeline::Options& options()const{return options_;}
    void request_reset() {
        for (auto& stream : streams_) {
            stream.reset_requested = true;
            stream.motion_samples = stream.motion_cursor = 0;
        }
    }
    struct Frame { ID3D12Resource* motion; bool reset; bool estimated; };
    Frame prepare(unsigned id,uint64_t capture,ID3D12Resource* color,ID3D12Resource* fallback,
                  bool explicit_reset,ID3D12Device* device,ID3D12CommandQueue* queue,ID3D12Fence* consumer,
                  FeedGuideState guides={}) {
        if(id>=streams_.size())throw std::invalid_argument("Invalid helper motion stream");
        if(!feed_guides_valid(guides))throw std::invalid_argument("Invalid helper engine guides");
        auto& stream=streams_[id];
        // Transport-level repeated evaluations are refinements of this same
        // input, not a new motion sample. Keep those evaluations spatial.
        if(stream.capture==capture)return {stream.provider?stream.provider->zero_motion():fallback,true,false};
        if(stream.capture!=capture){
            stream.capture=capture;const bool first=stream.sequence++==0;
            stream.result={};
            const auto description=color->GetDesc();
            stream.analysis_extent=phoenix_pipeline::flow_extent(unsigned(description.Width),description.Height,unsigned(options_.flow_height));
            if(supported_){
                if(!stream.provider)stream.provider=std::make_unique<phoenix_pipeline::D3D12Motion>(device,queue,consumer);
                const bool profile = options_.monitoring == int(phoenix_pipeline::Monitoring::Detailed);
                const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                stream.result=stream.provider->prepare(color,D3D12_RESOURCE_STATE_COMMON,guides.engine_motion?fallback:nullptr,options_,
                    explicit_reset||stream.reset_requested,int64_t(GetTickCount64())*1000000,id,stream.sequence);
                if (profile) {
                    stream.motion_ms[stream.motion_cursor] = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    stream.motion_cursor = (stream.motion_cursor + 1) % stream.motion_ms.size();
                    stream.motion_samples = std::min<unsigned>(stream.motion_samples + 1, unsigned(stream.motion_ms.size()));
                }
                stream.status=stream.provider->status();
                stream.analysis_extent=stream.provider->analysis_extent();
            }else stream.status="Motion services require Native RTX execution";
            const bool valid=valid_motion(stream.result);
            stream.result.reset=phoenix_compat::helper_history_reset(explicit_reset||stream.reset_requested,first,valid,stream.result.reset);
            stream.reset_requested=false;
            if(stream.result.reset)++stream.resets;
            if(valid&&phoenix_pipeline::estimated_motion(stream.result.source))++stream.estimated;
        }
        const bool valid=valid_motion(stream.result);
        return {valid?stream.result.motion:stream.provider?stream.provider->zero_motion():fallback,stream.result.reset,
            valid&&phoenix_pipeline::estimated_motion(stream.result.source)};
    }
    // Return provider textures to their persistent NPS state after each actual
    // submission, including feature initialization, before publishing output.
    void record_state(ID3D12GraphicsCommandList* list,Frame frame,bool enter) {
        if(!frame.estimated)return;
        phoenix_native::transition(list,frame.motion,
            enter?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_COMMON,
            enter?D3D12_RESOURCE_STATE_COMMON:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    void consumed(unsigned id,uint64_t fence){if(streams_[id].provider)streams_[id].provider->consumed(fence);}
    void visualize(unsigned id,ID3D12GraphicsCommandList* list,ID3D12Resource* output){
        // The confidence mask belongs to NVIDIA Optical Flow. Do not display a
        // stale estimate as diagnostics for an engine-vector or zero-motion frame.
        if(!options_.motion_view||!streams_[id].provider||
            streams_[id].result.source!=phoenix_pipeline::MotionSource::Nvidia)return;
        phoenix_native::transition(list,output,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        streams_[id].provider->visualize(list,output,unsigned(options_.motion_view));
        phoenix_native::transition(list,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
    }
    nlohmann::json snapshot()const{
        auto streams=nlohmann::json::array();bool history=false;
        for(unsigned id=0;id<streams_.size();++id){const auto& s=streams_[id];if(!s.sequence)continue;
            const bool valid=valid_motion(s.result);
            history|=valid;
            auto timings = std::vector<double>(s.motion_ms.begin(), s.motion_ms.begin() + s.motion_samples);
            std::sort(timings.begin(), timings.end());
            nlohmann::json motion_window;
            if (!timings.empty()) motion_window = {{"samples", timings.size()},
                {"median", timings[timings.size() / 2]}, {"p95", timings[(timings.size() * 95 - 1) / 100]}};
            streams.push_back({{"stream",id},{"frames",s.sequence},{"estimated_frames",s.estimated},
                {"history_resets",s.resets},{"history_available",valid},{"motion_source",int(s.result.source)},
                {"motion_wall_ms",motion_window},{"timing_ms",s.provider?s.provider->timings():nullptr},
                {"analysis_width",s.analysis_extent[0]},{"analysis_height",s.analysis_extent[1]},{"status",s.status}});
        }
        return {{"supported",supported_},{"enabled",options_.motion_enabled},{"scene_detection",options_.scene_detection},
            {"history_available",history},{"streams",streams},
            {"status",streams.empty()?"Waiting for captured input":
                streams.back().at("status").get<std::string>()+"; analysis "+
                std::to_string(streams.back().at("analysis_width").get<unsigned>())+" x "+
                std::to_string(streams.back().at("analysis_height").get<unsigned>())+
                (history?"; valid motion permits temporal history":"; spatial fallback resets history")}};
    }
    void clear(){for(auto& stream:streams_)stream={};}
private:
    static bool valid_motion(const phoenix_pipeline::D3D12Motion::Result& result){
        return result.motion&&(phoenix_pipeline::estimated_motion(result.source)||
            result.source==phoenix_pipeline::MotionSource::Engine);
    }
};
