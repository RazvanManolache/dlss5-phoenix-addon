#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <tuple>

namespace phoenix_pipeline {
enum class MotionSource : int { Automatic, Engine, Nvidia, Zero };
enum class Monitoring : int { Off, Basic, Detailed };
enum class Priority : int { Fixed, Responsiveness, Quality };
enum class Stage : int { Neural, Scale, Color, Fusion };

// Persisted by all frontends using the same field names. Defaults preserve the
// motion source preference: real engine vectors first, estimate missing vectors.
// Additional monitoring and image-processing stages remain opt-in.
struct Options {
    static constexpr float maximum_depth_strength = 10.f;
    bool motion_enabled = true, scene_detection = false;
    int motion_source = 0, flow_height = 180, motion_view = 0;
    int legacy_depth = 0; // Auto, Small, NVIDIA, None, Game depth.
    float legacy_depth_strength = 1.f;
    bool legacy_depth_preview = false;
    bool legacy_depth_invert = false, legacy_depth_fixed_range = false;
    float legacy_depth_near = 0.f, legacy_depth_far = 1.f;
    float flow_consistency = 1.5f, flow_cost = 100.f;
    int monitoring = 0, priority = 0, sample_interval = 30;
    float latency_ms = 50.f;
    bool adjust_flow = true, adjust_scale = false, adjust_blocks = false, adjust_passes = false;
    int minimum_flow_height = 180, minimum_passes = 1;
    float minimum_scale = 50.f;
    bool recovery = true, chain_enabled = false, fusion = false;
    int scaler = 0, vsr_quality = 2;
    float sharpness = .25f;
    float color_brightness=0, color_contrast=1, color_saturation=1, color_gamma=1;
    std::array<int, 4> order{0, 3, 1, 2};
    std::array<bool, 4> enabled{true, true, true, false};
    int fusion_layers = 2;
    float fusion_second_scale = .5f, fusion_third_scale = .75f;
    float compare_zoom = 1.f, loupe_x = .5f, loupe_y = .5f;
    bool loupe = false, screenshot_receipt = true;
    bool operator==(const Options&) const = default;
    // UI/persisted cost is per mille of the provider's normalized cost range.
    float normalized_flow_cost() const {return std::clamp(flow_cost/1000.f,0.f,.99f);}
    bool valid() const {
        auto finite = [](float v, float lo, float hi) { return std::isfinite(v) && v >= lo && v <= hi; };
        auto permutation = order;
        std::sort(permutation.begin(), permutation.end());
        return legacy_depth >= 0 && legacy_depth <= 4 &&
            finite(legacy_depth_strength, 0.f, maximum_depth_strength) &&
            finite(legacy_depth_near, 0.f, 1.f) && finite(legacy_depth_far, 0.f, 1.f) &&
            legacy_depth_far - legacy_depth_near >= 0.00001f &&
            motion_source >= 0 && motion_source <= 3 &&
            flow_height >= 64 && flow_height <= 2160 && motion_view >= 0 && motion_view <= 2 &&
            finite(flow_consistency, .1f, 20.f) && finite(flow_cost, 1.f, 1000.f) &&
            monitoring >= 0 && monitoring <= 2 && priority >= 0 && priority <= 2 &&
            sample_interval >= 1 && sample_interval <= 600 && finite(latency_ms, 1.f, 1000.f) &&
            minimum_flow_height >= 64 && minimum_flow_height <= 2160 && minimum_passes >= 1 && minimum_passes <= 10 &&
            finite(minimum_scale, 10.f, 100.f) &&
            scaler >= 0 && scaler <= 3 && vsr_quality >= 1 && vsr_quality <= 4 &&
            finite(sharpness, 0, 1) && finite(color_brightness,-2,2) && finite(color_contrast,0,4) &&
            finite(color_saturation,0,2) && finite(color_gamma,.1f,4) && permutation == std::array<int, 4>{0, 1, 2, 3} &&
            fusion_layers >= 2 && fusion_layers <= 3 &&
            finite(fusion_second_scale, .1f, 1.f) && finite(fusion_third_scale, .1f, 1.f) &&
            finite(compare_zoom, 1, 8) && finite(loupe_x, 0, 1) && finite(loupe_y, 0, 1);
    }
};
inline bool copy_comparison(Options& target,const Options& source){
    auto fields=[](auto& p){return std::tie(p.compare_zoom,p.loupe_x,p.loupe_y,p.loupe);};
    const bool changed=fields(target)!=fields(source);fields(target)=fields(source);return changed;
}
inline Options without_comparison(Options options){copy_comparison(options,Options{});return options;}
template<class Visitor> void visit_depth_options(Options& p, Visitor&& visit) {
    visit("LegacyDepth", p.legacy_depth);
    visit("LegacyDepthStrength", p.legacy_depth_strength);
    visit("LegacyDepthPreview", p.legacy_depth_preview);
    visit("LegacyDepthInvert", p.legacy_depth_invert);
    visit("LegacyDepthFixedRange", p.legacy_depth_fixed_range);
    visit("LegacyDepthNear", p.legacy_depth_near);
    visit("LegacyDepthFar", p.legacy_depth_far);
}
template<class Visitor> void visit_options(Options& p, Visitor visit) {
    visit_depth_options(p, visit);
    visit("MotionEstimation", p.motion_enabled);
    visit("MotionProvider", p.motion_source);
    visit("FlowAnalysisHeight", p.flow_height);
    visit("MotionDiagnosticView", p.motion_view);
    visit("FlowConsistencyPixels", p.flow_consistency);
    visit("FlowCostThreshold", p.flow_cost);
    visit("SceneDetection", p.scene_detection);
    visit("PipelineMonitoring", p.monitoring);
    visit("PipelinePriority", p.priority);
    visit("MonitoringSampleInterval", p.sample_interval);
    visit("MaximumFrameAgeMs", p.latency_ms);
    visit("AdjustMotionResolution", p.adjust_flow);
    visit("AdjustNeuralResolution", p.adjust_scale);
    visit("AdjustCentralBlocks", p.adjust_blocks);
    visit("AdjustNetworkPasses", p.adjust_passes);
    visit("MinimumFlowHeight", p.minimum_flow_height);
    visit("MinimumNetworkPasses", p.minimum_passes);
    visit("MinimumNeuralScale", p.minimum_scale);
    visit("StartupRecovery", p.recovery);
    visit("ProcessingChain", p.chain_enabled);
    visit("AdditionalScaler", p.scaler);
    visit("VsrQuality", p.vsr_quality);
    visit("ScalerSharpness", p.sharpness);
    visit("ChainBrightness",p.color_brightness);
    visit("ChainContrast",p.color_contrast);
    visit("ChainSaturation",p.color_saturation);
    visit("ChainGamma",p.color_gamma);
    visit("MultiscaleFusion", p.fusion);
    visit("FusionLayers", p.fusion_layers);
    visit("FusionSecondScale", p.fusion_second_scale);
    visit("FusionThirdScale", p.fusion_third_scale);
    visit("CompareZoom", p.compare_zoom);
    visit("CompareLoupe", p.loupe);
    visit("LoupeX", p.loupe_x);
    visit("LoupeY", p.loupe_y);
    visit("ScreenshotReceipt", p.screenshot_receipt);
    const char* order_names[]{"ChainStage1", "ChainStage2", "ChainStage3", "ChainStage4"};
    const char* enabled_names[]{"ChainNeural", "ChainScale", "ChainColor", "ChainFusion"};
    for (unsigned i = 0; i != 4; ++i) {
        visit(order_names[i], p.order[i]);
        visit(enabled_names[i], p.enabled[i]);
    }
}
struct MotionChoice {
    MotionSource source = MotionSource::Zero;
    const char* reason = "No motion source";
};
inline MotionChoice choose_motion(const Options& p, bool engine_valid, bool nvidia_ready) {
    const auto requested = MotionSource(p.motion_source);
    if (!p.motion_enabled)
        return engine_valid && (requested == MotionSource::Automatic || requested == MotionSource::Engine)
            ? MotionChoice{MotionSource::Engine, "Engine vectors"}
            : MotionChoice{MotionSource::Zero, "Motion estimation disabled or zero motion selected"};
    if ((requested == MotionSource::Automatic || requested == MotionSource::Engine) && engine_valid)
        return {MotionSource::Engine, "Engine vectors"};
    if ((requested == MotionSource::Automatic || requested == MotionSource::Nvidia) && nvidia_ready)
        return {MotionSource::Nvidia, "NVIDIA Optical Flow"};
    return {MotionSource::Zero, requested == MotionSource::Zero ? "Motion disabled" : "Requested motion source unavailable"};
}
inline std::array<unsigned, 2> flow_extent(unsigned width, unsigned height, unsigned cap) {
    if (!width || !height || !cap) return {0, 0};
    const double scale = std::min(1., double(cap) / height);
    return {std::max(1u, unsigned(std::lround(width * scale))),
            std::max(1u, unsigned(std::lround(height * scale)))};
}
} // namespace phoenix_pipeline
