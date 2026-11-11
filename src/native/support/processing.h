#pragma once
#include "resolution.h"
#include "../model/native_layout.h"
#include <string_view>
#include <tuple>

namespace phoenix_addon::phoenix {
inline constexpr float kMaxProcessingScale=400.f;
// The legacy bridge remains the default for existing profiles. Enabling this
// pipeline makes scaling and encoding explicit and identical on both networks.
struct Processing {
    bool enabled = false;
    int hook = 2;         // Off, Render (before SR/RR), Upscaled, Present.
    int reuse_guides = 1; // Never, Optional, Required, at Present only.
    float scale = 100;
    int resize_method = 0; // Legacy manual, Auto, NIS, Bilinear, Bicubic, Nearest.
    float nis_sharpness = .25f;
    int domain = 0, downsample = 0, upsample = 0, reconstruction = 0;
    int encoding = 0, signal = 0; // Auto/linear709/sRGB/HDR10/scRGB/scRGB-nl; sRGB/PQ/linear.
    bool white_override = false;
    float white_nits = 100;
    float exposure = 1;
    int exposure_mode = 1; // Off, On, Inverse.
    bool block_hudless = true;
    int ui_policy = 0; // Auto (host policy), Off, On.
    bool operator==(const Processing&) const = default;
    bool valid() const {
        return hook >= 0 && hook <= 3 && reuse_guides >= 0 && reuse_guides <= 2 &&
               std::isfinite(scale) && scale >= 0 && scale <= kMaxProcessingScale &&
               resize_method>=0 && resize_method<=5 && std::isfinite(nis_sharpness) && nis_sharpness>=0 && nis_sharpness<=1 &&
               domain >= 0 && domain <= 2 &&
               downsample >= 0 && downsample <= 2 && upsample >= 0 && upsample <= 2 &&
               reconstruction >= 0 && reconstruction <= 1 && encoding >= 0 && encoding <= 5 &&
               signal >= 0 && signal <= 2 && std::isfinite(white_nits) && white_nits >= 48 &&
               white_nits <= 500 && std::isfinite(exposure) && exposure >= 0 && exposure <= 100 &&
               exposure_mode >= 0 && exposure_mode <= 2 && ui_policy >= 0 && ui_policy <= 2;
    }
    int enlargement_filter() const {return resize_method==3?0:resize_method==5?2:1;}
    int reduction_filter() const {return resize_method==1?0:downsample;}
    float effective_exposure(float host = 1) const {
        if (exposure_mode == 0)
            return 1;
        const float value = std::clamp(exposure * (std::isfinite(host) ? host : 1.f), 0.f, 10.f);
        // Zero remains a black exposure in either convention; never divide by zero.
        return exposure_mode == 2 && value > 0 ? 1.f / value : value;
    }
    float diffuse_white(float host = 100) const {
        return white_override ? white_nits
                              : (std::isfinite(host) && host >= 48 && host <= 500 ? host : 100.f);
    }
    int input_encoding(bool srgb, bool hdr10 = false, bool scrgb = false) const {
        return encoding ? encoding : hdr10 ? 3 : scrgb ? 4 : srgb ? 2 : 1;
    }
    bool correction(bool host) const {
        return ui_policy == 0 ? host : ui_policy == 2;
    }
};
template <class Visitor> void visit_processing(Processing& p, Visitor visit) {
    visit("ProcessingEnabled", p.enabled);
    visit("HookStage", p.hook);
    visit("ReuseDlssUpscalingResources", p.reuse_guides);
    visit("ProcessingScale", p.scale);
    visit("ResizeMethod",p.resize_method);visit("NisSharpness",p.nis_sharpness);
    visit("FilterDomain", p.domain);
    visit("DownsampleFilter", p.downsample);
    visit("UpsampleFilter", p.upsample);
    visit("ScaleReconstruction", p.reconstruction);
    visit("Encoding", p.encoding);
    visit("SignalEncoding", p.signal);
    visit("DiffuseWhiteOverride", p.white_override);
    visit("DiffuseWhiteNits", p.white_nits);
    visit("Exposure", p.exposure);
    visit("ExposureMode", p.exposure_mode);
    visit("BlockHudless", p.block_hudless);
    visit("UiCorrectionMode", p.ui_policy);
}
// Color choices are shader constants, not model or canvas creation settings.
template<class Visitor> void visit_processing_resources(Processing& p,Visitor visit) {
    visit("ProcessingEnabled",p.enabled);visit("HookStage",p.hook);
    visit("ReuseDlssUpscalingResources",p.reuse_guides);visit("ProcessingScale",p.scale);
    visit("ResizeMethod",p.resize_method);
    visit("BlockHudless",p.block_hudless);visit("UiCorrectionMode",p.ui_policy);
}
template<class Visitor> void visit_processing_color(Processing& p, Visitor visit) {
    visit("FilterDomain",p.domain);visit("DownsampleFilter",p.downsample);
    visit("UpsampleFilter",p.upsample);visit("ScaleReconstruction",p.reconstruction);
    visit("Encoding",p.encoding);visit("SignalEncoding",p.signal);
    visit("DiffuseWhiteOverride",p.white_override);visit("DiffuseWhiteNits",p.white_nits);
    visit("Exposure",p.exposure);visit("ExposureMode",p.exposure_mode);
    visit("NisSharpness",p.nis_sharpness);
}
inline bool copy_processing_color(Processing& target,const Processing& source) {
    const auto fields=[](auto& p){return std::tie(p.domain,p.downsample,p.upsample,p.reconstruction,
        p.encoding,p.signal,p.white_override,p.white_nits,p.exposure,p.exposure_mode,p.nis_sharpness);};
    const bool changed=fields(target)!=fields(source);fields(target)=fields(source);return changed;
}
inline bool same_processing_resources(Processing before,const Processing& after) {
    copy_processing_color(before,after);return before==after;
}
inline Resolution processing_resolution(unsigned width, unsigned height, const Processing& p,
                                        bool native) {
    if (!width || !height)
        return {};
    const unsigned maximum = native ? 16384u : 8192u;
    const unsigned minimum = 16; // Network convolution/tile minimum, not a zero-size dispatch.
    const double factor=std::min({double(p.scale)/100.0,double(maximum)/width,double(maximum)/height});
    auto dimension = [&](unsigned value) {
        return std::clamp(unsigned(std::lround(double(value) * factor)), minimum, maximum);
    };
    const auto w = dimension(width), h = dimension(height);
    return {width, height, w, h, w, h, 0, 0, false};
}
// Preset is a separate architecture selector, never the artistic A/B/C Style.
// -1 follows the selected Phoenix architecture; 0 explicitly asks the DLL default.
inline unsigned nvidia_preset(int choice, std::string_view model) {
    return choice < 0 ? phoenix_model::definition(model.empty() ? "original" : model).nvidia_preset
                      : unsigned(choice);
}
struct ModelProfile {
    unsigned learned_blocks, tensors, records, native_launches;
    const char* weights;
    unsigned first_transformer;
    phoenix_model::NativeLayout layout;
    unsigned scheduled_launches(unsigned depth, const std::array<int, 8>& repeats) const {
        return layout.scheduled_launches(depth, repeats);
    }
};
inline ModelProfile model_profile(std::string_view model) {
    const auto& definition = phoenix_model::definition(model);
    const auto layout = phoenix_model::native_layout(definition.architecture);
    return {definition.history_gain_block + 1, unsigned(definition.decoded_tensor_count),
            unsigned(definition.record_count), layout.launches(), definition.resource.data(),
            definition.native.first_transformer, layout};
}
} // namespace phoenix_addon::phoenix
