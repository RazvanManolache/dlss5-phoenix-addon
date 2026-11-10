#pragma once
#include "session_policy.h"
#include "cpu_color_transfer.h"
#include "processing_pixels.h"
#include "../../pipeline/recipe.h"
#include "../../pipeline/comparison_view.h"
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <memory>

namespace phoenix_compat {
struct PortableGpuHandles;
struct PortableOptions {
    std::filesystem::path root;
    std::string device = "auto";
    bool allow_unvalidated = false;
    bool gpu_transport = true;
    std::shared_ptr<PortableGpuHandles> gpu;
};
// Physical-device enumeration order can change with active graphics drivers.
// Automatic must prefer a qualified device rather than fail on an unqualified
// first entry while another installed GPU has a matching validation receipt.
inline std::string automatic_portable_device(const SettingsJson& devices,
    const std::vector<std::string>& validated, bool allow_unvalidated) {
    std::string first_capable;
    for (const auto& entry : devices) {
        const auto id = entry.value("id", std::string{});
        if (!id.starts_with("vulkan:") || !phoenix_addon::phoenix::valid_execution_device(id)) continue;
        if (first_capable.empty()) first_capable = id;
        if (std::find(validated.begin(), validated.end(), id) != validated.end()) return id;
    }
    if (allow_unvalidated && !first_capable.empty()) return first_capable;
    throw std::runtime_error("No capable validated portable Vulkan device found; explicit opt-in required");
}
inline std::string portable_backend(const std::string& device) {
    if (!phoenix_addon::phoenix::valid_execution_device(device) || device == "native" ||
        device == "sycl:cpu")
        throw std::runtime_error("Choose a portable device; SYCL CPU requires the isolated helper.");
    return device == "cpu" ? "cpu" : device.substr(0, device.find(':'));
}
inline bool portable_frame_pipeline_requested(const Settings& settings){
    const auto& p=settings.pipeline;
    return settings.controls.appearance||p.motion_enabled||p.scene_detection||p.monitoring||p.priority||
        (p.chain_enabled&&(p.scaler!=0||!phoenix_pipeline::default_chain(p,1,1,1,1)||
            (settings.processing.enabled&&settings.processing.scale!=100)));
}
inline SettingsJson native_cuda_configuration(const Settings& settings, const SettingsJson& base) {
    const auto controls=settings.controls.apply(phoenix_addon::Controls{});
    SettingsJson config={{"backend","native-cuda"},{"device",base.at("device")},
        {"runtime_root",base.at("runtime_root")},{"weight_source",base.at("weight_source")},
        {"allow_unvalidated",base.value("allow_unvalidated",false)},
        {"validated_devices",base.value("validated_devices",std::vector<std::string>{})},
        {"local_tone",controls.tone},{"local_structure",controls.structure},
        {"skin_structure",controls.skin},{"auto_mask",controls.auto_mask!=0}};
    if(base.contains("native_bundle"))config["native_bundle"]=base.at("native_bundle");
    return config;
}
// GPU transport preserves the existing SDR same-size temporal SDK contract.
// Comparison/remapping and nonidentity host color processing retain CPU transport.
inline bool portable_gpu_settings(const Settings& s,unsigned width,unsigned height){
    return sizeof(void*)==8&&s.mode==phoenix_addon::phoenix::ours&&
        phoenix_pipeline::comparison_view_bits(s.pipeline)==0&&portable_frame_pipeline_requested(s)&&
        (!s.processing.enabled||pixels::identity_transfer(width,height,s.processing));
}
inline bool native_cuda_packed_frame(const Settings& settings){
    return !portable_frame_pipeline_requested(settings)&&!settings.processing.enabled&&settings.refinement_passes==1&&
        settings.blocks_for(settings.mode)==8&&!settings.automatic_budget&&
        std::all_of(settings.block_repeats.begin(),settings.block_repeats.end(),[](auto n){return n==1;});
}
inline const char* portable_frame_services_unavailable(const Settings& settings,unsigned bits){
    // PipelineClient::supported currently exposes the shared frame SDK only
    // to x64 callers. Qualify this before loading a legacy portable model, not
    // after its first frame fails and Automatic has to cold-start a helper.
    return bits!=64&&portable_frame_pipeline_requested(settings)
        ? "The x86 portable runtime cannot provide the requested motion/scene/frame services; select the helper or disable those optional services."
        : "";
}
inline SettingsJson portable_metadata(const Settings& settings, bool reset, bool frame_pipeline=false) {
    if(settings.controls.appearance&&!frame_pipeline)
        throw std::runtime_error("Portable appearance controls require the shared x64 pipeline SDK");
    if(!frame_pipeline&&settings.pipeline.chain_enabled&&settings.pipeline.enabled[unsigned(phoenix_pipeline::Stage::Scale)]&&settings.pipeline.scaler!=0)
        throw std::runtime_error("Portable final-frame processing currently requires the existing spatial scaler");
    auto controls = settings.controls.apply(phoenix_addon::Controls{});
    if(settings.processing.enabled && !controls.auto_mask)controls.skin=0;
    auto repeats = settings.block_repeats;
    for (unsigned index = settings.blocks_for(settings.mode); index < repeats.size(); ++index)
        repeats[index] = 0;
    SettingsJson metadata = {{"reset", reset}, {"has_motion", false}, {"has_depth", false},
        {"has_history_validity", false}, {"repeats", repeats},
        {"passes", settings.refinement_passes}, {"controls", {
            {"intensity", controls.intensity}, {"style", controls.style},
            {"local_tone", controls.tone}, {"local_structure", controls.structure},
            {"skin_structure", controls.skin}, {"auto_mask", controls.auto_mask != 0}}}};
    if (settings.mode == phoenix_addon::phoenix::ours && settings.experimental && settings.automatic_budget) {
        if (settings.budget_mode != 0)
            throw std::runtime_error("Portable final-frame FPS adaptation needs application presentation timestamps");
        metadata["repeats"] = settings.block_repeats;
        metadata["transformer_blocks"] = settings.transformer_blocks;
        metadata["adaptive_budget"] = {{"enabled", true}, {"mode", "gpu_time"},
            {"target_ms", settings.budget_ms}, {"minimum_blocks", settings.minimum_blocks}};
    }
    if(frame_pipeline)metadata["transformer_blocks"]=settings.blocks_for(settings.mode);
    if(frame_pipeline||!phoenix_pipeline::default_chain(settings.pipeline,1,1,1,1))
        metadata["pipeline"]=phoenix_pipeline::options_json(phoenix_pipeline::without_comparison(settings.pipeline));
    return metadata;
}
inline std::vector<float> portable_color(const std::vector<uint8_t>& rgba) {
    if (rgba.empty() || rgba.size() % 4)
        throw std::runtime_error("Incomplete portable input image");
    std::vector<float> result(rgba.size() / 4 * 3);
    for (size_t pixel = 0; pixel < rgba.size() / 4; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            result[pixel * 3 + channel] = rgba[pixel * 4 + channel] / 255.f;
    return result;
}
// Mirrors comparison.h's shader, using this evaluation's input/output pair.
inline std::vector<uint8_t> portable_compose(const std::vector<uint8_t>& input,
                                            const std::vector<float>& output,
                                            unsigned width, const Settings& settings,
                                            const std::vector<float>* nvidia = nullptr) {
    if (!width || input.empty() || input.size() % (size_t(width) * 4) ||
        output.size() != input.size() / 4 * 3 || (needs_nvidia_dll(settings) && (!nvidia || nvidia->size()!=output.size())))
        throw std::runtime_error("Invalid portable comparison sources or geometry");
    std::vector<uint8_t> result(input.size());
    const unsigned height=unsigned(input.size()/(size_t(width)*4));
    const auto view=phoenix_pipeline::comparison_view_bits(settings.pipeline);
    if (settings.mode == phoenix_addon::phoenix::ours && view == 0) {
        // The usual enhanced-only view needs neither pixel-coordinate remaps
        // nor construction of every comparison source on each pixel.
        for (size_t pixel = 0; pixel < input.size() / 4; ++pixel) {
            for (unsigned channel = 0; channel < 3; ++channel)
                result[pixel * 4 + channel] = cpu_color::unorm8(output[pixel * 3 + channel]);
            result[pixel * 4 + 3] = input[pixel * 4 + 3];
        }
        return result;
    }
    auto saturate = [](float value) { return std::clamp(value, 0.f, 1.f); };
    for (size_t pixel = 0; pixel < input.size() / 4; ++pixel) {
        const auto point=phoenix_pipeline::comparison_point(unsigned(pixel%width),unsigned(pixel/width),width,height,view);
        const size_t sampled=size_t(point[1])*width+point[0];
        std::array<float, 3> a{}, b{}, display{}, difference{};
        for (unsigned channel = 0; channel < 3; ++channel) {
            const float enhanced = output[sampled * 3 + channel];
            if (!cpu_color::finite(enhanced)) throw std::runtime_error("Nonfinite portable output");
            const float normal = input[sampled * 4 + channel] / 255.f;
            auto source=[&](int index) {return index==0?normal:index==1&&nvidia?nvidia->at(sampled*3+channel):enhanced;};
            a[channel] = source(settings.source_a);
            b[channel] = source(settings.source_b);
            display[channel] = settings.mode==phoenix_addon::phoenix::nvidia&&nvidia?nvidia->at(sampled*3+channel):enhanced;
            difference[channel] = (a[channel] - b[channel]) * settings.difference_gain;
        }
        if (settings.mode == phoenix_addon::phoenix::input)
            for (unsigned channel = 0; channel < 3; ++channel)
                display[channel] = input[sampled * 4 + channel] / 255.f;
        else if (settings.mode == phoenix_addon::phoenix::split) {
            const auto mode = settings.comparison == 0 ? 1 : settings.comparison;
            if (mode == 1) display = pixel % width < unsigned(width * settings.divider) ? a : b;
            else for (unsigned channel = 0; channel < 3; ++channel)
                display[channel] = mode == 3 ? .5f + .5f * difference[channel] : std::abs(difference[channel]);
            if (mode == 4) {
                const float v = saturate(std::max({std::abs(difference[0]), std::abs(difference[1]), std::abs(difference[2])}));
                display = {saturate(4*v-2), saturate(4*v)-saturate(4*v-3), saturate(4*v)-saturate(4*v-1)};
            }
            if (mode == 5) display.fill(std::abs(difference[0]*.2126f + difference[1]*.7152f + difference[2]*.0722f));
        }
        for (unsigned channel = 0; channel < 3; ++channel)
            result[pixel * 4 + channel] = cpu_color::unorm8(display[channel]);
        result[pixel * 4 + 3] = input[sampled * 4 + 3];
    }
    return result;
}
}
