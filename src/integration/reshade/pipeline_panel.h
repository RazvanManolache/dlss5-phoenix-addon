#pragma once
#include "widgets.h"
#include <imgui.h>
#include "choice_buttons.h"
#include "../../pipeline/options.h"
#include "../../pipeline/tiny_model.h"

namespace phoenix_ui {
enum class PipelineSection { Motion, Responsiveness, Chain, Inspection, Recovery, All };
// Each group keeps its original capability gates. Selecting a tab never
// normalizes a saved recipe or enables a service that the route cannot supply.
inline bool draw_pipeline_options(phoenix_pipeline::Options& p,
    const std::string& model_library = {}, bool native_chain = true,
    bool portable_chain = false, bool frame_services = true,
    PipelineSection section = PipelineSection::All, bool motion_services = false,bool managed_resize=false,
    bool x86_transport = false) {
    if (section == PipelineSection::All &&
        !ImGui::CollapsingHeader("Motion, latency and processing chain")) return false;
    bool changed = false;
    if (section == PipelineSection::Motion || section == PipelineSection::All) {
        if(!frame_services&&!motion_services&&!portable_chain)ImGui::TextWrapped("Flow and scene analysis require a compatible final-frame bridge or Native RTX helper. Pipeline-age controls require the local bridge.");
        ImGui::BeginDisabled(!frame_services&&!motion_services&&!x86_transport);
        changed |= ImGui::Checkbox("Estimate motion when engine vectors are unavailable", &p.motion_enabled);
        const char* motion_route = "In x86 games this forces the external x64 helper if the installed in-process runtime cannot provide the requested motion services. This applies in Automatic and locks the execution selector while required.";
        if (x86_transport && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", motion_route);
        const char* providers[]{"Automatic (prefer engine vectors)", "Engine vectors", "NVIDIA Optical Flow", "Zero motion",
            "AMD FidelityFX Optical Flow", "OpenCV DIS (CPU)", "D3D12 Video Motion Estimation", "Luma block matching"};
        const char* provider_captions[]{"Auto","Engine","NVIDIA flow","Zero","FidelityFX","OpenCV DIS","D3D12 Video","Luma blocks"};
        const char* route_notes[]{motion_route, motion_route, motion_route, motion_route,
            motion_route, motion_route, motion_route, motion_route};
        changed |= choice_buttons("Motion source", &p.motion_source, providers, int(std::size(providers)), provider_captions,
            nullptr, x86_transport ? route_notes : nullptr);
        const auto source = phoenix_pipeline::MotionSource(p.motion_source);
        const bool nvidia = source == phoenix_pipeline::MotionSource::Automatic || source == phoenix_pipeline::MotionSource::Nvidia;
        const bool estimating = p.motion_enabled && (nvidia || phoenix_pipeline::alternative_motion(source));
        ImGui::BeginDisabled(!estimating);
        changed |= phoenix_ui::slider_int("Motion analysis height", &p.flow_height, 64, 2160, phoenix_pipeline::Options{}.flow_height);
        ImGui::TextWrapped("Analysis preserves aspect ratio. D3D12 Video uses even dimensions for NV12. Lower heights reduce estimation work; neural output resolution is unchanged.");
        ImGui::BeginDisabled(!nvidia);
        changed |= phoenix_ui::slider_float("Motion consistency tolerance (pixels)", &p.flow_consistency, .1f, 20.f, phoenix_pipeline::Options{}.flow_consistency);
        changed |= phoenix_ui::slider_float("Motion cost threshold (per mille)", &p.flow_cost, 1.f, 990.f, phoenix_pipeline::Options{}.flow_cost);
        const char* views[]{"Processed image", "Motion direction and magnitude", "Motion confidence"};
        changed |= choice_buttons("Motion diagnostic view", &p.motion_view, views, 3);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::TextWrapped("Engine vectors require a real producer. Missing or rejected motion uses spatial processing with history reset. Confidence views and rejection thresholds apply when NVIDIA Optical Flow is active.");
        changed |= ImGui::Checkbox("Detect scene changes and reset history", &p.scene_detection);
        if (x86_transport && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", motion_route);
        ImGui::EndDisabled();
    }
    if (section == PipelineSection::Responsiveness || section == PipelineSection::All) {
        ImGui::BeginDisabled(!frame_services && !x86_transport);
        const char* monitoring[]{"Off", "Basic (CPU counters)", "Detailed (sample GPU stages)"};
        const char* monitoring_helper = "In an x86 game, this forces the external x64 helper when the installed in-process runtime lacks frame services.";
        const char* monitoring_notes[]{nullptr, monitoring_helper, monitoring_helper};
        changed |= choice_buttons("Pipeline monitoring", &p.monitoring, monitoring, 3, nullptr, nullptr,
                                  x86_transport ? monitoring_notes : nullptr);
        ImGui::BeginDisabled(!p.monitoring);
        changed |= phoenix_ui::slider_int("GPU monitoring sample interval", &p.sample_interval, 1, 120, phoenix_pipeline::Options{}.sample_interval);
        const char* priority[]{"Keep quality fixed", "Prefer responsiveness", "Prefer image quality"};
        const char* priority_helper = "Forces the external x64 helper in an x86 game, including in Automatic. Other execution mechanisms are locked while active.";
        const char* priority_notes[]{nullptr, priority_helper, priority_helper};
        changed |= choice_buttons("Adaptive priority", &p.priority, priority, 3, nullptr, nullptr,
                                  x86_transport ? priority_notes : nullptr);
        changed |= phoenix_ui::slider_float("Pipeline age budget (ms)", &p.latency_ms, 5.f, 250.f, phoenix_pipeline::Options{}.latency_ms);
        changed |= ImGui::Checkbox("Allow motion resolution adjustment", &p.adjust_flow);
        changed |= phoenix_ui::slider_int("Minimum motion analysis height", &p.minimum_flow_height, 64, 1080, phoenix_pipeline::Options{}.minimum_flow_height);
        changed |= ImGui::Checkbox("Allow neural resolution adjustment", &p.adjust_scale);
        changed |= phoenix_ui::slider_float("Minimum neural scale", &p.minimum_scale, 10.f, 100.f, phoenix_pipeline::Options{}.minimum_scale, "%.0f%%");
        changed |= ImGui::Checkbox("Allow central block adjustment", &p.adjust_blocks);
        changed |= ImGui::Checkbox("Allow network pass adjustment", &p.adjust_passes);
        changed |= phoenix_ui::slider_int("Minimum network passes", &p.minimum_passes, 1, 10, phoenix_pipeline::Options{}.minimum_passes);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
    }
    if (section == PipelineSection::Chain || section == PipelineSection::All) {
        const char* chain_route = "In x86 games, a non-default processing chain forces the external x64 helper if the installed in-process runtime cannot run it. This applies in Automatic and locks the execution mechanism while required.";
        const auto show_chain_route = [&] {
            if (x86_transport && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s", chain_route);
        };
        ImGui::BeginDisabled(!native_chain&&!portable_chain&&!p.chain_enabled);
        changed |= ImGui::Checkbox("Enable processing chain", &p.chain_enabled);
        show_chain_route();
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!p.chain_enabled);
        if(portable_chain)ImGui::TextWrapped(frame_services?
            "This portable route uses the shared native pipeline around your selected backend. Motion falls back when hardware flow is unavailable. NIS supports 1x to 2x; Tiny needs a 50% processing canvas for this fixed-size output.":
            "This portable route supports color/fusion with equal processing/output extents, the existing spatial scaler and fixed pipeline quality. Update/use the x64 final-frame route for the shared pipeline services.");
        if(!native_chain&&!portable_chain)ImGui::TextWrapped("Custom ordering, color and fusion require Native RTX or a running portable runtime with the host-chain capability. Disable the chain or reset its options to clear an unsupported saved recipe.");
        native_chain=native_chain||portable_chain;
        const char* names[]{"Neural rendering", "Scaling", "Color treatment", "Multiscale fusion"};
        ImGui::BeginDisabled(!native_chain);
        for (int row = 0; row != 4; ++row) {
            ImGui::PushID(row);
            const int stage = p.order[unsigned(row)];
            changed |= ImGui::Checkbox(names[stage], &p.enabled[unsigned(stage)]);
            show_chain_route();
            ImGui::SameLine();
            ImGui::BeginDisabled(row == 0);
            if (phoenix_ui::small_button("Move up") && row) {
                std::swap(p.order[unsigned(row)], p.order[unsigned(row - 1)]); changed = true;
            }
            show_chain_route();
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        ImGui::EndDisabled();
        const char* scalers[]{"Existing scaler", "NVIDIA Image Scaling", "RTX Video Super Resolution", "Tiny neural 2x"};
        ImGui::BeginDisabled(!native_chain||managed_resize);
        const bool tiny_available=bool(phoenix_pipeline::find_tiny_model(std::filesystem::u8path(model_library)));
        const uint8_t scaler_available[]{1,uint8_t(!portable_chain||frame_services),0,
            uint8_t(tiny_available&&(!portable_chain||frame_services))};
        const char* scaler_captions[]{"Existing","NIS","RTX VSR","Tiny 2x"};
        const char* scaler_notes[]{nullptr, chain_route, nullptr, chain_route};
        changed |= choice_buttons("Additional upscaler",&p.scaler,scalers,4,scaler_captions,scaler_available,
            x86_transport ? scaler_notes : nullptr);
        if(managed_resize)ImGui::TextWrapped("Resize method is managed in Color and scaling. Choose Manual there to use this additional chain scaler.");
        if(p.scaler)ImGui::TextWrapped("NIS/tiny reconstruction currently uses explicit sRGB input encoding. Choose one reconstruction provider. Tiny models are installed through DLSS Everything.");
        changed |= phoenix_ui::slider_float("Scaling sharpness", &p.sharpness, 0.f, 1.f, phoenix_pipeline::Options{}.sharpness);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!native_chain||!p.enabled[unsigned(phoenix_pipeline::Stage::Color)]);
        changed |= phoenix_ui::slider_float("Chain brightness",&p.color_brightness,-2.f,2.f, phoenix_pipeline::Options{}.color_brightness);
        changed |= phoenix_ui::slider_float("Chain contrast",&p.color_contrast,0.f,4.f, phoenix_pipeline::Options{}.color_contrast);
        changed |= phoenix_ui::slider_float("Chain saturation",&p.color_saturation,0.f,2.f, phoenix_pipeline::Options{}.color_saturation);
        changed |= phoenix_ui::slider_float("Chain gamma",&p.color_gamma,.1f,4.f, phoenix_pipeline::Options{}.color_gamma);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!native_chain);
        changed |= ImGui::Checkbox("Experimental multiscale fusion", &p.fusion);
        show_chain_route();
        ImGui::BeginDisabled(!p.fusion);
        changed |= phoenix_ui::slider_int("Fusion layers", &p.fusion_layers, 2, 3, phoenix_pipeline::Options{}.fusion_layers);
        changed |= phoenix_ui::slider_float("Second layer scale", &p.fusion_second_scale, .1f, 1.f, phoenix_pipeline::Options{}.fusion_second_scale);
        changed |= phoenix_ui::slider_float("Third layer scale", &p.fusion_third_scale, .1f, 1.f, phoenix_pipeline::Options{}.fusion_third_scale);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::EndDisabled();
    }
    if (section == PipelineSection::Inspection || section == PipelineSection::All) {
        changed |= ImGui::Checkbox("Comparison loupe", &p.loupe);
        changed |= phoenix_ui::slider_float("Comparison zoom", &p.compare_zoom, 1.f, 8.f, phoenix_pipeline::Options{}.compare_zoom);
        changed |= phoenix_ui::slider_float("Comparison focus X", &p.loupe_x, 0.f, 1.f, phoenix_pipeline::Options{}.loupe_x);
        changed |= phoenix_ui::slider_float("Comparison focus Y", &p.loupe_y, 0.f, 1.f, phoenix_pipeline::Options{}.loupe_y);
        changed |= ImGui::Checkbox("Save screenshot settings receipt", &p.screenshot_receipt);
    }
    if (section == PipelineSection::Recovery || section == PipelineSection::All) {
        changed |= ImGui::Checkbox("Restore last working settings after initialization failure", &p.recovery);
        if(phoenix_ui::button("Reset motion and pipeline options")){p={};changed=true;}
    }
    return changed;
}
} // namespace phoenix_ui
