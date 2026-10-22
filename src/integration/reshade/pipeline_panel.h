#pragma once
#include <imgui.h>
#include "../../pipeline/options.h"
#include "../../pipeline/tiny_model.h"

namespace phoenix_ui {
inline bool draw_pipeline_options(phoenix_pipeline::Options& p,const std::string& model_library={},bool native_chain=true,bool portable_chain=false,bool frame_services=true) {
    if (!ImGui::CollapsingHeader("Motion, latency and processing chain")) return false;
    if(!frame_services&&!portable_chain)ImGui::TextWrapped("Flow, scene analysis and pipeline-age controls require the local final-frame D3D12 bridge. This route uses its supplied guides and reset flags; its neural budget controls are separate.");
    ImGui::BeginDisabled(!frame_services);
    bool changed = ImGui::Checkbox("Estimate motion when engine vectors are unavailable", &p.motion_enabled);
    ImGui::BeginDisabled(!p.motion_enabled);
    const char* providers[]{"Automatic (prefer engine vectors)", "Engine vectors", "NVIDIA Optical Flow", "Zero motion"};
    changed |= ImGui::Combo("Motion source", &p.motion_source, providers, 4);
    changed |= ImGui::SliderInt("Motion analysis height", &p.flow_height, 64, 1080);
    changed |= ImGui::SliderFloat("Motion consistency tolerance (pixels)", &p.flow_consistency, .1f, 20.f);
    changed |= ImGui::SliderFloat("Motion cost threshold (per mille)", &p.flow_cost, 1.f, 990.f);
    const char* views[]{"Processed image", "Motion direction and magnitude", "Motion confidence"};
    changed |= ImGui::Combo("Motion diagnostic view", &p.motion_view, views, 3);
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Detect scene changes and reset history", &p.scene_detection);
    const char* monitoring[]{"Off", "Basic (CPU counters)", "Detailed (sample GPU stages)"};
    changed |= ImGui::Combo("Pipeline monitoring", &p.monitoring, monitoring, 3);
    ImGui::BeginDisabled(!p.monitoring);
    changed |= ImGui::SliderInt("GPU monitoring sample interval", &p.sample_interval, 1, 120);
    const char* priority[]{"Keep quality fixed", "Prefer responsiveness", "Prefer image quality"};
    changed |= ImGui::Combo("Adaptive priority", &p.priority, priority, 3);
    changed |= ImGui::SliderFloat("Pipeline age budget (ms)", &p.latency_ms, 5.f, 250.f);
    changed |= ImGui::Checkbox("Allow motion resolution adjustment", &p.adjust_flow);
    changed |= ImGui::SliderInt("Minimum motion analysis height", &p.minimum_flow_height, 64, 1080);
    changed |= ImGui::Checkbox("Allow neural resolution adjustment", &p.adjust_scale);
    changed |= ImGui::SliderFloat("Minimum neural scale", &p.minimum_scale, 10.f, 100.f, "%.0f%%");
    changed |= ImGui::Checkbox("Allow central block adjustment", &p.adjust_blocks);
    changed |= ImGui::Checkbox("Allow network pass adjustment", &p.adjust_passes);
    changed |= ImGui::SliderInt("Minimum network passes", &p.minimum_passes, 1, 10);
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Restore last working settings after initialization failure", &p.recovery);
    ImGui::BeginDisabled(!native_chain&&!portable_chain&&!p.chain_enabled);
    changed |= ImGui::Checkbox("Enable processing chain", &p.chain_enabled);
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
        ImGui::SameLine();
        ImGui::BeginDisabled(row == 0);
        if (ImGui::SmallButton("Move up") && row) {
            std::swap(p.order[unsigned(row)], p.order[unsigned(row - 1)]); changed = true;
        }
        ImGui::EndDisabled();
        ImGui::PopID();
    }
    ImGui::EndDisabled();
    const char* scalers[]{"Existing scaler", "NVIDIA Image Scaling", "RTX Video Super Resolution", "Tiny neural 2x"};
    ImGui::BeginDisabled(!native_chain);
    if (ImGui::BeginCombo("Additional upscaler", scalers[std::clamp(p.scaler,0,3)])) {
        const bool tiny_available=bool(phoenix_pipeline::find_tiny_model(std::filesystem::u8path(model_library)));
        for (int provider : {0,1,3}) {
            if(portable_chain&&!frame_services&&provider!=0)continue;
            if(provider==3&&!tiny_available)continue;
            if (ImGui::Selectable(scalers[provider], p.scaler == provider)) { p.scaler=provider;changed=true; }
        }
        ImGui::EndCombo();
    }
    if(p.scaler)ImGui::TextWrapped("NIS/tiny reconstruction currently uses explicit sRGB input encoding. Choose one reconstruction provider. Tiny models are installed through DLSS Everything.");
    changed |= ImGui::SliderFloat("Scaling sharpness", &p.sharpness, 0.f, 1.f);
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!native_chain||!p.enabled[unsigned(phoenix_pipeline::Stage::Color)]);
    changed |= ImGui::SliderFloat("Chain brightness",&p.color_brightness,-2.f,2.f);
    changed |= ImGui::SliderFloat("Chain contrast",&p.color_contrast,0.f,4.f);
    changed |= ImGui::SliderFloat("Chain saturation",&p.color_saturation,0.f,2.f);
    changed |= ImGui::SliderFloat("Chain gamma",&p.color_gamma,.1f,4.f);
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!native_chain);
    changed |= ImGui::Checkbox("Experimental multiscale fusion", &p.fusion);
    ImGui::BeginDisabled(!p.fusion);
    changed |= ImGui::SliderInt("Fusion layers", &p.fusion_layers, 2, 3);
    changed |= ImGui::SliderFloat("Second layer scale", &p.fusion_second_scale, .1f, 1.f);
    changed |= ImGui::SliderFloat("Third layer scale", &p.fusion_third_scale, .1f, 1.f);
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Comparison loupe", &p.loupe);
    changed |= ImGui::SliderFloat("Comparison zoom", &p.compare_zoom, 1.f, 8.f);
    changed |= ImGui::SliderFloat("Comparison focus X", &p.loupe_x, 0.f, 1.f);
    changed |= ImGui::SliderFloat("Comparison focus Y", &p.loupe_y, 0.f, 1.f);
    changed |= ImGui::Checkbox("Save screenshot settings receipt", &p.screenshot_receipt);
    if(ImGui::Button("Reset motion and pipeline options")){p={};changed=true;}
    return changed;
}
} // namespace phoenix_ui
