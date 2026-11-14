#pragma once
#include "widgets.h"
#include "panel_sections.h"

namespace phoenix_ui {
inline void draw_geometry_summary(const Context& context) {
    if (!context.direct_runner) return;
    const auto& geometry = context.direct_runner->resolution();
    ImGui::Text("%u x %u  ->  %u x %u", geometry.width, geometry.height,
                geometry.output_width, geometry.output_height);
    if (geometry.left || geometry.top)
        ImGui::TextDisabled("Image %u x %u / padding %u, %u", geometry.image_width,
                            geometry.image_height, geometry.left, geometry.top);
}
struct QuickMeasurements { std::optional<double> neural, route; };
inline QuickMeasurements quick_measurements(const Context& context) {
    std::optional<double> neural, route;
    if (context.live_timing) {
        const auto& data = context.live_timing->data;
        if (data.contains("active_settings"))
            for (const auto& row : data.at("rows")) {
                if (row.value("mode", -1) != context.last_mode ||
                    row.value("transformer_blocks", 0u) != context.last_transformer_blocks ||
                    row.at("settings") != data.at("active_settings") || !row.contains("recent")) continue;
                const auto& recent = row.at("recent");
                if (recent.value("samples", uint64_t(0))) route = recent.value("route_ms", 0.);
                const bool nvidia = context.last_mode == phoenix::nvidia;
                if (recent.value(nvidia ? "nvidia_samples" : "phoenix_samples", uint64_t(0)))
                    neural = recent.value(nvidia ? "nvidia_ms" : "phoenix_ms", 0.);
                break;
            }
    }
    return {neural, route};
}
inline void draw_fact_tiles(const Context& context) {
    const auto measured = quick_measurements(context);
    char neural[48] = "--", fps[48] = "--", processed_fps[48] = "--", canvas[48] = "--", roundtrip[48] = "--";
    if (measured.neural) snprintf(neural,sizeof neural,"%.2f ms",*measured.neural);
    else if (context.portable_wall_ms) snprintf(neural,sizeof neural,"%.1f ms",*context.portable_wall_ms);
    if (context.present_fps > 0) snprintf(fps,sizeof fps,"%.1f FPS",context.present_fps);
    if (context.pipeline_roundtrip_ms)
        snprintf(roundtrip, sizeof roundtrip, "%.1f ms", *context.pipeline_roundtrip_ms);
    if (const auto rate = fresh_dlss5_fps(context))
        snprintf(processed_fps, sizeof processed_fps, "%.1f FPS", *rate);
    if (context.direct_runner) {
        const auto& geometry = context.direct_runner->resolution();
        snprintf(canvas,sizeof canvas,"%ux%u",geometry.width,geometry.height);
    }
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float available = ImGui::GetContentRegionAvail().x;
    const int columns = available >= ImGui::GetFontSize() * 60.f ? 5 : 2;
    const float width = std::max(1.f, (available - (columns - 1) * gap) / columns);
    fact_tile("Game FPS", fps, width, mint,
        "Game presentation rate, including repeats of the last processed image.");
    ImGui::SameLine(0, gap);
    fact_tile("Fresh DLSS5 FPS", processed_fps, width, mint,
        "Game presentations with a new DLSS5 result per second. Excludes held and reused results. -- means off, unavailable, or waiting for a full sample.");
    if (columns == 5) ImGui::SameLine(0, gap);
    fact_tile(context.portable_wall_ms ? "Worker wall" : "Neural GPU", neural, width, blue,
        "Recent neural GPU interval for the active workload. Worker wall time includes host preparation/inference/resolve. Missing measurements are --.");
    ImGui::SameLine(0, gap);
    fact_tile(width < 145.f ? "Canvas" : "Processing canvas",canvas,width,violet,
        "Current loaded runner canvas, including internal padding. Output geometry is shown in the status card.");
    if (columns == 5) ImGui::SameLine(0, gap);
    fact_tile("Pipeline round trip", roundtrip, width, blue,
        "Elapsed time from capture of the source frame through preparation, processing, transfers, and composition for presentation. Held frames are excluded. This does not measure display scanout.");
    ImGui::Spacing();
}
inline void draw_metrics(const Context& context) {
    const auto [neural, route] = quick_measurements(context);
    ImGui::Separator();
    const int columns = ImGui::GetContentRegionAvail().x >= 420.f ? 3 : 1;
    if (ImGui::BeginTable("quick_metrics", columns, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        if (context.present_fps > 0) ImGui::Text("Game %.1f FPS", context.present_fps);
        else ImGui::TextDisabled("Game FPS  --");
        ImGui::TableNextColumn();
        if (neural) ImGui::Text("Neural GPU %.2f ms", *neural);
        else if (context.portable_wall_ms) ImGui::Text("Worker %.2f ms", *context.portable_wall_ms);
        else ImGui::TextDisabled("Neural GPU  --");
        ImGui::TableNextColumn();
        if (route) ImGui::Text("Addon GPU %.2f ms", *route);
        else ImGui::TextDisabled("Addon GPU  --");
        ImGui::EndTable();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Game FPS counts presentation callbacks. GPU intervals exclude the rest of the game.\n"
        "Worker wall time includes its host processing. Missing or stale measurements show --.");
}
inline bool draw_header(Context& context, bool& advanced) {
    ImGui::TextDisabled("%s%s%s", context.game_api.c_str(), context.game_api.empty() ? "" : " / ",
                        sizeof(void*) == 4 ? "x86 game" : "x64 game");
    const float available = ImGui::GetContentRegionAvail().x;
    if (!context.update_available.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0,0,0,0));
        ImGui::PushStyleColor(ImGuiCol_Text, rgb(mint));
        const std::string label = "Update available: " + context.update_available + "  -  Open GitHub";
        if (phoenix_ui::small_button(label.c_str(),"Open the available Phoenix release page. Installing an update is a separate action.") && context.open_update) context.open_update();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open the Phoenix releases page. Nothing is downloaded or installed automatically.");
        ImGui::PopStyleColor(2);
    }
    heading("DLSS5 Phoenix",0xecf0f5,1.3f);
    if (available >= 420.f) ImGui::SameLine(std::max(150.f, available - 245.f));
    ImGui::PushStyleColor(ImGuiCol_Button,rgb(0x365b84));
    ImGui::PushStyleColor(ImGuiCol_Border,rgb(0x6f9fd1));
    if (phoenix_ui::button("Save configuration")) {
        if (context.save) context.save();
        ImGui::GetStateStorage()->SetFloat(ImGui::GetID("last_save"),float(ImGui::GetTime()));
    }
    ImGui::PopStyleColor(2);
    if (available >= 330.f) ImGui::SameLine();
    const float saved = ImGui::GetStateStorage()->GetFloat(ImGui::GetID("last_save"),-100.f);
    ImGui::TextColored(rgb(mint),"%s",ImGui::GetTime()-saved < 4. ? "Saved" : "Autosave on");
    const bool enabled = context.processing_enabled ? *context.processing_enabled : context.panel.uplift;
    const bool failed = context.route_failed || !context.error.empty() || !context.model_switch_error.empty();
    const bool active = enabled && context.processing_active && context.last_mode != phoenix::input;
    const auto state = failed ? "Needs attention" : !enabled ? "Off / original image"
        : context.last_mode == phoenix::input ? "Input / passthrough" : active ? "Processing" : "Starting / waiting";
    {
        PanelCard status("status_card","",nullptr,failed ? amber : active ? mint : blue,
            failed ? 0x382c25 : active ? 0x1d302a : 0x222c38);
        if (status) {
            heading(state,failed ? amber : active ? mint : blue,1.15f);
            if (available > 500.f) { ImGui::SameLine(); ImGui::TextDisabled("  /  %u-bit addon",unsigned(sizeof(void*)*8)); }
            if (!context.active_route.empty()) ImGui::TextWrapped("%s",context.active_route.c_str());
            if (!context.loaded_model_id.empty()) {
                ImGui::TextColored(rgb(mint),"Loaded: %s",std::string(phoenix_model::definition(context.loaded_model_id).label).c_str());
                if (available > 600.f && !context.active_device.empty()) ImGui::SameLine();
            }
            if (!context.active_device.empty()) ImGui::TextDisabled("%s",context.active_device.c_str());
            draw_geometry_summary(context);
            if (!context.route_notice.empty()) ImGui::TextWrapped("%s",context.route_notice.c_str());
            if (!context.error.empty()) ImGui::TextWrapped("%s",context.error.c_str());
            if (!context.model_switch_error.empty()) ImGui::TextWrapped("%s",context.model_switch_error.c_str());
            if (failed && context.retry && phoenix_ui::button("Retry rendering")) context.retry();
        }
    }
    draw_fact_tiles(context);
    const bool changed = draw_master_switch(context);
    if (available > 500.f) ImGui::SameLine();
    ImGui::Checkbox("Advanced details",&advanced);
    return changed;
}
} // namespace phoenix_ui
