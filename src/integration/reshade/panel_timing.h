#pragma once
#include "panel_context.h"
namespace phoenix_ui {
inline void draw_timing(Context &addon_state) {
    if (addon_state.portable_wall_ms) {
        ImGui::Text("Game presentation: %.1f FPS", addon_state.present_fps);
        ImGui::Text("Last processed frame: %.3f ms (host + inference + resolve)", *addon_state.portable_wall_ms);
        ImGui::TextWrapped("Timing source: %s", addon_state.timing_provenance.c_str());
        if (confirm_clear_timing("Clear processing timing")) addon_state.action(Action::ClearTiming);
        return;
    }
    if (addon_state.live_timing) {
        ImGui::Text("Presentation rate: %.1f FPS (%.2f ms between presents)",
                    addon_state.present_fps,
                    addon_state.present_fps > 0 ? 1000.0 / addon_state.present_fps : 0);
        {
            const auto recent_report = addon_state.live_timing->report();
            bool recent_found = false;
            for (const auto &row : recent_report.at("rows"))
                if (row.at("mode").get<unsigned>() == unsigned(addon_state.last_mode) &&
                    row.at("transformer_blocks").get<unsigned>() ==
                        addon_state.last_transformer_blocks &&
                    row.at("settings") == recent_report.at("active_settings") &&
                    row.at("recent").at("samples").get<uint64_t>()) {
                    const auto &recent = row.at("recent");
                    char recent_line[192]{};
                    snprintf(
                        recent_line, sizeof(recent_line),
                        "Networks last 1s: Phoenix DLSS5 %.3f ms (%llu) | NVIDIA DLSS5 %.3f ms (%llu)",
                        recent.at("phoenix_ms").get<double>(),
                        recent.at("phoenix_samples").get<uint64_t>(),
                        recent.at("nvidia_ms").get<double>(),
                        recent.at("nvidia_samples").get<uint64_t>());
                    ImGui::TextUnformatted(recent_line);
                    recent_found = true;
                    break;
                }
            if (!recent_found)
                ImGui::TextUnformatted("Networks last 1s: --");
        }
        if (confirm_clear_timing("Clear GPU timing")) {
            addon_state.action(Action::ClearTiming);
        }
        if (!addon_state.worker_timing.report().at("rows").empty()) {
            ImGui::TextUnformatted("Other processing devices (wall time, ms)");
            ImGui::TextWrapped("Rows retain completed results across device switches. Execute "
                               "includes transfers; result age includes queueing. These are "
                               "separate from the render-GPU timings below.");
            if (ImGui::BeginTable("phoenix_workers", 8,
                                  ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                                  ImVec2(0,std::min(360.f,65.f + 75.f * float(addon_state.worker_timing.report().at("rows").size()))))) {
                for (const char *title : {"Device / method", "Results", "Worker ms", "Prepare",
                                          "Execute", "Resolve", "Age ms", "State"})
                    ImGui::TableSetupColumn(title);
                ImGui::TableHeadersRow();
                const auto workers_report = addon_state.worker_timing.report();
                for (const auto &row : workers_report.at("rows")) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextWrapped("%s", row.at("label").get<std::string>().c_str());
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", row.at("key").get<std::string>().c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%llu", row.at("completed").get<uint64_t>());
                    for (const char *key : {"worker", "prepare", "execute", "resolve", "age"}) {
                        ImGui::TableNextColumn();
                        const auto &metric = row.at(key);
                        if (metric.at("samples").get<uint64_t>())
                            ImGui::Text("%.2f", metric.at("mean_ms").get<double>());
                        else
                            ImGui::TextUnformatted("--");
                    }
                    ImGui::TableNextColumn();
                    ImGui::TextWrapped("%s", row.at("state").get<std::string>().c_str());
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Actual method: %s\nErrors: %llu\n%s",
                                          row.at("actual_engine").get<std::string>().c_str(),
                                          row.at("failed").get<uint64_t>(),
                                          row.at("error").get<std::string>().c_str());
                }
                ImGui::EndTable();
            }
        }
        ImGui::TextWrapped("Timing source: %s", addon_state.timing_provenance.c_str());
        ImGui::TextUnformatted(
            (!phoenix::uses_ours(addon_state.panel.mode) ||
             addon_state.panel.execution_device == "native")
                ? "GPU intervals; retained rows identify their processing device"
                : "Measured queue: worker capture/compose only; inference is in the table above");
        if (addon_state.inference_process)
            ImGui::TextWrapped("Helper GPU timings exclude the game's capture/upload/display work. "
                               "Helper round trip and repeated display of a held result do not "
                               "measure newly processed frames per second.");
        ImGui::TextWrapped("Timing history survives route, runner and settings changes. Each row "
                           "identifies its workload. Clear timing removes the retained results.");
        ImGui::TextWrapped("Recent %s route intervals (ms)", addon_state.timing_provenance.c_str());
        ImGui::PushItemWidth(-1);
        ImGui::PlotLines("##GPU_route_ms", addon_state.live_timing->history().data(),
                         int(addon_state.live_timing->history_count()),
                         int(addon_state.live_timing->history_offset()), nullptr, 0.f, FLT_MAX,
                         ImVec2(0, 70));
        ImGui::PopItemWidth();
        if (ImGui::BeginTable("phoenix_gpu", 5,
                              ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                              ImVec2(0,std::min(360.f,65.f + 90.f * float(addon_state.live_timing->data.at("rows").size()))))) {
            ImGui::TableSetupColumn("Route");
            ImGui::TableSetupColumn("Samples");
            ImGui::TableSetupColumn("Phoenix DLSS5 ms");
            ImGui::TableSetupColumn("NVIDIA DLSS5 ms");
            ImGui::TableSetupColumn("Route ms");
            ImGui::TableHeadersRow();
            const auto live_report = addon_state.live_timing->report();
            for (const auto &row : live_report.at("rows")) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                const char *labels[]{"Input", "NVIDIA DLSS5", "Split", "Phoenix DLSS5"};
                ImGui::TextWrapped("%s / %s / %u blocks", labels[row.at("mode").get<unsigned>()],
                                   row.at("execution_device").get<std::string>().c_str(),
                                   row.at("transformer_blocks").get<unsigned>());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%u x %u\n%s", row.at("width").get<unsigned>(),
                                      row.at("height").get<unsigned>(),
                                      row.at("settings").dump(2).c_str());
                const auto &context = row.at("settings");
                ImGui::TextDisabled(
                    "%u x %u -> %u x %u",
                    context.value("InferenceWidth", row.at("width").get<unsigned>()),
                    context.value("InferenceHeight", row.at("height").get<unsigned>()),
                    row.at("width").get<unsigned>(), row.at("height").get<unsigned>());
                ImGui::TextDisabled("%s", context.value("UpscalerStatus", std::string()).c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%llu", row.at("route").at("samples").get<uint64_t>());
                for (const char *key : {"phoenix", "nvidia", "route"}) {
                    ImGui::TableNextColumn();
                    const auto &value = row.at(key);
                    if (value.at("samples").get<uint64_t>())
                        ImGui::Text("%.3f", value.at("mean_ms").get<double>());
                    else
                        ImGui::TextUnformatted("--");
                }
            }
            ImGui::EndTable();
        }
        if (phoenix::PanelVisibility{addon_state.panel}.own_details() &&
            ImGui::CollapsingHeader("GPU stage breakdown")) {
            const auto report = addon_state.live_timing->report();
            for (const auto &row : report.at("rows"))
                if (row.at("mode") == addon_state.last_mode &&
                    row.at("transformer_blocks") == addon_state.last_transformer_blocks &&
                    row.at("settings") == report.at("active_settings"))
                    for (const auto &group : row.at("groups"))
                        ImGui::Text("%s: %.3f ms", group.at("group").get<std::string>().c_str(),
                                    group.at("gpu").at("mean_ms").get<double>());
        }
    }
}
// Both addons also share the small panel in ReShade's general overlay. The
// adapter persists a changed route and handles model ownership/reconfiguration.
} // namespace phoenix_ui
