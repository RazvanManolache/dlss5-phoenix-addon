// One set of widgets for both addon bitnesses. Runtime adapters own actions;
// this view consumes settings and plain metadata, with no inference ownership.
#pragma once
#include <imgui.h>
#include "../../native/support/comparison.h"
#include "../../native/support/cpu_threads.h"
#include "../../native/support/model_tools.h"
#include "../../native/model/model_library.h"
#include "../../native/support/panel_visibility.h"
#include "panel_contract.h"
#include <cfloat>
#include <functional>
#include <optional>
namespace phoenix_ui {
namespace phoenix = phoenix_addon::phoenix;
using Json = nlohmann::json;
using Settings = phoenix::Settings;
struct Preview {
    phoenix::Resolution geometry;
    std::string upscale;
    const phoenix::Resolution &resolution() const { return geometry; }
    const std::string &upscale_status() const { return upscale; }
};
struct BudgetView {
    double estimate = 0;
    double ema() const { return estimate; }
};
struct ValidationView {
    std::function<phoenix_model_tools::Result()> poll;
    std::function<void(const phoenix_model::WeightSource &, const std::filesystem::path &)>
        validate;
};
struct TimingView {
    Json data = {{"rows", Json::array()}};
    std::vector<float> samples;
    Json report() const { return data; }
    const std::vector<float> &history() const { return samples; }
    size_t history_count() const { return samples.size(); }
    size_t history_offset() const { return 0; }
};
enum class PanelPage { Play, Image, Performance, Setup, All };
struct Context {
    Settings &panel;
    phoenix_addon::PhoenixLiveControls &live;
    std::array<Settings, 4> &presets;
    std::array<int, 7> &hotkeys;
    int &capture_hotkey;
    uint64_t &hotkey_deadline;
    phoenix_model_tools::Task &model_picker;
    std::vector<phoenix::ExecutionDevice> &execution_devices;
    std::string &error;
    uint64_t &markers;
    bool final_frame_source = false, host_controls_stage = false;
    bool direct_game = true, direct_nr_ready = true, runtime_available = true;
    bool real_guides = true, temporal_history = true, comparison_views = true, input_capture = true;
    bool fps_budget = true;
    bool portable_backbuffer = false;
    bool portable_chain_available = false;
    bool portable_backend_selected = false;
    bool native_chain_available = true, frame_services_available = false;
    bool portable_gpu_budget = false;
    bool fixed_native_blocks = false;
    bool* processing_enabled = nullptr;
    bool processing_changed = false, live_selection_changed = false, live_tuning_changed = false;
    std::string nvidia_status;
    Json appearance = Json::object();
    Json pipeline_status = Json::object();
    Json native_kernel_styles = Json::object();
    // GPU timings belong to the queue that measured them, which may be the
    // hidden companion rather than the game's rendering queue.
    std::string timing_provenance = "Game rendering GPU";
    bool inference_process = false, telemetry_available = true;
    std::optional<PanelPage> requested_page;
    std::string status;
    bool screenshot_pair_available = false;
    std::string screenshot_pair_status;
    std::string active_route, active_device, route_notice;
    bool processing_active = false, route_failed = false;
    std::function<void()> setup_details, performance_details;
    std::function<void()> retry;
    std::string loaded_model_id, model_switch_error;
    std::string model_restart = "Model selection saved. Restart the game to load it.";
    std::optional<Preview> direct_runner;
    BudgetView depth_budget;
    ValidationView model_validation;
    phoenix_model::WeightSource configured_weight_source;
    std::filesystem::path model_path, runtime_root, bundle_path, overrides_path;
    bool disk_model = false;
    std::string active_model_source, active_model_folder, active_model_dll, active_model_id, active_nvidia_dll_folder, override_status;
    uint64_t live_resets_recorded = 0, switches = 0;
    unsigned last_transformer_blocks = 8;
    int last_mode = -1;
    double present_fps = 0;
    std::optional<double> portable_wall_ms;
    std::optional<TimingView> live_timing;
    TimingView worker_timing;
    std::array<uint64_t, 4> mode_frames{}, mode_own{}, mode_native{};
    std::function<void()> save, persist, reload, marker;
    std::function<void(unsigned)> capture;
    std::function<void(Action)> action;
};
// Retained timing rows are valuable comparison evidence. Clearing needs an
// explicit confirmation; ordinary temporal-history resets remain immediate.
inline bool confirm_clear_timing(const char* label) {
    if (ImGui::Button(label)) ImGui::OpenPopup("Clear retained timings?");
    bool confirmed = false;
    ImGui::SetNextWindowSizeConstraints(ImVec2(240, 0), ImVec2(std::max(240.f, ImGui::GetContentRegionAvail().x), FLT_MAX));
    if (ImGui::BeginPopupModal("Clear retained timings?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Remove retained GPU and worker timing comparisons? Image history and settings are kept.");
        if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ImGui::Button("Clear timings")) { confirmed = true; ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    return confirmed;
}
inline void persist_configuration(Context &context) {
    if (context.persist)
        context.persist();
    else if (context.save)
        context.save();
}
inline bool load_preset(Context &context, unsigned slot) {
    if (slot >= context.presets.size() || !context.presets[slot].valid())
        return false;
    context.panel = context.presets[slot];
    return true;
}
inline bool store_preset(Context &context, unsigned slot) {
    if (slot >= context.presets.size() || !context.panel.valid())
        return false;
    context.presets[slot] = context.panel;
    persist_configuration(context);
    return true;
}
inline void clear_binding(Context &context, unsigned slot) {
    if (slot >= context.hotkeys.size())
        return;
    context.hotkeys[slot] = 0;
    persist_configuration(context);
}
} // namespace phoenix_ui
