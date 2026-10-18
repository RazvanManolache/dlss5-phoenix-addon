// One set of widgets for both addon bitnesses. Runtime adapters own actions;
// this view consumes settings and plain metadata, with no inference ownership.
#pragma once
#include <imgui.h>
#include "../../native/support/comparison.h"
#include "../../native/support/cpu_threads.h"
#include "../../native/support/model_tools.h"
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
struct Context {
    Settings &panel;
    phoenix_addon::PhoenixLiveControls &live;
    std::array<Settings, 4> &presets;
    std::array<int, 6> &hotkeys;
    int &capture_hotkey;
    uint64_t &hotkey_deadline;
    phoenix_model_tools::Task &model_picker;
    std::vector<phoenix::ExecutionDevice> &execution_devices;
    std::string &error;
    uint64_t &markers;
    bool direct_game = true, direct_nr_ready = true, runtime_available = true;
    bool real_guides = true, temporal_history = true, comparison_views = true, input_capture = true;
    bool fps_budget = true;
    bool portable_backbuffer = false;
    bool fixed_native_blocks = false;
    bool* processing_enabled = nullptr;
    bool processing_changed = false, live_selection_changed = false, live_tuning_changed = false;
    std::string nvidia_status;
    Json appearance = Json::object();
    // GPU timings belong to the queue that measured them, which may be the
    // hidden companion rather than the game's rendering queue.
    std::string timing_provenance = "Game rendering GPU";
    bool inference_process = false, telemetry_available = true;
    std::string status;
    std::string model_restart = "Model selection saved. Restart the game to load it.";
    std::optional<Preview> direct_runner;
    BudgetView depth_budget;
    ValidationView model_validation;
    phoenix_model::WeightSource configured_weight_source;
    std::filesystem::path model_path, runtime_root, bundle_path, overrides_path;
    bool disk_model = false;
    std::string active_model_source, active_model_folder, active_model_dll, override_status;
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
inline bool draw_controls(Context &addon_state) {
    bool changed = false;
    auto before = addon_state.panel;
    auto& enabled = addon_state.processing_enabled ? *addon_state.processing_enabled : addon_state.panel.uplift;
    addon_state.processing_changed = ImGui::Checkbox("Enable neural rendering", &enabled);
    changed |= addon_state.processing_changed;
    if (!enabled) ImGui::TextDisabled("Off - showing original game image");
    const auto &model_path = addon_state.model_path;
    const bool disk_model = addon_state.disk_model;
    const char *route_labels[]{"Input",
                               (addon_state.direct_game && !addon_state.direct_nr_ready &&
                                !addon_state.nvidia_status.empty())
                                   ? "NVIDIA DLSS5 DLL (NR unavailable)"
                                   : "NVIDIA DLSS5 DLL",
                               "Comparison", "Phoenix DLSS5"};
    // All routes are always offered; an NVIDIA route without a live feature
    // defers honestly with the reason in status instead of hiding.
    changed = ImGui::Combo("Backend", &addon_state.panel.mode, route_labels, 4) || changed;
    if (addon_state.direct_game)
        changed = phoenix::normalize_direct_comparison(addon_state.panel) || changed;
    if (addon_state.direct_game && !addon_state.direct_nr_ready &&
        (addon_state.panel.mode == phoenix::nvidia || addon_state.panel.mode == phoenix::split) &&
        !addon_state.nvidia_status.empty())
        ImGui::TextWrapped("NVIDIA DLSS5 NR: %s", addon_state.nvidia_status.c_str());
    const phoenix::PanelVisibility visible{addon_state.panel};
    if (visible.neural() && addon_state.direct_game) {
        ImGui::BeginDisabled(addon_state.portable_backbuffer);
        const char *modes[]{"Normal", "Super resolution"};
        changed = ImGui::Combo("Resolution mode", &addon_state.panel.resolution_mode, modes, 2) ||
                  changed;
        if (addon_state.panel.resolution_mode) {
            const char *upscalers[]{"Spatial", "NVIDIA DLSS Super Resolution", "Intel XeSS"};
            if (ImGui::BeginCombo("Upscaler", upscalers[addon_state.panel.upscaler])) {
                for (int choice=0; choice<3; ++choice) {
                    ImGui::BeginDisabled(choice != 0 && !addon_state.real_guides);
                    if (ImGui::Selectable(upscalers[choice], addon_state.panel.upscaler == choice)) {
                        addon_state.panel.upscaler=choice;
                        changed=true;
                    }
                    ImGui::EndDisabled();
                }
                ImGui::EndCombo();
            }
            if (!addon_state.real_guides)
                ImGui::TextWrapped("Only spatial upscaling is available on this color-only capture route. Temporal upscalers need engine depth and motion guides.");
            if (addon_state.panel.upscaler) {
                ImGui::TextWrapped(
                    "Temporal upscaling runs on the rendering GPU and requires observed depth and "
                    "motion. Unsupported runtimes or dimensions use a visible spatial fallback.");
                ImGui::TextWrapped("Runtime beside the game: %s",
                                   addon_state.panel.upscaler == 1
                                       ? "nvngx_dlss.dll"
                                       : "libxess.dll and its dependencies");
            }
            if (addon_state.direct_runner)
                ImGui::TextWrapped("%s", addon_state.direct_runner->upscale_status().c_str());
            const char *qualities[]{"Quality (2/3 scale)", "Performance (1/2 scale)",
                                    "Ultra performance (1/3 scale)"};
            const char *aspects[]{"Game aspect", "16:9", "16:10", "21:9", "4:3", "Custom ratio"};
            changed = ImGui::Combo("Processing quality", &addon_state.panel.resolution_quality,
                                   qualities, 3) ||
                      changed;
            changed = ImGui::Combo("Processing aspect", &addon_state.panel.resolution_aspect,
                                   aspects, 6) ||
                      changed;
            if (addon_state.panel.resolution_aspect == 5) {
                int ratio[]{addon_state.panel.resolution_ratio_width,
                            addon_state.panel.resolution_ratio_height};
                if (ImGui::InputInt2("Aspect width : height", ratio)) {
                    addon_state.panel.resolution_ratio_width = std::clamp(ratio[0], 1, 100);
                    addon_state.panel.resolution_ratio_height = std::clamp(ratio[1], 1, 100);
                    changed = true;
                }
            }
            ImGui::TextWrapped(
                "Runs neural rendering at a smaller resolution, then uses the selected upscaler. "
                "Padding preserves the game's image aspect. Native Phoenix DLSS5 generates a plan for "
                "the requested resolution and pads its internal tiles.");
        }
        if (addon_state.direct_runner) {
            const auto &geometry = addon_state.direct_runner->resolution();
            ImGui::Text("Processing: %u x %u (image %u x %u) | Output: %u x %u", geometry.width,
                        geometry.height, geometry.image_width, geometry.image_height,
                        geometry.output_width, geometry.output_height);
            const auto requested = phoenix::choose_resolution(
                geometry.output_width, geometry.output_height, addon_state.panel.resolution_mode,
                addon_state.panel.resolution_quality, addon_state.panel.resolution_aspect,
                addon_state.panel.resolution_ratio_width, addon_state.panel.resolution_ratio_height,
                phoenix::uses_ours(addon_state.panel.mode) &&
                    addon_state.panel.execution_device == "native");
            if (requested.fallback)
                ImGui::TextWrapped(
                    "No smaller supported canvas at this quality; using Normal resolution.");
        }
        ImGui::EndDisabled();
        if (addon_state.portable_backbuffer && addon_state.panel.resolution_mode &&
            ImGui::Button("Use normal resolution for portable processing")) {
            addon_state.panel.resolution_mode = 0;
            changed = true;
        }
    }
    if (visible.own_details()) {
        const char *model_sources[]{"Configured source", "Model folder on disk", "NVIDIA DLSS5 DLL file"};
        int model_selection = addon_state.panel.model_source == "model_folder" ? 1
                              : addon_state.panel.model_source == "nvidia_dll" ? 2
                                                                               : 0;
        if (ImGui::Combo("Model source", &model_selection, model_sources, 3)) {
            addon_state.panel.model_source = model_selection == 1   ? "model_folder"
                                             : model_selection == 2 ? "nvidia_dll"
                                                                    : "configured";
            changed = true;
        }
        const auto picked = addon_state.model_picker.poll();
        if (!picked.busy && !picked.picked.empty()) {
            if (picked.key == "folder") {
                addon_state.panel.model_folder = picked.picked;
                addon_state.panel.model_source = "model_folder";
                model_selection = 1;
            } else {
                addon_state.panel.model_dll = picked.picked;
                addon_state.panel.model_source = "nvidia_dll";
                model_selection = 2;
            }
            addon_state.model_picker.clear_pick();
            changed = true;
        }
        if (model_selection) {
            auto &selected_path =
                model_selection == 1 ? addon_state.panel.model_folder : addon_state.panel.model_dll;
            std::array<char, 4096> path_text{};
            memcpy(path_text.data(), selected_path.data(),
                   std::min(selected_path.size(), path_text.size() - 1));
            if (ImGui::InputText(model_selection == 1 ? "Model folder" : "NVIDIA DLSS5 DLL",
                                 path_text.data(), path_text.size())) {
                selected_path = path_text.data();
                changed = true;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(picked.busy);
            if (ImGui::Button("Browse..."))
                addon_state.model_picker.browse(model_selection == 1, GetActiveWindow(),
                                                std::filesystem::u8path(selected_path));
            ImGui::EndDisabled();
            if (picked.busy)
                ImGui::TextUnformatted("File browser is open...");
            else if (!picked.passed && !picked.cancelled && !picked.message.empty())
                ImGui::TextWrapped("%s", picked.message.c_str());
        } else
            ImGui::TextWrapped("Model: %s", phoenix_model::source_path_string(model_path).c_str());
        const auto validation = addon_state.model_validation.poll();
        ImGui::BeginDisabled(validation.busy);
        if (ImGui::Button("Validate model")) {
            try {
                addon_state.model_validation.validate(
                    phoenix_model_tools::selection(
                        addon_state.panel.model_source, addon_state.panel.model_folder,
                        addon_state.panel.model_dll, addon_state.configured_weight_source),
                    addon_state.runtime_root);
            } catch (const std::exception &feature_entry) {
                addon_state.error = feature_entry.what();
            }
        }
        ImGui::EndDisabled();
        if (validation.busy)
            ImGui::TextUnformatted("Checking model and running 3 small CPU test frames...");
        else if (!validation.message.empty()) {
            bool current = false;
            try {
                current = validation.key ==
                          phoenix_model_tools::selection(
                              addon_state.panel.model_source, addon_state.panel.model_folder,
                              addon_state.panel.model_dll, addon_state.configured_weight_source)
                              .configuration()
                              .dump();
            } catch (const std::exception &) {
            }
            ImGui::TextWrapped("%s%s",
                               current ? "" : "Previous selection: ", validation.message.c_str());
        }
        if (addon_state.panel.model_source != addon_state.active_model_source ||
            (addon_state.panel.model_folder != addon_state.active_model_folder ||
             addon_state.panel.model_dll != addon_state.active_model_dll))
            ImGui::TextWrapped("%s", addon_state.model_restart.c_str());
        ImGui::TextWrapped("Active model source: %s", disk_model ? "Model folder" : "NVIDIA DLSS5 DLL");
    } else if (addon_state.panel.mode == phoenix::nvidia) {
        ImGui::TextWrapped("NVIDIA DLSS5 DLL runs on the game's rendering GPU. Phoenix DLSS5 runner and "
                           "model-folder selections apply only to Phoenix DLSS5 output.");
    }
    if (visible.neural() && ImGui::Button("Reset neural controls")) {
        phoenix::apply_nr_reference(addon_state.panel);
        addon_state.panel.host_bridge = 0;
        addon_state.panel.host_unjitter = false;
        addon_state.panel.ui_correction = 0;
        addon_state.panel.controls.enabled = true;
        addon_state.panel.controls.advanced = true;
        addon_state.panel.controls.auto_mask = false;
        changed = true;
    }
    if (addon_state.direct_game && phoenix::uses_ours(addon_state.panel.mode) && !addon_state.portable_backbuffer) {
        std::string selected = addon_state.panel.execution_device;
        for (const auto &device_info : addon_state.execution_devices)
            if (device_info.id == selected) {
                selected = device_info.label;
                break;
            }
        if (addon_state.panel.execution_device.find(':') != std::string::npos)
            selected += " [" +
                        addon_state.panel.execution_device.substr(
                            addon_state.panel.execution_device.size() - 8) +
                        "]";
        const bool installed = addon_state.runtime_available;
        if (ImGui::BeginCombo("Phoenix DLSS5 processing device", selected.c_str())) {
            for (const auto &device_info : addon_state.execution_devices) {
                ImGui::PushID(device_info.id.c_str());
                const auto label = device_info.id.find(':') == std::string::npos
                                       ? device_info.label
                                       : device_info.label + " [" +
                                             device_info.id.substr(device_info.id.size() - 8) + "]";
                if (ImGui::Selectable(
                        label.c_str(), addon_state.panel.execution_device == device_info.id,
                        device_info.id == "native" || installed ? 0
                                                                : ImGuiSelectableFlags_Disabled)) {
                    addon_state.panel.execution_device = device_info.id;
                    changed = true;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", device_info.id.c_str());
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (!installed)
            ImGui::TextUnformatted("C++ runtime or selected model source is missing. Configure "
                                   "portable_cpp and the model source.");
        ImGui::TextWrapped("Select the card that runs Phoenix DLSS5. The rendering GPU and NVIDIA DLSS5 "
                           "DLL route follow the game.");
        if (addon_state.panel.execution_device == "cpu" ||
            addon_state.panel.execution_device == "sycl:cpu") {
            const int available = int(phoenix_threads::available());
            bool automatic = addon_state.panel.cpu_threads == 0;
            if (ImGui::Checkbox("Automatic CPU threads (all available)", &automatic)) {
                addon_state.panel.cpu_threads = automatic ? 0 : available;
                changed = true;
            }
            if (!automatic) {
                if (addon_state.panel.cpu_threads > available) {
                    addon_state.panel.cpu_threads = available;
                    changed = true;
                }
                changed =
                    ImGui::SliderInt("CPU threads", &addon_state.panel.cpu_threads, 1, available) ||
                    changed;
            }
            ImGui::Text("CPU workers: %u of %d available",
                        phoenix_threads::resolve(unsigned(addon_state.panel.cpu_threads)),
                        available);
            ImGui::TextUnformatted("Thread changes restart the CPU runner.");
        }
        if (addon_state.panel.execution_device != "native")
            ImGui::TextUnformatted(
                "C++ runner: host preparation and resolve, selected-device network.");
        if (addon_state.panel.execution_device != "native") {
            changed = ImGui::SliderInt("Maximum result age (frames)",
                                       &addon_state.panel.offload_age_frames, 1,
                                       phoenix::kMaxOffloadAgeFrames) ||
                      changed;
            changed = ImGui::SliderInt("Maximum result age (ms)", &addon_state.panel.offload_age_ms,
                                       1, phoenix::kMaxOffloadAgeMs) ||
                      changed;
            ImGui::TextWrapped(
                "Phoenix DLSS5 runs on the selected device. The NVIDIA DLSS5 DLL route remains on the "
                "rendering GPU. Reference execution may miss the age limit.");
        }
    }
    if (visible.comparison()) {
        ImGui::BeginDisabled(!addon_state.comparison_views);
        if (ImGui::Button("Defaults##comparison")) {
            phoenix::reset_comparison_section(addon_state.panel);
            changed = true;
        }
        if (addon_state.direct_game) {
            const bool nvidia_source = addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1;
            if (nvidia_source && addon_state.panel.comparison > 1) {
                addon_state.panel.comparison = 1;
                changed = true;
            }
            int view = std::max(0, addon_state.panel.comparison - 1);
            if (ImGui::Combo("View", &view, phoenix::comparison_names + 1, nvidia_source ? 1 : 5)) {
                addon_state.panel.comparison = view + 1;
                changed = true;
            }
        } else
            changed |=
                ImGui::Combo("View", &addon_state.panel.comparison, phoenix::comparison_names, 6);
        if (visible.pair_sources()) {
            changed |=
                ImGui::Combo("Source A", &addon_state.panel.source_a, phoenix::source_names, 3);
            changed |=
                ImGui::Combo("Source B", &addon_state.panel.source_b, phoenix::source_names, 3);
            if (ImGui::Button("Swap A / B")) {
                std::swap(addon_state.panel.source_a, addon_state.panel.source_b);
                changed = true;
            }
            if (addon_state.panel.source_a == addon_state.panel.source_b)
                ImGui::TextDisabled("Both sides use the same source");
            if (addon_state.direct_game && (addon_state.panel.source_a == 1 || addon_state.panel.source_b == 1)) {
                if (addon_state.panel.comparison > 1) {
                    addon_state.panel.comparison = 1;
                    changed = true;
                }
                ImGui::TextDisabled("NVIDIA DLSS5 supports split comparison; difference views use Input and Phoenix DLSS5");
            }
        }
        if (visible.divider())
            changed |= ImGui::SliderFloat("Split position", &addon_state.panel.divider, .1f, .9f,
                                          "%.2f", ImGuiSliderFlags_AlwaysClamp);
        if (visible.difference_gain())
            changed |= ImGui::SliderFloat(
                "Difference gain", &addon_state.panel.difference_gain, 1.f, 64.f, "%.1fx",
                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        changed |= ImGui::Checkbox("Labels", &addon_state.panel.comparison_labels);
        if (visible.divider())
            changed |= ImGui::Checkbox("Separator lines", &addon_state.panel.comparison_lines);
        ImGui::EndDisabled();
        if (!addon_state.comparison_views)
            ImGui::TextWrapped("Bridge comparison uses matching input and output; NVIDIA DLSS5 and "
                               "difference views need the direct D3D12 route.");
    }
    static int preset_slot = 0;
    const char *slots[]{"Preset 1", "Preset 2", "Preset 3", "Preset 4"};
    ImGui::Combo("Preset", &preset_slot, slots, 4);
    if (ImGui::Button("Load preset")) {
        changed = load_preset(addon_state, unsigned(preset_slot)) || changed;
    }
    ImGui::SameLine();
    if (ImGui::Button("Save preset")) {
        store_preset(addon_state, unsigned(preset_slot));
    }
    ImGui::Separator();
    if (visible.neural()) {
        if (ImGui::Button("Defaults##neural")) {
            phoenix::reset_neural_section(addon_state.panel);
            changed = true;
        }
        ImGui::BeginDisabled(!addon_state.input_capture);
        if (ImGui::Button("Capture input burst"))
            addon_state.capture(addon_state.panel.capture_burst > 0
                                    ? unsigned(addon_state.panel.capture_burst)
                                    : 8u);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Saves consecutive game inputs and their settings for offline replay.");
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Override strengths", &addon_state.live.enabled);
        ImGui::BeginDisabled(!visible.strengths());
        {
            changed |= ImGui::SliderFloat("Local tone", &addon_state.live.tone, 0.f, 2.f, "%.2f",
                                          ImGuiSliderFlags_AlwaysClamp);
            ImGui::SameLine();
            if (ImGui::SmallButton("R##s.live.tone")) {
                addon_state.live.tone = 1;
                changed = true;
            }
            changed |= ImGui::SliderFloat("Local structure", &addon_state.live.structure, 0.f, 2.f,
                                          "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::SameLine();
            if (ImGui::SmallButton("R##s.live.structure")) {
                addon_state.live.structure = 1;
                changed = true;
            }
        }
        ImGui::EndDisabled();
        changed |= ImGui::Checkbox("Override skin and guides", &addon_state.live.advanced);
        ImGui::BeginDisabled(!visible.guides());
        {
            changed |= ImGui::Checkbox("Automatic skin mask", &addon_state.live.auto_mask);
            ImGui::BeginDisabled(!addon_state.live.auto_mask);
            changed |= ImGui::SliderFloat("Skin structure", &addon_state.live.skin, -1.f, 2.f,
                                          "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::SameLine();
            if (ImGui::SmallButton("R##s.live.skin")) {
                addon_state.live.skin = -1;
                changed = true;
            }
            ImGui::EndDisabled();
            ImGui::TextWrapped("Skin structure uses the automatic skin mask; -1 follows local structure.");
            ImGui::SeparatorText("Engine motion and depth");
            ImGui::BeginDisabled(!addon_state.real_guides);
            changed |= ImGui::SliderFloat("Motion X scale", &addon_state.live.motion_x, -4.f, 4.f,
                                          "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::SameLine();
            if (ImGui::SmallButton("R##s.live.motion_x")) {
                addon_state.live.motion_x = 1;
                changed = true;
            }
            changed |= ImGui::SliderFloat("Motion Y scale", &addon_state.live.motion_y, -4.f, 4.f,
                                          "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::SameLine();
            if (ImGui::SmallButton("R##s.live.motion_y")) {
                addon_state.live.motion_y = 1;
                changed = true;
            }
            int depth = addon_state.live.depth + 1;
            const char *depths[]{"Game depth", "Normal depth", "Inverted depth"};
            if (ImGui::Combo("Depth", &depth, depths, 3)) {
                addon_state.live.depth = depth - 1;
                changed = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("R##live_depth")) {
                addon_state.live.depth = -1;
                changed = true;
            }
            ImGui::EndDisabled();
        }
        ImGui::EndDisabled();
        if (!addon_state.real_guides)
            ImGui::TextDisabled("Engine motion and depth are unavailable for final-color capture.");
        if(phoenix::uses_ours(addon_state.panel.mode)&&!addon_state.portable_backbuffer) {
            changed|=ImGui::Checkbox("Phoenix appearance (experimental)",&addon_state.live.appearance);
            ImGui::BeginDisabled(!addon_state.live.appearance);
            changed|=ImGui::SliderFloat("Phoenix intensity",&addon_state.live.intensity,0.f,2.f,"%.2f",ImGuiSliderFlags_AlwaysClamp);
            const char* styles[]={"Default","Natural","Cinematic"};
            changed|=ImGui::Combo("Phoenix style",&addon_state.live.style,styles,3);
            ImGui::EndDisabled();
            ImGui::TextDisabled("Surface selection requires an engine control map.");
        }
        if(addon_state.appearance.value("source",std::string("none"))!="none") {
            ImGui::Text("Developer controls: %s",addon_state.appearance.value("source",std::string()).c_str());
            ImGui::TextDisabled("Map %s | UI %s | distortion %s | revision %u",
                addon_state.appearance.value("control_map",false)?"connected":"none",
                addon_state.appearance.value("ui",false)?"connected":"none",
                addon_state.appearance.value("distortion",false)?"connected":"none",
                addon_state.appearance.value("generation",0u));
            ImGui::TextDisabled("Experimental. Per-object strengths come from the engine's map.");
        }
        if (visible.native() && !addon_state.portable_backbuffer) {
            if (ImGui::Button("Defaults##native")) {
                phoenix::reset_native_section(addon_state.panel);
                changed = true;
            }
            changed |= ImGui::Checkbox("NVIDIA DLSS5-only overrides", &addon_state.panel.native_options);
            if (visible.native_values()) {
                changed |= ImGui::SliderFloat("Intensity", &addon_state.panel.intensity, 0.f, 2.f,
                                              "%.2f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SameLine();
                if (ImGui::SmallButton("R##s.panel.intensity")) {
                    addon_state.panel.intensity = 1;
                    changed = true;
                }
                const char *styles[]{"Default", "Natural", "Cinematic"};
                changed |= ImGui::Combo("Style", &addon_state.panel.style, styles, 3);
                ImGui::SameLine();
                if (ImGui::SmallButton("R##nr_style")) {
                    addon_state.panel.style = 0;
                    changed = true;
                }
                bool correction = addon_state.panel.ui_correction != 0;
                if (ImGui::Checkbox("UI correction", &correction)) {
                    addon_state.panel.ui_correction = correction ? 1 : 0;
                    changed = true;
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("R##nr_uicorr")) {
                    addon_state.panel.ui_correction = 0;
                    changed = true;
                }
                changed |= ImGui::SliderFloat("Host exposure", &addon_state.panel.host_exposure,
                                              0.05f, 4.f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SameLine();
                if (ImGui::SmallButton("R##s.panel.host_exposure")) {
                    addon_state.panel.host_exposure = 1;
                    changed = true;
                }
                changed |= ImGui::SliderFloat("Host exposure scale",
                                              &addon_state.panel.host_exposure_scale, 0.05f, 4.f,
                                              "%.3f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SameLine();
                if (ImGui::SmallButton("R##s.panel.host_exposure_scale")) {
                    addon_state.panel.host_exposure_scale = 1;
                    changed = true;
                }
                changed |=
                    ImGui::SliderFloat("Host global tone", &addon_state.panel.host_globaltone, 0.f,
                                       2.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::SameLine();
                if (ImGui::SmallButton("R##s.panel.host_globaltone")) {
                    addon_state.panel.host_globaltone = 1;
                    changed = true;
                }
                bool host_jitter = addon_state.panel.host_jitter != 0;
                if (ImGui::Checkbox("Host forward game jitter", &host_jitter)) {
                    addon_state.panel.host_jitter = host_jitter ? 1 : 0;
                    changed = true;
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("R##nr_hjitter")) {
                    addon_state.panel.host_jitter = 1;
                    changed = true;
                }
                int host_hdr = addon_state.panel.host_hdr;
                const char *hdr_modes[]{"SDR declare", "HDR declare"};
                if (ImGui::Combo("Host HDR declare", &host_hdr, hdr_modes, 2)) {
                    addon_state.panel.host_hdr = host_hdr;
                    changed = true;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Experimental DLL declaration. Both proxy paths encode sRGB, "
                                      "so SDR is the supported setting.");
                changed |= ImGui::Checkbox("NR auto exposure", &addon_state.panel.nr_auto_exposure);
                ImGui::TextUnformatted("Native NR: feature 18 (DLSSNR runtime)");
                if (addon_state.panel.nr_feature != 18) {
                    ImGui::TextWrapped(
                        "Saved RR experiment is unavailable: it needs pre-denoise guide inputs.");
                    if (ImGui::Button("Use native NR")) {
                        addon_state.panel.nr_feature = 18;
                        changed = true;
                    }
                }
            }
        }
        if (ImGui::Button(addon_state.processing_enabled ? "Reset neural overrides" : "Restore game controls")) {
            addon_state.live = {};
            addon_state.panel.native_options = false;
            if (!addon_state.processing_enabled) {
                addon_state.panel.uplift = false;
                addon_state.processing_changed = true;
            }
            changed = true;
        }
        if (ImGui::Button("Reset history"))
            addon_state.action(Action::ResetHistory);
        if (ImGui::CollapsingHeader("History policy")) {
            ImGui::BeginDisabled(!addon_state.temporal_history);
            if (ImGui::Button("Defaults##history")) {
                phoenix::reset_history_section(addon_state.panel);
                changed = true;
            }
            changed |= ImGui::Checkbox("Force reset every frame", &addon_state.panel.force_reset);
            if (visible.history_cadence()) {
                changed |= ImGui::SliderInt("Reset frames after switch",
                                            &addon_state.panel.reset_after_switch, 1, 120);
                changed |= ImGui::SliderInt("Reset every N frames (0 = off)",
                                            &addon_state.panel.reset_every, 0, 600);
            }
            ImGui::EndDisabled();
            if (!addon_state.temporal_history)
                ImGui::TextWrapped(
                    "This bridge resets every input; persistent history needs engine guides.");
        }
        ImGui::Text("History resets: %llu%s", addon_state.live_resets_recorded,
                    addon_state.live.reset_requested ? " (pending)" : "");
    }
    if (visible.experiments() && ImGui::CollapsingHeader("Phoenix DLSS5 experiments")) {
        if (addon_state.direct_game) {
            changed |=
                ImGui::SliderInt("Network passes per frame", &addon_state.panel.refinement_passes,
                                 1, 4, "%d", ImGuiSliderFlags_AlwaysClamp);
            ImGui::TextWrapped(
                "Each pass refines the preceding output; each has separate previous-frame history. "
                "More passes cost GPU time and memory and may overprocess the image.");
            if (ImGui::TreeNode("Transformer block repetitions")) {
                ImGui::BeginDisabled(addon_state.fixed_native_blocks);
                for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index) {
                    char label[32];
                    snprintf(label, sizeof(label), "Block %u repetitions", iteration_index + 1);
                    changed |=
                        ImGui::SliderInt(label, &addon_state.panel.block_repeats[iteration_index],
                                         0, 4, "%d", ImGuiSliderFlags_AlwaysClamp);
                }
                ImGui::TextWrapped(
                    "0 skips a block and passes its input forward; 1 is the trained default. "
                    "Higher counts repeat the block on its preceding result. Truncated blocks are "
                    "not executed. Skipping or repetition changes the image and is experimental.");
                ImGui::EndDisabled();
                ImGui::TreePop();
            }
            if (ImGui::Button("Reset repetitions to defaults")) {
                addon_state.panel.refinement_passes = 1;
                addon_state.panel.block_repeats.fill(1);
                changed = true;
            }
        }
        ImGui::BeginDisabled(addon_state.fixed_native_blocks);
        changed |= ImGui::Checkbox("Enable approximate network", &addon_state.panel.experimental);
        ImGui::EndDisabled();
        if(addon_state.fixed_native_blocks && (addon_state.panel.experimental ||
            std::any_of(addon_state.panel.block_repeats.begin(),addon_state.panel.block_repeats.end(),[](int n){return n!=1;}))) {
            if(ImGui::Button("Use full network for native x86 driver")) {
                addon_state.panel.experimental=false;addon_state.panel.block_repeats.fill(1);changed=true;
            }
        }
        if (visible.depth()) {
            ImGui::BeginDisabled(addon_state.fixed_native_blocks);
            changed |= ImGui::SliderInt("Transformer blocks", &addon_state.panel.transformer_blocks,
                                        1, 8, "%d / 8", ImGuiSliderFlags_AlwaysClamp);
            ImGui::EndDisabled();
            addon_state.panel.minimum_blocks =
                std::min(addon_state.panel.minimum_blocks, addon_state.panel.transformer_blocks);
            ImGui::BeginDisabled(addon_state.portable_backbuffer);
            changed |= ImGui::Checkbox("Adaptive blocks", &addon_state.panel.automatic_budget);
            ImGui::EndDisabled();
            if (addon_state.portable_backbuffer && addon_state.panel.automatic_budget &&
                ImGui::Button("Use fixed blocks for portable processing")) {
                addon_state.panel.automatic_budget = false;
                changed = true;
            }
            if (visible.budget() && !addon_state.portable_backbuffer) {
                const char *target_names[]{"Phoenix DLSS5 GPU time", "Desired game FPS"};
                if (ImGui::BeginCombo("Target", target_names[addon_state.panel.budget_mode])) {
                    for (int target = 0; target < 2; ++target) {
                        ImGui::BeginDisabled(target == 1 && !addon_state.fps_budget);
                        if (ImGui::Selectable(target_names[target],
                                              addon_state.panel.budget_mode == target)) {
                            addon_state.panel.budget_mode = target;
                            changed = true;
                        }
                        ImGui::EndDisabled();
                    }
                    ImGui::EndCombo();
                }
                if (!addon_state.fps_budget)
                    ImGui::TextWrapped(
                        "GPU-time adaptation is available. The helper cannot drive a whole-game "
                        "FPS target from its hidden window's presentation rate.");
                if (addon_state.panel.budget_mode == 0) {
                    changed |= ImGui::SliderFloat(
                        "Phoenix DLSS5 GPU budget (ms)", &addon_state.panel.budget_ms, .1f, 100.f,
                        "%.2f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                    ImGui::TextWrapped("Budget estimate: %.3f ms | target is not guaranteed",
                                       addon_state.depth_budget.ema());
                } else {
                    changed |=
                        ImGui::SliderFloat("Desired FPS", &addon_state.panel.desired_fps, 10.f,
                                           360.f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::Text("Controller presentation rate: %.1f FPS",
                                addon_state.depth_budget.ema() > 0
                                    ? 1000.0 / addon_state.depth_budget.ema()
                                    : 0);
                    ImGui::TextWrapped(
                        "Adjusts one block at a time from the game's presentation rate. Pauses "
                        "while unfocused or inference is inactive. A frame cap or other game work "
                        "can prevent reaching the target.");
                }
                changed |= ImGui::SliderInt("Minimum blocks", &addon_state.panel.minimum_blocks, 1,
                                            addon_state.panel.transformer_blocks, "%d",
                                            ImGuiSliderFlags_AlwaysClamp);
            }
            unsigned repeat_launches = 114;
            for (unsigned iteration_index = 0;
                 iteration_index < addon_state.last_transformer_blocks; ++iteration_index)
                repeat_launches +=
                    5 * unsigned(addon_state.direct_game
                                     ? addon_state.panel.block_repeats[iteration_index]
                                     : 1);
            ImGui::Text(
                "Depth: %u / 8 | launches per frame: %u", addon_state.last_transformer_blocks,
                repeat_launches *
                    unsigned(addon_state.direct_game ? addon_state.panel.refinement_passes : 1));
            ImGui::TextWrapped("Below 8: untrained approximation, quality may deteriorate.");
            if (ImGui::Button("Restore full network")) {
                addon_state.panel.experimental = false;
                addon_state.panel.transformer_blocks = 8;
                addon_state.panel.refinement_passes = 1;
                addon_state.panel.block_repeats.fill(1);
                changed = true;
            }
            ImGui::TextUnformatted("Production scheduling: chained groups, full barriers.");
        }
    }
    if (ImGui::CollapsingHeader("Hotkeys")) {
        const char *actions[]{"Cycle backend",    "Reset history",
                              "Log marker",       "Cycle experimental depth",
                              "Clear GPU timing", "Capture inputs"};
        for (unsigned iteration_index = 0; iteration_index < addon_state.hotkeys.size();
             ++iteration_index) {
            if (!visible.hotkey(iteration_index)) {
                if (addon_state.capture_hotkey == int(iteration_index))
                    addon_state.capture_hotkey = -1;
                continue;
            }
            ImGui::PushID(int(iteration_index));
            char label[80]{};
            const int key = addon_state.hotkeys[iteration_index];
            if (addon_state.capture_hotkey == int(iteration_index))
                snprintf(label, sizeof(label), "Press a key...");
            else if (key >= VK_F1 && key <= VK_F24)
                snprintf(label, sizeof(label), "F%d", key - VK_F1 + 1);
            else if (key == 0)
                snprintf(label, sizeof(label), "Unbound");
            else
                snprintf(label, sizeof(label), "Key %d", key);
            if (ImGui::Button(label, ImVec2(112, 0))) {
                addon_state.capture_hotkey = int(iteration_index);
                addon_state.hotkey_deadline = GetTickCount64() + 10000;
            }
            ImGui::SameLine();
            ImGui::TextUnformatted(actions[iteration_index]);
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear")) {
                clear_binding(addon_state, iteration_index);
            }
            ImGui::PopID();
        }
    }
    if (ImGui::CollapsingHeader("Settings / logging")) {
        const char *log_levels[]{"Quiet", "Normal", "Verbose"};
        changed |= ImGui::Combo("Log level", &addon_state.panel.log_level, log_levels, 3);
        ImGui::BeginDisabled(addon_state.portable_backbuffer);
        changed |=
            ImGui::SliderInt("Capture burst frames", &addon_state.panel.capture_burst, 0, 64);
        changed |= ImGui::Checkbox("Live settings reload", &addon_state.panel.external_reload);
        if (ImGui::Button("Reload overrides now"))
            addon_state.reload();
        ImGui::EndDisabled();
        ImGui::TextWrapped("%s", addon_state.override_status.c_str());
        ImGui::TextWrapped("%s",
                           phoenix_model::source_path_string(addon_state.overrides_path).c_str());
        if (ImGui::Button("Log marker")) {
            ++addon_state.markers;
            addon_state.marker();
        }
        ImGui::Text("Markers: %llu", addon_state.markers);
        if (addon_state.telemetry_available)
            ImGui::Text("Switches: %llu", addon_state.switches);
        else
            ImGui::TextUnformatted("Per-route frame counts are unavailable for this transport.");
        for (unsigned iteration_index = 0; addon_state.telemetry_available && iteration_index < 4; ++iteration_index)
            ImGui::TextWrapped(
                "%s: %llu frames | own %llu | native %llu", phoenix::names[iteration_index],
                addon_state.mode_frames[iteration_index], addon_state.mode_own[iteration_index],
                addon_state.mode_native[iteration_index]);
    }
    if (visible.own_details() && ImGui::CollapsingHeader("Phoenix DLSS5 model")) {
        ImGui::TextWrapped("%s",
                           phoenix_model::source_path_string(addon_state.bundle_path).c_str());
        ImGui::Text("Model graph: 41 modules / %u transformer blocks",
                    addon_state.last_transformer_blocks);
    }
    addon_state.live_selection_changed = phoenix::copy_live_selection(before, addon_state.panel);
    addon_state.live_tuning_changed = phoenix::copy_live_tuning(before, addon_state.panel);
    return changed;
}
inline void draw_timing(Context &addon_state) {
    if (addon_state.portable_wall_ms) {
        ImGui::Text("Game presentation: %.1f FPS", addon_state.present_fps);
        ImGui::Text("Last processed frame: %.3f ms (host + inference + resolve)", *addon_state.portable_wall_ms);
        ImGui::TextWrapped("Timing source: %s", addon_state.timing_provenance.c_str());
        if (ImGui::Button("Clear processing timing")) addon_state.action(Action::ClearTiming);
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
        if (ImGui::Button("Clear GPU timing")) {
            addon_state.action(Action::ClearTiming);
        }
        if (!addon_state.worker_timing.report().at("rows").empty()) {
            ImGui::TextUnformatted("Other processing devices (wall time, ms)");
            ImGui::TextWrapped("Rows retain completed results across device switches. Execute "
                               "includes transfers; result age includes queueing. These are "
                               "separate from the render-GPU timings below.");
            if (ImGui::BeginTable("phoenix_workers", 8,
                                  ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg)) {
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
                              ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg)) {
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
inline bool draw_compact(Context &context) {
    context.live_tuning_changed = false;
    ImGui::TextWrapped("Status: %s", context.status.empty() ? "Waiting" : context.status.c_str());
    auto& enabled = context.processing_enabled ? *context.processing_enabled : context.panel.uplift;
    context.processing_changed = ImGui::Checkbox("Enable neural rendering", &enabled);
    const char *labels[]{"Input", "NVIDIA DLSS5 DLL",
                          "Comparison", "Phoenix DLSS5"};
    context.live_selection_changed = ImGui::Combo("Backend##phoenix_compact", &context.panel.mode, labels, 4);
    if (context.live_selection_changed && context.direct_game)
        phoenix::normalize_direct_comparison(context.panel);
    const bool changed = context.processing_changed || context.live_selection_changed;
    if (phoenix::PanelVisibility{context.panel}.own_details())
        ImGui::Text("Transformer depth: %u / 8", context.last_transformer_blocks);
    if (context.live_timing) {
        const auto report = context.live_timing->report();
        for (const auto &row : report.at("rows"))
            if (row.at("mode") == context.last_mode &&
                row.at("transformer_blocks") == context.last_transformer_blocks &&
                row.at("settings") == report.at("active_settings"))
                ImGui::Text("%s route: %.3f ms", context.timing_provenance.c_str(),
                            row.at("route").at("mean_ms").get<double>());
    }
    if (!context.error.empty())
        ImGui::TextWrapped("%s", context.error.c_str());
    return changed;
}
} // namespace phoenix_ui
