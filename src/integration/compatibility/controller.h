#pragma once
#include "../../native/support/model_tools.h"
#include "../reshade/shared_panel.h"
#include "d3d11_bridge.h"
#include "readback_bridge.h"
#include "session_policy.h"
#include "screenshot.h"
#include "../../native/support/control_mailbox.h"
#include <map>

namespace phoenix_compat {
struct Session {
    phoenix_addon::phoenix::Settings settings;
    phoenix_addon::phoenix::Settings applied_settings;
    std::array<phoenix_addon::phoenix::Settings, 4> presets;
    std::array<int, 6> hotkeys{{VK_F9, VK_F10, VK_F11, VK_F8, VK_F7, 0}};
    int capture_hotkey = -1;
    uint64_t hotkey_deadline = 0, markers = 0, fps_tick = 0, fps_frames = 0;
    double present_fps = 0;
    bool settings_dirty = false;
    std::vector<phoenix_addon::phoenix::ExecutionDevice> execution_devices{
        {"native", "Native RTX (helper)"}};
    std::string ui_error;
    Json previous_gpu_rows = Json::array(), previous_worker_rows = Json::array();
    std::unique_ptr<D3D11Bridge> bridge;
    std::unique_ptr<ReadbackBridge> readback;
    phoenix_model_tools::Task picker, validator;
    fs::path output_directory;
    phoenix_addon::phoenix::control_bridge::Mailbox control;
    uint64_t switches = 0;
    bool timing_clear_pending = false;
    uint64_t timing_clear_revision = 0, timing_clear_generation = 0, helper_generation = 0;
    fs::path screenshot_pending, screenshot_complete;
    std::unique_ptr<Snapshot> screenshot;
    std::string screenshot_error;
    bool deferred_destroy = false;
    std::array<uint64_t, 4> mode_frames{}, mode_own{}, mode_native{};
    int recorded_mode = -1;
    phoenix_model::WeightSource configured;
    bool enabled = false, restart = false, shared_unavailable = false;
    // Hold complete input/output pairs: never put an old mask over newer motion.
    bool coherent_frames = true;
    unsigned width = 0, height = 0;
    uint64_t frames = 0;
    uint64_t logged_outputs = 0;
    reshade::api::format readback_format = reshade::api::format::unknown;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::string message = "Choose a model and start the graphics bridge.";
};
inline HMODULE module = nullptr;
inline const char *api_name(reshade::api::device_api api) {
    using reshade::api::device_api;
    switch (api) {
    case device_api::d3d9:
        return "Direct3D 9";
    case device_api::d3d10:
        return "Direct3D 10 / 10.1";
    case device_api::d3d11:
        return "Direct3D 11";
    case device_api::d3d12:
        return "Direct3D 12";
    case device_api::opengl:
        return "OpenGL";
    case device_api::vulkan:
        return "Vulkan";
    default:
        return "Unknown graphics API";
    }
}
inline void log_progress(Session &value) {
    auto *host = value.bridge ? value.bridge->host() : value.readback->host();
    if (host->phase() == Phase::Failed) {
        const auto completed =
            value.bridge ? value.bridge->delivered() : value.readback->delivered();
        if (!value.coherent_frames || completed == 0)
            require(false, host->error().c_str());
        value.message = "Helper stopped; holding the last completed frame. " + host->error();
    }
    const auto outputs = value.bridge ? value.bridge->delivered() : value.readback->delivered();
    const auto presented = value.bridge ? value.bridge->presented() : value.readback->presented();
    if (outputs > value.logged_outputs &&
        (outputs <= 3 || outputs / 100 > value.logged_outputs / 100)) {
        const auto message = "DLSS5 Phoenix bridge: completed=" + std::to_string(outputs) +
                             ", presented=" + std::to_string(presented) + ", " + value.message;
        reshade::log::message(reshade::log::level::info, message.c_str());
    }
    if (outputs > value.logged_outputs) {
        auto& counts = value.applied_settings.mode == phoenix_addon::phoenix::nvidia ? value.mode_native : value.mode_own;
        counts[size_t(value.applied_settings.mode)] += outputs - value.logged_outputs;
    }
    value.logged_outputs = outputs;
}
inline std::recursive_mutex sessions_mutex;
// A D3D12 sharing device is tied to the game's D3D11 device, not its canvas.
// Releasing the last device reference during resize can wait for driver/helper
// teardown. Keep this lightweight owner across runtime resets; frame resources
// remain per-bridge and are released before DXGI replaces its backbuffers.
// Normal device destruction or addon detachment clears this owner. At process
// exit, let the OS reclaim it rather than call graphics drivers under DLL lock.
inline auto &sharing_devices = *new std::map<reshade::api::device*, Ptr<ID3D12Device>>;
// Normal unloading calls detach before the loader lock. At process exit the
// OS closes the helper job; static destruction must not join terminated threads.
inline auto &sessions = *new std::map<reshade::api::effect_runtime *, std::unique_ptr<Session>>;
inline bool applies(reshade::api::effect_runtime *runtime) {
    // The standalone thin frontend can also be built x64 for transport tests.
#ifdef PHOENIX_COMPATIBILITY_ONLY
    return runtime != nullptr;
#else
    return runtime && uses_helper(runtime->get_device()->get_api(), sizeof(void *) * 8);
#endif
}
inline Session &session(reshade::api::effect_runtime *runtime) {
    auto &entry = sessions[runtime];
    if (!entry) {
        entry = std::make_unique<Session>();
        phoenix_addon::phoenix::visit_settings(entry->settings, [&](const char *key, auto &value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
                char text[4096]{};
                size_t length = sizeof(text);
                if (reshade::get_config_value(runtime, "Phoenix", key, text, &length))
                    value.assign(text, strnlen_s(text, sizeof(text)));
            } else
                reshade::get_config_value(runtime, "Phoenix", key, value);
        });
        if (!entry->settings.valid())
            entry->settings = {};
        // Frame-count limits from a synchronous DX12 profile are too small
        // for an asynchronous helper at high game FPS. Time still caps age.
        entry->settings.offload_age_frames = 60;
        entry->settings.offload_age_ms = 100;
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeFrames",
                                  entry->settings.offload_age_frames);
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeMs",
                                  entry->settings.offload_age_ms);
        entry->settings.offload_age_frames =
            std::clamp(entry->settings.offload_age_frames, 1, 6000);
        entry->settings.offload_age_ms = std::clamp(entry->settings.offload_age_ms, 1, 120000);
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "Enabled", entry->enabled);
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                                  entry->coherent_frames);
        if (normalize_capabilities(entry->settings))
            entry->ui_error = "FPS budgeting is unavailable in a companion; using GPU-time budget.";
        entry->applied_settings = entry->settings;
        for (size_t index = 0; index < entry->presets.size(); ++index) {
            auto &preset = entry->presets[index];
            preset.mode = int(index);
            const auto section = "Phoenix.Preset" + std::to_string(index + 1);
            phoenix_addon::phoenix::visit_settings(preset, [&](const char *key, auto &field) {
                if constexpr (std::is_same_v<std::decay_t<decltype(field)>, std::string>) {
                    char text[4096]{};
                    size_t bytes = sizeof(text);
                    if (reshade::get_config_value(runtime, section.c_str(), key, text, &bytes))
                        field.assign(text, strnlen_s(text, sizeof(text)));
                } else
                    reshade::get_config_value(runtime, section.c_str(), key, field);
            });
            if (!preset.valid())
                preset = {};
            normalize_capabilities(preset);
        }
        for (size_t index = 0; index < entry->hotkeys.size(); ++index) {
            auto key = "Hotkey" + std::to_string(index);
            int binding = entry->hotkeys[index];
            reshade::get_config_value(runtime, "Phoenix", key.c_str(), binding);
            if (binding >= 0 && binding <= 255)
                entry->hotkeys[index] = binding;
        }
        phoenix_ui::sanitize_bindings(entry->hotkeys);
        wchar_t path[32768]{};
        GetModuleFileNameW(module, path, 32768);
        entry->output_directory = fs::path(path).parent_path() / "PhoenixOutput";
        try {
            std::ifstream file(fs::path(path).parent_path() / "Phoenix.json");
            if (file) {
                Json config;
                file >> config;
                if (config.contains("sandbox")) {
                    const auto root = fs::u8path(config.at("sandbox").get<std::string>());
                    if (root.is_absolute()) entry->output_directory = root / "PhoenixOutput";
                }
                if (config.contains("weight_source"))
                    entry->configured =
                        phoenix_model::WeightSource::config(config["weight_source"]);
            }
        } catch (const std::exception &error) {
            entry->message = error.what();
        }
    }
    if (entry->control.folder().empty()) {
        const auto window = uint64_t(reinterpret_cast<uintptr_t>(runtime->get_hwnd()));
        auto folder = entry->output_directory / "bridge";
        if (sessions.size() > 1) folder += "-" + std::to_string(window);
        entry->control.configure(folder, window);
    }
    return *entry;
}
inline void save(reshade::api::effect_runtime *runtime, Session &value, bool apply = true) {
    if (normalize_capabilities(value.settings))
        value.ui_error = "FPS budgeting is unavailable in a companion; using GPU-time budget.";
    for (auto& preset : value.presets) normalize_capabilities(preset);
    phoenix_addon::phoenix::visit_settings(value.settings, [&](const char *key, auto &field) {
        if constexpr (std::is_same_v<std::decay_t<decltype(field)>, std::string>)
            reshade::set_config_value(runtime, "Phoenix", key, field.c_str());
        else
            reshade::set_config_value(runtime, "Phoenix", key, field);
    });
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "Enabled", value.enabled);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                              value.coherent_frames);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeFrames",
                              value.settings.offload_age_frames);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeMs",
                              value.settings.offload_age_ms);
    for (size_t index = 0; index < value.presets.size(); ++index) {
        const auto section = "Phoenix.Preset" + std::to_string(index + 1);
        phoenix_addon::phoenix::visit_settings(
            value.presets[index], [&](const char *key, auto &field) {
                if constexpr (std::is_same_v<std::decay_t<decltype(field)>, std::string>)
                    reshade::set_config_value(runtime, section.c_str(), key, field.c_str());
                else
                    reshade::set_config_value(runtime, section.c_str(), key, field);
            });
    }
    for (size_t index = 0; index < value.hotkeys.size(); ++index) {
        const auto key = "Hotkey" + std::to_string(index);
        reshade::set_config_value(runtime, "Phoenix", key.c_str(), value.hotkeys[index]);
    }
    if (apply) {
        value.restart = value.restart || helper_settings_changed(value.applied_settings, value.settings);
        value.applied_settings = value.settings;
        value.settings_dirty = false;
    }
}
inline HostClient *active_host(Session &value) {
    return value.bridge ? value.bridge->host() : value.readback ? value.readback->host() : nullptr;
}
inline void remember_timing(Session &value) {
    auto *host = active_host(value);
    if (!host || value.timing_clear_pending)
        return;
    const auto info = host->ui_snapshot();
    for (auto item : {std::pair{"gpu_timing", &value.previous_gpu_rows},
                      std::pair{"worker_timing", &value.previous_worker_rows}}) {
        const auto report = info.value(item.first, Json::object());
        if (!report.is_object())
            continue;
        for (auto row : report.value("rows", Json::array())) {
            if (std::string(item.first) == "gpu_timing")
                row["settings"]["HelperProcess"] = host->process_id();
            item.second->push_back(std::move(row));
        }
        // Bound UI-only history independently of inference allocations.
        while (item.second->size() > 64)
            item.second->erase(item.second->begin());
    }
}
inline bool send_action(Session &value, phoenix_ui::Action action, unsigned argument = 0) {
    auto *host = active_host(value);
    if (action == phoenix_ui::Action::ClearTiming) {
        value.previous_gpu_rows = Json::array();
        value.previous_worker_rows = Json::array();
        value.timing_clear_pending = host != nullptr;
        value.timing_clear_generation = value.helper_generation;
        value.timing_clear_revision = host ? host->ui_snapshot().value("ui_clear_revision", uint64_t(0)) : 0;
        if (!host) { value.ui_error.clear(); return true; }
    }
    if (!host || !host->ui_action(action, argument)) {
        if (action == phoenix_ui::Action::ClearTiming) value.timing_clear_pending = false;
        value.ui_error = "Start the graphics bridge before using this action.";
        return false;
    }
    value.ui_error.clear();
    return true;
}
inline phoenix_ui::Context panel_context(reshade::api::effect_runtime *runtime, Session &value) {
    auto *host = active_host(value);
    auto info = host ? host->ui_snapshot() : Json::object();
    if (value.timing_clear_pending && value.timing_clear_generation != value.helper_generation)
        value.timing_clear_pending = false;
    if (value.timing_clear_pending) {
        const bool clear = info.value("ui_clear_revision", uint64_t(0)) > value.timing_clear_revision;
        if (clear) value.timing_clear_pending = false;
        else { info.erase("gpu_timing"); info.erase("worker_timing"); info.erase("gpu_history"); }
    }
    if (info.contains("ui_action_error"))
        value.ui_error = info["ui_action_error"].get<std::string>();
    if (info.contains("execution_devices")) {
        std::vector<phoenix_addon::phoenix::ExecutionDevice> devices;
        for (const auto &device : info["execution_devices"]) {
            const auto id = device.value("id", std::string());
            if (phoenix_addon::phoenix::valid_execution_device(id))
                devices.push_back({id, device.value("label", id)});
        }
        if (!devices.empty())
            value.execution_devices = std::move(devices);
    }
    phoenix_ui::Context ui{value.settings, value.settings.controls, value.presets,
                           value.hotkeys,  value.capture_hotkey,    value.hotkey_deadline,
                           value.picker,   value.execution_devices, value.ui_error,
                           value.markers};
    ui.direct_nr_ready = info.value("nvidia", Json::object()).value("ready", false);
    ui.timing_provenance = "Helper GPU (D3D12 companion)";
    ui.inference_process = true;
    ui.status = value.message;
    ui.switches = value.switches;
    ui.mode_frames = value.mode_frames;
    ui.mode_own = value.mode_own;
    ui.mode_native = value.mode_native;
    ui.real_guides = ui.temporal_history = ui.comparison_views = false;
    ui.fps_budget = false;
    ui.model_restart = "Save configuration / apply to restart the helper and load this model.";
    ui.runtime_available = !info.empty();
    ui.configured_weight_source = value.configured;
    ui.model_path = value.settings.model_source == "model_folder"
                        ? fs::u8path(value.settings.model_folder) / "model.json"
                    : value.settings.model_source == "nvidia_dll"
                        ? fs::u8path(value.settings.model_dll)
                    : value.configured.disk_model() ? value.configured.folder / "model.json"
                                                    : value.configured.dll;
    ui.disk_model = value.settings.model_source == "model_folder" ||
                    (value.settings.model_source == "configured" && value.configured.disk_model());
    ui.runtime_root = fs::u8path(info.value("runtime_root", std::string()));
    ui.bundle_path = fs::u8path(info.value("bundle_path", std::string()));
    ui.overrides_path = fs::u8path(info.value("overrides_path", std::string()));
    ui.override_status = info.value("override_status", "Helper settings available after startup.");
    ui.active_model_source = value.applied_settings.model_source;
    ui.active_model_folder = value.applied_settings.model_folder;
    ui.active_model_dll = value.applied_settings.model_dll;
    ui.last_mode = info.value("last_mode", -1);
    ui.last_transformer_blocks = info.value("transformer_blocks", 8u);
    ui.live_resets_recorded = info.value("history_resets", uint64_t(0));
    ui.depth_budget.estimate = info.value("budget_estimate_ms", 0.0);
    ui.present_fps = value.present_fps;
    if (info.contains("resolution")) {
        const auto &geometry = info["resolution"];
        ui.direct_runner = phoenix_ui::Preview{
            {geometry.value("output_width", 0u), geometry.value("output_height", 0u),
             geometry.value("width", 0u), geometry.value("height", 0u),
             geometry.value("image_width", 0u), geometry.value("image_height", 0u)},
            info.value("upscaler_status", std::string())};
    }
    ui.live_timing.emplace();
    const auto gpu = info.value("gpu_timing", Json::object());
    if (gpu.is_object())
        ui.live_timing->data = gpu;
    if (!ui.live_timing->data.contains("active_settings"))
        ui.live_timing->data["active_settings"] = Json::object();
    if (!ui.live_timing->data.contains("rows"))
        ui.live_timing->data["rows"] = Json::array();
    for (const auto &row : value.previous_gpu_rows)
        ui.live_timing->data["rows"].push_back(row);
    if (info.contains("gpu_history"))
        ui.live_timing->samples = info["gpu_history"].get<std::vector<float>>();
    const auto worker = info.value("worker_timing", Json::object());
    if (worker.is_object())
        ui.worker_timing.data = worker;
    if (!ui.worker_timing.data.contains("rows"))
        ui.worker_timing.data["rows"] = Json::array();
    for (const auto &row : value.previous_worker_rows)
        ui.worker_timing.data["rows"].push_back(row);
    ui.save = [runtime, &value] { save(runtime, value); };
    ui.persist = [runtime, &value] { save(runtime, value, false); };
    ui.capture = [&value](unsigned frames) {
        send_action(value, phoenix_ui::Action::CaptureInputs, frames);
    };
    ui.reload = [&value] { send_action(value, phoenix_ui::Action::ReloadOverrides); };
    ui.marker = [&value] { send_action(value, phoenix_ui::Action::Marker); };
    ui.action = [&value](phoenix_ui::Action action) { send_action(value, action); };
    ui.model_validation.poll = [&value] { return value.validator.poll(); };
    ui.model_validation.validate = [&value](const phoenix_model::WeightSource &candidate, const fs::path &) {
        value.validator.validate(candidate, {}, module);
    };
    return ui;
}
inline Json control_snapshot(reshade::api::effect_runtime* runtime, Session& value) {
    auto* host = active_host(value);
    const auto validation = value.validator.poll();
    return {{"tick_ms", GetTickCount64()}, {"api", api_name(runtime->get_device()->get_api())},
        {"window_id", uint64_t(reinterpret_cast<uintptr_t>(runtime->get_hwnd()))},
        {"enabled", value.enabled}, {"coherent_frames", value.coherent_frames},
        {"settings", settings_json(value.settings)}, {"applied_settings", settings_json(value.applied_settings)},
        {"restart_pending", value.restart}, {"status", value.message},
        {"helper_pid", host ? host->process_id() : 0},
        {"phase", host ? unsigned(host->phase()) : unsigned(Phase::Stopped)},
        {"delivered", value.bridge ? value.bridge->delivered() : value.readback ? value.readback->delivered() : 0},
        {"presented", value.bridge ? value.bridge->presented() : value.readback ? value.readback->presented() : 0},
        {"helper_roundtrip_ms", host ? host->roundtrip() : 0},
        {"timing_provenance", "Helper GPU; game capture and display excluded"},
        {"model_validation", {{"busy", validation.busy}, {"passed", validation.passed},
                              {"key", validation.key}, {"message", validation.message}}},
        {"helper", host ? host->ui_snapshot() : Json::object()},
        {"present_fps", value.present_fps},
        {"screenshot", {{"pending", !value.screenshot_pending.empty()},
                        {"path", phoenix_model::source_path_string(value.screenshot_complete)},
                        {"error", value.screenshot_error}}}};
}
inline void poll_control(reshade::api::effect_runtime* runtime, Session& value) {
    try {
        value.control.poll(value.settings, [&] { return control_snapshot(runtime, value); },
            [&](const phoenix_addon::phoenix::control_bridge::Command& command, Json& response) -> std::string {
                if (command.op == "set") {
                    value.settings = command.settings;
                    if (command.enabled) {
                        value.restart = value.restart || value.enabled != *command.enabled;
                        value.enabled = *command.enabled;
                    }
                    if (command.coherent_frames) value.coherent_frames = *command.coherent_frames;
                    save(runtime, value);
                    return value.restart ? "restart_queued" : "applied";
                }
                if (command.op == "save") { save(runtime, value); return "saved"; }
                if (command.op == "validate_model") {
                    require(!value.validator.poll().busy, "Model validation is already running");
                    const auto source = phoenix_model_tools::selection(value.settings.model_source,
                        value.settings.model_folder, value.settings.model_dll, value.configured);
                    value.validator.validate(source, {}, module);
                    return "queued";
                }
                if (command.op == "screenshot") {
                    require(value.screenshot_pending.empty(), "A game screenshot is already queued");
                    fs::create_directories(value.output_directory);
                    const auto postfix = "Phoenix-bridge-" + command.id;
                    value.screenshot_pending = value.output_directory / (postfix + ".png");
                    value.screenshot_error.clear();
                    response["screenshot_postfix"] = postfix;
                    response["screenshot_path"] = phoenix_model::source_path_string(value.screenshot_pending);
                    return "queued";
                }
                if (command.op == "reset_history" || command.op == "capture") {
                    const auto action = command.op == "capture" ? phoenix_ui::Action::CaptureInputs
                                                                  : phoenix_ui::Action::ResetHistory;
                    require(send_action(value, action, command.frames), value.ui_error.c_str());
                    return "queued";
                }
                return "observed";
            });
    } catch (const std::exception& error) { value.ui_error = error.what(); }
}
inline void record_screenshot(reshade::api::effect_runtime* runtime, Session& value,
                              reshade::api::command_list* command, reshade::api::resource back) {
    if (value.screenshot_pending.empty() || value.screenshot || value.deferred_destroy) return;
    try {
        value.screenshot = std::make_unique<Snapshot>(runtime);
        value.screenshot->record(command, back, value.screenshot_pending);
    } catch (const std::exception& error) {
        value.screenshot_error = error.what();
        value.screenshot_pending.clear();
        value.screenshot.reset();
    }
}
inline void compact(reshade::api::effect_runtime* runtime) {
    if (!applies(runtime)) return;
    std::lock_guard lock(sessions_mutex);
    auto& value = session(runtime);
    auto ui = panel_context(runtime, value);
    if (phoenix_ui::draw_compact(ui)) save(runtime, value);
}
inline void panel(reshade::api::effect_runtime *runtime) {
    std::lock_guard lock(sessions_mutex);
    auto &value = session(runtime);
    if (ImGui::Button("Save configuration / apply"))
        save(runtime, value);
    ImGui::SameLine();
    if (ImGui::Checkbox("Enable graphics bridge", &value.enabled)) {
        value.restart = true;
        save(runtime, value);
    }
    ImGui::TextWrapped(
        "The game keeps its own graphics API. A separate 64-bit D3D12 helper runs the model.");
    ImGui::Text("Game API: %s", api_name(runtime->get_device()->get_api()));
    if (ImGui::Checkbox("Keep input and output from the same frame", &value.coherent_frames))
        reshade::set_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                                  value.coherent_frames);
    ImGui::TextWrapped(value.coherent_frames
                           ? "Hold the completed frame until the next result is ready. Camera "
                             "movement is delayed by processing time."
                           : "Experimental: apply the old model edit to the current image on "
                             "D3D9/D3D11. Moving objects can show through older masks.");
#ifndef _WIN64
    ImGui::TextUnformatted("32-bit game addon / 64-bit neural renderer");
#endif
    ImGui::TextWrapped("%s", value.message.c_str());
    auto *active_host = value.bridge     ? value.bridge->host()
                        : value.readback ? value.readback->host()
                                         : nullptr;
    if (active_host) {
        auto *host = active_host;
        ImGui::Text("Completed: %llu | helper round trip: %.2f ms",
                    (value.bridge ? value.bridge->delivered() : value.readback->delivered()),
                    host->roundtrip());
        ImGui::TextWrapped(value.coherent_frames
                               ? "Display: matched input/output; last completed frame held."
                           : value.bridge || value.readback->presents_current_frame()
                               ? "Display: current game frame plus the completed model edit."
                               : "Display: delayed completed image (readback transport).");
        if (value.readback) {
            const auto timings = value.readback->transfer_timings();
            ImGui::Text("Last transfer CPU calls: capture %.2f | upload %.2f | display %.2f ms",
                        timings["last_capture_ms"].get<double>(),
                        timings["last_upload_ms"].get<double>(),
                        timings["last_compose_ms"].get<double>());
        }
        if (host->phase() == Phase::Starting)
            ImGui::TextUnformatted("Starting / validating model... (game rendering continues)");
        if (host->phase() == Phase::Failed)
            ImGui::TextWrapped("Helper stopped: %s", host->error().c_str());
    }
    ImGui::Separator();
    auto ui = panel_context(runtime, value);
    if (phoenix_ui::draw_controls(ui))
        value.settings_dirty = true;
    phoenix_ui::draw_timing(ui);
    if (value.settings_dirty)
        ImGui::TextWrapped("Settings changed. Save configuration / apply to restart the helper "
                           "with this selection.");
    if (!value.ui_error.empty())
        ImGui::TextWrapped("%s", value.ui_error.c_str());
}
inline void destroy(reshade::api::effect_runtime *runtime) {
    std::lock_guard lock(sessions_mutex);
    sessions.erase(runtime);
}
inline void destroy_graphics_device(reshade::api::device *device) {
    std::lock_guard lock(sessions_mutex);
    sharing_devices.erase(device);
}
inline void process(reshade::api::effect_runtime *runtime, reshade::api::command_list *command,
                    reshade::api::resource back_resource) {
    if (!applies(runtime))
        return;
    std::lock_guard lock(sessions_mutex);
    auto &value = session(runtime);
    poll_control(runtime, value);
    ++value.frames;
    ++value.mode_frames[size_t(value.applied_settings.mode)];
    if (value.recorded_mode != value.applied_settings.mode) {
        if (value.recorded_mode >= 0) ++value.switches;
        value.recorded_mode = value.applied_settings.mode;
    }
    ++value.fps_frames;
    const auto now = GetTickCount64();
    if (!value.fps_tick)
        value.fps_tick = now;
    if (now - value.fps_tick >= 1000) {
        value.present_fps = value.fps_frames * 1000.0 / double(now - value.fps_tick);
        value.fps_frames = 0;
        value.fps_tick = now;
    }
    if (value.capture_hotkey >= 0) {
        if (now > value.hotkey_deadline)
            value.capture_hotkey = -1;
        else
            for (unsigned key = 1; key <= 255; ++key)
                if (runtime->is_key_pressed(key)) {
                    const auto result = phoenix_ui::capture_binding(value.hotkeys, value.capture_hotkey, key);
                    if (result == phoenix_ui::BindingResult::Accepted || result == phoenix_ui::BindingResult::Cancelled) {
                        value.capture_hotkey = -1;
                        save(runtime, value, false);
                    } else value.ui_error = phoenix_ui::binding_error(result);
                    break;
                }
    } else
        for (size_t index = 0; index < value.hotkeys.size(); ++index) {
            if (!value.hotkeys[index] || !runtime->is_key_pressed(unsigned(value.hotkeys[index])))
                continue;
            if (index == 0) {
                value.settings.mode = (value.applied_settings.mode + 1) % 4;
                save(runtime, value);
            }
            if (index == 1)
                send_action(value, phoenix_ui::Action::ResetHistory);
            if (index == 2) {
                ++value.markers;
                send_action(value, phoenix_ui::Action::Marker);
            }
            if (index == 3 && value.settings.experimental) {
                value.settings.transformer_blocks = value.settings.transformer_blocks == 1
                                                        ? 8
                                                        : value.settings.transformer_blocks - 1;
                value.settings.minimum_blocks =
                    std::min(value.settings.minimum_blocks, value.settings.transformer_blocks);
                save(runtime, value);
            }
            if (index == 4)
                send_action(value, phoenix_ui::Action::ClearTiming);
            if (index == 5)
                send_action(value, phoenix_ui::Action::CaptureInputs,
                            value.settings.capture_burst ? value.settings.capture_burst : 8);
        }
    // A failed Vulkan recording still belongs to ReShade's pending Present;
    // finish_present submits the application's waits before we retire it.
    if (value.deferred_destroy) return;
    if (!value.enabled) {
        remember_timing(value);
        value.bridge.reset();
        value.readback.reset();
        return;
    }
    if (value.applied_settings.mode == phoenix_addon::phoenix::input) {
        remember_timing(value);
        value.bridge.reset();
        value.readback.reset();
        value.message = "Original game (model bypass).";
        return;
    }
    auto *device = runtime->get_device();
    if (value.restart) {
        remember_timing(value);
        value.bridge.reset();
        value.readback.reset();
        value.shared_unavailable = false;
        value.restart = false;
        value.logged_outputs = 0;
    }
    if (device->get_api() != reshade::api::device_api::d3d11 || value.shared_unavailable) {
        try {
            const auto back = back_resource;
            const auto description = device->get_resource_desc(back);
            if (value.restart || value.width != description.texture.width ||
                value.height != description.texture.height ||
                value.readback_format != description.texture.format) {
                remember_timing(value);
                value.readback.reset();
                value.restart = false;
                value.width = description.texture.width;
                value.height = description.texture.height;
                value.readback_format = description.texture.format;
                value.logged_outputs = 0;
            }
            if (!value.readback) {
                const auto selected = value.configured;
                require(
                    selected.external() || value.applied_settings.model_source != "configured",
                    "No model selected. Browse to your NVIDIA DLL or a compatible model folder.");
                ++value.helper_generation;
                value.readback = std::make_unique<ReadbackBridge>(runtime, description, module,
                                                                  selected, value.applied_settings);
                value.message = std::string(api_name(device->get_api())) +
                                " readback / asynchronous 64-bit helper. CPU pixel transfers add "
                                "latency and bandwidth.";
                reshade::log::message(reshade::log::level::info, value.message.c_str());
            }
            value.readback->frame(command, back, value.frames, value.applied_settings,
                                  value.coherent_frames);
            log_progress(value);
        } catch (const std::exception &error) {
            value.message = error.what();
            reshade::log::message(reshade::log::level::error, value.message.c_str());
            if (device->get_api() == reshade::api::device_api::vulkan)
                value.deferred_destroy = true;
            else value.readback.reset();
            value.enabled = false;
        }
        return;
    }
    try {
        auto *native_texture = reinterpret_cast<ID3D11Texture2D *>(uintptr_t(back_resource.handle));
        require(native_texture != nullptr, "No presentation texture");
        Ptr<ID3D11Texture2D> back = native_texture;
        D3D11_TEXTURE2D_DESC description{};
        back->GetDesc(&description);
        require(description.SampleDesc.Count == 1, "Multisampled presentation is not supported");
        if (value.width != description.Width || value.height != description.Height ||
            value.format != description.Format) {
            remember_timing(value);
            value.bridge.reset();
            value.width = description.Width;
            value.height = description.Height;
            value.format = description.Format;
        }
        if (value.restart) {
            remember_timing(value);
            value.bridge.reset();
            value.restart = false;
            value.logged_outputs = 0;
        }
        if (!value.bridge) {
            const auto selected = value.configured;
            require(selected.external() || value.applied_settings.model_source != "configured",
                    "No model selected. Browse to your NVIDIA DLL or a compatible model folder.");
            auto *native_device = reinterpret_cast<ID3D11Device *>(uintptr_t(device->get_native()));
            ++value.helper_generation;
            auto &sharing = sharing_devices[device];
            value.bridge = std::make_unique<D3D11Bridge>(native_device, value.width, value.height,
                                                         value.format, sharing.Get());
            if (!sharing) sharing = value.bridge->sharing_device();
            value.bridge->start(module, selected, value.applied_settings);
            value.message = "D3D11 shared GPU textures / asynchronous 64-bit D3D12 helper (SDR).";
        }
        value.bridge->frame(back.Get(), value.frames, value.applied_settings,
                            value.coherent_frames);
        log_progress(value);
    } catch (const std::exception &error) {
        value.message = error.what();
        reshade::log::message(reshade::log::level::warning, value.message.c_str());
        value.bridge.reset();
        // Windows/driver versions without D3D11 shared NT fences still have
        // the staging route; lack of sharing is not an unsupported game API.
        value.shared_unavailable = true;
    }
}
inline void initialize(reshade::api::effect_runtime *runtime) {
    if (applies(runtime)) {
        std::lock_guard lock(sessions_mutex);
        session(runtime);
    }
}
inline void present(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain,
                    const reshade::api::rect *, const reshade::api::rect *, uint32_t,
                    const reshade::api::rect *) {
    if (!queue || !swapchain)
        return;
    std::lock_guard lock(sessions_mutex);
    const auto back = swapchain->get_current_back_buffer();
    for (auto &[runtime, value] : sessions) {
        if (runtime->get_device() != queue->get_device() ||
            runtime->get_hwnd() != swapchain->get_hwnd() ||
            runtime->get_current_back_buffer().handle != back.handle)
            continue;
        if (runtime->get_command_queue() != queue) {
            value->message =
                "Presentation and graphics queues differ; resource ownership is not proven.";
            return;
        }
        // Present is independent of ReShade shader loading. Process before its
        // effects and overlay, so neither a dummy shader nor UI filtering is needed.
        process(runtime, queue->get_immediate_command_list(), back);
        record_screenshot(runtime, *value, queue->get_immediate_command_list(), back);
        return;
    }
}
inline void finish_present(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain) {
    if (!queue || !swapchain) return;
    std::lock_guard lock(sessions_mutex);
    const auto back = swapchain->get_current_back_buffer();
    for (auto &[runtime, value] : sessions) {
        if (runtime->get_command_queue() != queue || runtime->get_hwnd() != swapchain->get_hwnd()) continue;
        if (value->readback && queue->get_device()->get_api() == reshade::api::device_api::vulkan) {
            try {
                value->readback->signal_after_present();
                if (value->deferred_destroy) {
                    value->readback.reset();
                    value->deferred_destroy = false;
                }
            } catch (const std::exception &error) {
                value->message = error.what();
                reshade::log::message(reshade::log::level::error, value->message.c_str());
                value->readback.reset();
                value->deferred_destroy = false;
                value->enabled = false;
            }
        }
        if (value->screenshot) {
            try {
                value->screenshot->signal_after_present();
                if (value->screenshot->poll()) {
                    value->screenshot_complete = value->screenshot_pending;
                    value->screenshot_pending.clear();
                    value->screenshot.reset();
                }
            } catch (const std::exception& error) {
                value->screenshot_error = error.what();
                value->screenshot_pending.clear();
                value->screenshot.reset();
            }
        }
        return;
    }
}
inline void attach(HMODULE addon) {
    module = addon;
    reshade::register_event<reshade::addon_event::init_effect_runtime>(initialize);
    reshade::register_event<reshade::addon_event::present>(present);
    reshade::register_event<reshade::addon_event::finish_present>(finish_present);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(destroy);
    reshade::register_event<reshade::addon_event::destroy_device>(destroy_graphics_device);
}
inline void detach() {
    reshade::unregister_event<reshade::addon_event::init_effect_runtime>(initialize);
    reshade::unregister_event<reshade::addon_event::present>(present);
    reshade::unregister_event<reshade::addon_event::finish_present>(finish_present);
    reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(destroy);
    reshade::unregister_event<reshade::addon_event::destroy_device>(destroy_graphics_device);
    std::lock_guard lock(sessions_mutex);
    sessions.clear();
    sharing_devices.clear();
    module = nullptr;
}
// Read-only bridge diagnostics, usable by a matching-bitness host without
// game-window automation. The required byte count includes the trailing NUL.
extern "C" __declspec(dllexport) uint32_t CnrPHOENIXCompatibilityStatus(char *destination,
                                                                        uint32_t capacity) {
    std::lock_guard lock(sessions_mutex);
    Json result = Json::array();
    for (auto &[runtime, value] : sessions) {
        auto *host = value->bridge     ? value->bridge->host()
                     : value->readback ? value->readback->host()
                                       : nullptr;
        result.push_back(
            {{"api", unsigned(runtime->get_device()->get_api())},
             {"window_id", uint64_t(reinterpret_cast<uintptr_t>(runtime->get_hwnd()))},
             {"api_name", api_name(runtime->get_device()->get_api())},
             {"enabled", value->enabled},
             {"control_folder", phoenix_model::source_path_string(value->control.folder())},
             {"settings", settings_json(value->settings)},
             {"applied_settings", settings_json(value->applied_settings)},
             {"width", value->width},
             {"height", value->height},
             {"message", value->message},
             {"phase", host ? unsigned(host->phase()) : unsigned(Phase::Stopped)},
             {"delivered", value->bridge     ? value->bridge->delivered()
                           : value->readback ? value->readback->delivered()
                                             : 0},
             {"presented", value->bridge     ? value->bridge->presented()
                           : value->readback ? value->readback->presented()
                                             : 0},
             {"composition",
              value->coherent_frames ? "matched_frame_hold"
              : value->bridge || (value->readback && value->readback->presents_current_frame())
                  ? "current_frame_edit"
                  : "delayed_frame"},
             {"roundtrip_ms", host ? host->roundtrip() : 0},
             {"transfer_cpu_ms",
              value->readback ? value->readback->transfer_timings()
                              : value->bridge ? value->bridge->transfer_timings() : Json::object()},
             {"error", host ? host->error() : std::string()}});
    }
    const auto text = result.dump();
    const auto required = uint32_t(text.size() + 1);
    if (destination && capacity >= required)
        std::memcpy(destination, text.c_str(), required);
    return required;
}
} // namespace phoenix_compat
