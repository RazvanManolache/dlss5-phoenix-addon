#pragma once
#include "../../native/support/model_tools.h"
#include "../reshade/shared_panel.h"
#include "../reshade/comparison_hud.h"
#include "d3d11_bridge.h"
#include "readback_bridge.h"
#include "backbuffer_route.h"
#include "transport_availability.h"
#include "d3d12_bridge.h"
#include "session_policy.h"
#include "screenshot.h"
#include "../../native/support/control_mailbox.h"
#include <map>

namespace phoenix_compat {
struct Session {
    phoenix_addon::phoenix::Settings settings;
    phoenix_addon::phoenix::Settings applied_settings;
    phoenix_addon::phoenix::Settings displayed_settings;
    std::array<phoenix_addon::phoenix::Settings, 4> presets;
    std::array<int, 7> hotkeys{{VK_F9, VK_F10, VK_F11, VK_F8, VK_F7, 0, 0}};
    int capture_hotkey = -1;
    uint64_t hotkey_deadline = 0, markers = 0, fps_tick = 0, fps_frames = 0;
    double present_fps = 0;
    bool settings_dirty = false;
    std::vector<phoenix_addon::phoenix::ExecutionDevice> execution_devices{
        {"native", "Native RTX"}};
    std::string ui_error;
    Json previous_gpu_rows = Json::array(), previous_worker_rows = Json::array();
    std::unique_ptr<D3D11Bridge> bridge;
    std::unique_ptr<ReadbackBridge> readback;
    std::unique_ptr<D3D12Bridge> local;
    ExecutionPreference execution = ExecutionPreference::Automatic;
    std::string execution_name = "auto", fallback_reason, processing_error;
    std::string portable_root, portable_device = "auto", helper_transport = "auto";
    std::string native_root;
    bool native_fallback=true, native_library_failed=false, native_driver_failed=false;
    bool portable_allow_unvalidated = false, portable_fallback = true, portable_failed = false;
    bool local_failed = false;
    TransportAvailabilityProbe availability_probe;
    TransportRuntimes runtime_availability;
    BackbufferCapabilities capabilities, selectable_capabilities;
    std::string native_library_reason, native_driver_reason, portable_reason;
    std::vector<std::string> failed_routes;
    std::optional<bool> software_device;
    reshade::api::color_space color_space = reshade::api::color_space::srgb_nonlinear;
    phoenix_model_tools::Task picker, validator;
    fs::path output_directory;
    phoenix_addon::phoenix::control_bridge::Mailbox control;
    uint64_t switches = 0;
    uint64_t selection_revision = 1, presented_revision = 0;
    uint64_t local_composed_frame = 0;
    uint64_t local_composed_at_frame = 0;
    bool displayed = false;
    bool timing_clear_pending = false;
    uint64_t timing_clear_revision = 0, timing_clear_generation = 0, helper_generation = 0;
    bool pair_requested=false;
    Json screenshot_pair=Json::object();
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
    uint64_t presentation_holds = 0, loading_presentations = 0;
    unsigned width = 0, height = 0;
    uint64_t frames = 0;
    uint64_t logged_outputs = 0;
    reshade::api::format readback_format = reshade::api::format::unknown;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::string message = "Select a model and enable neural rendering.";
};
inline HMODULE module = nullptr;
template<class Bridge> inline void prepare_pair_request(Session& value,Bridge* bridge) {
    if(value.pair_requested) {
        if(!bridge->request_screenshot_pair(value.output_directory)) value.screenshot_pair={{"state","error"},{"error","A screenshot pair is already pending."}};
        value.pair_requested=false;
    } else if(value.screenshot_pair.value("state",std::string())=="pending" && bridge->screenshot_pair_status().empty()) {
        value.screenshot_pair={{"state","error"},{"error","Capture canceled because the processing route or resolution changed. Please request another pair."}};
    }
}
inline void update_pair_result(Session& value,const Json& result) {
    if(!result.empty() && result!=value.screenshot_pair)value.screenshot_pair=result;
}
inline void reset_route_failures(Session& value) {
    value.local_failed = value.portable_failed = value.native_library_failed = value.native_driver_failed = false;
    value.failed_routes.clear(); value.fallback_reason.clear();
}
inline void reject_route(Session& value, BackbufferRoute route, const std::string& error) {
    if (route == BackbufferRoute::InProcess) value.local_failed = true;
    else if (route == BackbufferRoute::NativeLibrary) value.native_library_failed = true;
    else if (route == BackbufferRoute::NativeDriver) value.native_driver_failed = true;
    else if (route == BackbufferRoute::Portable) value.portable_failed = true;
    value.failed_routes.push_back(error);
    value.fallback_reason = error;
    value.message = error + "; trying the next eligible route.";
    value.restart = true;
}
inline std::filesystem::path runtime_path(const std::string& configured, const std::filesystem::path& fallback) {
    wchar_t path[32768]{};
    require(GetModuleFileNameW(module, path, 32768) != 0, "Addon path unavailable");
    return transport_runtime_path(std::filesystem::path(path).parent_path(), configured, fallback);
}
inline BackbufferCapabilities transport_capabilities(Session& value, const Settings& settings,
                                                       BackbufferCapabilities caps) {
    try {
        value.runtime_availability = value.availability_probe.poll(
            runtime_path(value.native_root, "PhoenixRuntime/wow64"),
            runtime_path(value.portable_root, sizeof(void*) == 4 ? "PhoenixRuntime/win-x86" : "PhoenixRuntime/win-x64"),
            value.portable_device, value.portable_allow_unvalidated);
    } catch (const std::exception& error) {
        value.runtime_availability = {false, false, false, error.what(), error.what()};
    }
    caps.discovery_pending = value.runtime_availability.pending;
    caps.local_api = in_process_api.available();
    caps.native_executor = in_process_settings(settings).execution_device == "native";
    caps.nvidia_required = needs_nvidia_dll(settings);
    caps.native_library_available = value.runtime_availability.native;
    caps.native_driver_available = value.runtime_availability.native && value.runtime_availability.driver_graph;
    caps.native_tls_coordinated = value.runtime_availability.native;
    value.native_library_reason = value.runtime_availability.native_reason;
    const auto model = settings.model_id.empty() ? value.configured.model_id : settings.model_id;
    if (caps.native_library_available && std::find(value.runtime_availability.native_models.begin(),
            value.runtime_availability.native_models.end(), model.empty() ? "original" : model) == value.runtime_availability.native_models.end()) {
        caps.native_library_available = false;
        value.native_library_reason = "The native runtime package does not declare support for this model.";
    }
    value.native_driver_reason = value.runtime_availability.native_reason;
    if (value.runtime_availability.native && !value.runtime_availability.driver_graph)
        value.native_driver_reason = value.runtime_availability.driver_reason;
    for (bool driver : {false, true}) {
        const char* reason = native_settings_unavailable_reason(settings, driver);
        if (caps.bits == 32 && caps.api == reshade::api::device_api::opengl && !caps.native_tls_coordinated)
            reason = "Native x86 OpenGL requires the matching TLS-coordinated v3 addon/runtime.";
        if (!*reason && driver) {
            const auto image = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
            const auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
            const auto pe = reinterpret_cast<const IMAGE_NT_HEADERS*>(image + dos->e_lfanew);
            if (!(pe->FileHeader.Characteristics & IMAGE_FILE_LARGE_ADDRESS_AWARE))
                reason = "The x86 driver host requires a large-address-aware executable.";
        }
        if (*reason) {
            (driver ? caps.native_driver_available : caps.native_library_available) = false;
            (driver ? value.native_driver_reason : value.native_library_reason) = reason;
        }
    }
    caps.portable_available = value.runtime_availability.portable &&
        settings.resolution_mode == 0 && !settings.host_unjitter && !caps.nvidia_required;
    value.portable_reason = value.runtime_availability.portable_reason;
    if (settings.resolution_mode != 0 || settings.host_unjitter)
        value.portable_reason = "Portable backbuffer execution requires normal resolution and no unjitter.";
    if (caps.nvidia_required) value.portable_reason = "Portable processing cannot evaluate the NVIDIA DLL.";
    caps.native_library_reason=value.native_library_reason;
    caps.native_driver_reason=value.native_driver_reason;
    caps.portable_reason=value.portable_reason;
    // Failure history controls Automatic's next attempt, while compatible
    // choices remain available for an explicit retry. Status reads use the
    // cached selection rather than starting discovery or mutating the session.
    value.selectable_capabilities = caps;
    caps.local_api &= !value.local_failed;
    caps.native_library_available &= !value.native_library_failed;
    caps.native_driver_available &= !value.native_driver_failed;
    caps.portable_available &= !value.portable_failed;
    return caps;
}
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
        value.message = "Processor stopped; holding the last completed frame. " + host->error();
    }
    const auto outputs = value.bridge ? value.bridge->delivered() : value.readback->delivered();
    const auto presented = value.bridge ? value.bridge->presented() : value.readback->presented();
    if (outputs > value.logged_outputs &&
        (outputs <= 3 || outputs / 100 > value.logged_outputs / 100)) {
        const auto message = "Phoenix DLSS5 bridge: completed=" + std::to_string(outputs) +
                             ", presented=" + std::to_string(presented) + ", " + value.message;
        reshade::log::message(reshade::log::level::info, message.c_str());
    }
    if (outputs > value.logged_outputs && host->tuning_current_output()) {
        value.presented_revision = value.selection_revision;
        auto& counts = value.applied_settings.mode == phoenix_addon::phoenix::nvidia ? value.mode_native : value.mode_own;
        counts[size_t(value.applied_settings.mode)] += outputs - value.logged_outputs;
    }
    value.logged_outputs = outputs;
    if (presented && value.presented_revision == value.selection_revision) {
        value.displayed_settings = value.applied_settings;
        value.displayed = true;
    }
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
    if (!runtime) return false;
    if (uses_helper(runtime->get_device()->get_api(), sizeof(void *) * 8)) return true;
    // D3D12 games without a usable game-DLSS input also need a backbuffer
    // frontend. Keep this route separate from its on/off switch so disabling
    // enhancement does not replace the compatibility UI or its mailbox.
    bool standalone_input = false;
    reshade::get_config_value(runtime, "Phoenix.Compatibility", "StandaloneInput", standalone_input);
    return standalone_input;
#endif
}
inline Session &session(reshade::api::effect_runtime *runtime) {
    auto &entry = sessions[runtime];
    if (!entry) {
        entry = std::make_unique<Session>();
        bool library_folder_saved = false;
        bool model_library_saved = false;
        phoenix_addon::phoenix::visit_settings(entry->settings, [&](const char *key, auto &value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
                char text[4096]{};
                size_t length = sizeof(text);
                if (reshade::get_config_value(runtime, "Phoenix", key, text, &length)) {
                    value.assign(text, strnlen_s(text, sizeof(text)));
                    if (std::string_view(key) == "NvidiaDllFolder") library_folder_saved = true;
                    if (std::string_view(key) == "ModelLibraryDirectory") model_library_saved = true;
                }
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
        migrate_processing_switch(entry->enabled, entry->settings);
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                                  entry->coherent_frames);
        char execution[32]{}; size_t execution_bytes = sizeof(execution);
        if (reshade::get_config_value(runtime, "Phoenix.Compatibility", "ExecutionMode", execution, &execution_bytes))
            entry->execution_name.assign(execution, strnlen_s(execution, sizeof(execution)));
        entry->execution = execution_preference(entry->execution_name);
        for (auto field : {std::pair{"PortableRuntime", &entry->portable_root},
                           std::pair{"PortableDevice", &entry->portable_device},
                           std::pair{"HelperTransport", &entry->helper_transport}}) {
            char text[4096]{}; size_t bytes = sizeof(text);
            if (reshade::get_config_value(runtime, "Phoenix.Compatibility", field.first, text, &bytes))
                field.second->assign(text, strnlen_s(text, sizeof(text)));
        }
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "PortableAllowUnvalidated", entry->portable_allow_unvalidated);
        reshade::get_config_value(runtime, "Phoenix.Compatibility", "PortableFallback", entry->portable_fallback);
        char native_path[4096]{};size_t native_bytes=sizeof(native_path);
        if(reshade::get_config_value(runtime,"Phoenix.Compatibility","NativeRuntime",native_path,&native_bytes)) entry->native_root.assign(native_path,strnlen_s(native_path,sizeof(native_path)));
        reshade::get_config_value(runtime,"Phoenix.Compatibility","NativeFallback",entry->native_fallback);
        if (normalize_capabilities(entry->settings))
            entry->ui_error = "FPS budgeting is unavailable in a companion; using GPU-time budget.";
        phoenix_addon::phoenix::normalize_direct_comparison(entry->settings);
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
                if (!library_folder_saved)
                    entry->settings.nvidia_dll_folder = config.value("nvidia_dll_folder", std::string());
                if (!model_library_saved)
                    entry->settings.model_library_directory = config.value("model_library_directory", std::string());
                if (entry->portable_root.empty() && config.contains("portable_cpp"))
                    entry->portable_root = config["portable_cpp"].value("runtime_root", std::string());
            }
        } catch (const std::exception &error) {
            entry->message = error.what();
        }
        entry->applied_settings.nvidia_dll_folder = entry->settings.nvidia_dll_folder;
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
    phoenix_addon::phoenix::normalize_direct_comparison(value.settings);
    if (normalize_capabilities(value.settings))
        value.ui_error = "FPS budgeting is unavailable in a companion; using GPU-time budget.";
    for (auto& preset : value.presets) normalize_capabilities(preset);
    if (apply) migrate_processing_switch(value.enabled, value.settings);
    auto persisted_settings = apply ? value.settings : value.applied_settings;
    phoenix_addon::phoenix::visit_settings(persisted_settings, [&](const char *key, auto &field) {
        if constexpr (std::is_same_v<std::decay_t<decltype(field)>, std::string>)
            reshade::set_config_value(runtime, "Phoenix", key, field.c_str());
        else
            reshade::set_config_value(runtime, "Phoenix", key, field);
    });
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "Enabled", value.enabled);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                              value.coherent_frames);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "ExecutionMode", value.execution_name.c_str());
    reshade::set_config_value(runtime,"Phoenix.Compatibility","NativeRuntime",value.native_root.c_str());
    reshade::set_config_value(runtime,"Phoenix.Compatibility","NativeFallback",value.native_fallback);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "PortableRuntime", value.portable_root.c_str());
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "PortableDevice", value.portable_device.c_str());
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "HelperTransport", value.helper_transport.c_str());
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "PortableAllowUnvalidated", value.portable_allow_unvalidated);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "PortableFallback", value.portable_fallback);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeFrames",
                              persisted_settings.offload_age_frames);
    reshade::set_config_value(runtime, "Phoenix.Compatibility", "OffloadAgeMs",
                              persisted_settings.offload_age_ms);
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
        const bool retry = !value.processing_error.empty();
        value.processing_error.clear();
        if (retry) reset_route_failures(value);
        const bool changed = helper_settings_changed(value.applied_settings, value.settings);
        auto tuning = value.applied_settings;
        const bool tuning_changed = phoenix_addon::phoenix::copy_live_tuning(tuning, value.settings);
        if (changed || tuning_changed) ++value.selection_revision;
        value.restart = value.restart || changed || retry;
        value.applied_settings = value.settings;
        value.settings_dirty = false;
    }
}
inline void apply_ui_edits(reshade::api::effect_runtime* runtime, Session& value,
                           const phoenix_ui::Context& ui) {
    value.settings.uplift = true; // Legacy presets cannot create a hidden second switch.
    // Live controls update even during a drag. Resource edits wait until the
    // editing gesture ends, rather than rebuilding on every intermediate value.
    const auto previous_selection=value.applied_settings;
    const bool hud_only=hud_only_change(previous_selection,value.settings);
    bool selection = false;
    if (!ImGui::IsAnyItemActive())
        selection = apply_live_selection(value.applied_settings, value.settings);
    const auto* view_host=value.bridge?value.bridge->host():value.readback?value.readback->host():nullptr;
    const bool portable_view=selection&&view_host&&view_host->in_process_portable()&&
        previous_selection.mode==value.applied_settings.mode&&!needs_nvidia_dll(value.applied_settings)&&
        portable_view_only(previous_selection,value.applied_settings);
    const bool tuning = phoenix_addon::phoenix::copy_live_tuning(value.applied_settings, value.settings);
    if (selection || ui.processing_changed || tuning) {
        const bool retry = !value.processing_error.empty();
        if(!hud_only)++value.selection_revision;
        value.settings.uplift = value.applied_settings.uplift = true;
        value.processing_error.clear(); value.local_failed = false;
        if (!hud_only && (selection || ui.processing_changed || retry)) reset_route_failures(value);
        if (!value.portable_failed) value.fallback_reason.clear();
        value.restart = value.restart || (selection&&!portable_view&&!hud_only) || ui.processing_changed || retry;
        save(runtime, value, false);
    }
    value.settings_dirty = settings_json(value.settings) != settings_json(value.applied_settings);
    // Per-frame controls already took effect above, even while a slider is
    // active. Rebuild-only edits commit once the editing gesture has ended.
    if (value.settings_dirty && !ImGui::IsAnyItemActive())
        save(runtime, value);
}
inline HostClient *active_host(Session &value) {
    return value.bridge ? value.bridge->host() : value.readback ? value.readback->host() : nullptr;
}
inline bool portable_active(Session& value) {
    const auto* client = active_host(value);
    return client && client->in_process_portable();
}
inline bool native_active(Session& value) {const auto* client=active_host(value);return client&&client->native_execution();}
inline bool resident_active(Session& value) {const auto* client=active_host(value);return client&&client->in_process();}
inline const char* execution_mode(Session& value) {
    if(native_active(value)) return active_host(value)->native_execution();
    return value.local ? "in_process" : portable_active(value) ? "in_process_portable" :
        active_host(value) ? "helper" : "not_started";
}
inline const char* timing_provenance(Session& value) {
    if(native_active(value)) return "Same-process native worker; host pixel transfer and native dispatch included in wall time";
    return value.local ? "In-process GPU (same-device D3D12)" : portable_active(value) ?
        "In-process portable (host preparation, inference and resolve; CPU pixel transfers)" :
        active_host(value) ? "Helper GPU (D3D12 companion)" : "Processing has not started";
}
inline Json processing_info(Session& value) {
    if (value.local) {
        try { return value.local->ui_snapshot(); }
        catch (const std::exception& error) { return Json{{"error", error.what()}}; }
    }
    auto* host = active_host(value);
    return host ? host->ui_snapshot() : Json::object();
}
inline Json inference_processes(Session& value){
    auto* host=active_host(value);
    const auto fallback=value.local||resident_active(value)?GetCurrentProcessId():host?host->process_id():0;
    const auto info=processing_info(value);
    return {{"processing_host_pid",info.value("processing_host_pid",fallback)},
        {"inference_pid",info.value("inference_pid",fallback)},
        {"inference_pids",info.value("inference_pids",fallback?Json::array({fallback}):Json::array())}};
}
inline uint64_t delivered(Session& value) {
    return value.local ? value.local->delivered() : value.bridge ? value.bridge->delivered() :
           value.readback ? value.readback->delivered() : 0;
}
inline uint64_t presented(Session& value) {
    return value.local ? value.local->presented() : value.bridge ? value.bridge->presented() :
           value.readback ? value.readback->presented() : 0;
}
inline Phase processing_phase(Session& value) {
    if (!value.processing_error.empty()) return Phase::Failed;
    if (in_process_gpu_uncertain) return Phase::Failed;
    if (value.local && !value.local->safe_to_fallback()) return Phase::Failed;
    if (!value.enabled || value.applied_settings.mode == phoenix_addon::phoenix::input) return Phase::Stopped;
    if (value.restart) return Phase::Starting;
    if (value.local) return Phase::Ready;
    auto* host = active_host(value);
    return host ? host->phase() : Phase::Starting;
}
inline void remember_timing(Session &value) {
    auto *host = active_host(value);
    if ((!host && !value.local) || value.timing_clear_pending)
        return;
    const auto info = processing_info(value);
    for (auto item : {std::pair{"gpu_timing", &value.previous_gpu_rows},
                      std::pair{"worker_timing", &value.previous_worker_rows}}) {
        const auto report = info.value(item.first, Json::object());
        if (!report.is_object())
            continue;
        for (auto row : report.value("rows", Json::array())) {
            if (host && std::string(item.first) == "gpu_timing")
                row["settings"]["HelperProcess"] = host->process_id();
            item.second->push_back(std::move(row));
        }
        // Bound UI-only history independently of inference allocations.
        while (item.second->size() > 64)
            item.second->erase(item.second->begin());
    }
}
inline bool send_action(Session &value, phoenix_ui::Action action, unsigned argument = 0) {
    if(action==phoenix_ui::Action::SaveScreenshotPair) {
        if(!value.enabled || value.applied_settings.mode==phoenix_addon::phoenix::input || value.pair_requested || value.screenshot_pair.value("state",std::string())=="pending") {
            value.ui_error="Enable processed output before capturing a pair; an existing request must finish first.";return false;
        }
        value.pair_requested=true;value.screenshot_pair={{"state","pending"}};value.ui_error.clear();return true;
    }
    if (value.local) {
        if (action == phoenix_ui::Action::ClearTiming) {
            value.previous_gpu_rows = Json::array(); value.previous_worker_rows = Json::array();
        }
        const bool accepted = value.local->action(action, argument);
        value.ui_error = accepted ? "" : "In-process Phoenix DLSS5 action was rejected.";
        return accepted;
    }
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
        value.ui_error = host ? "This action is unavailable for the selected execution mechanism." :
                                "Enable neural rendering before using this action.";
        return false;
    }
    value.ui_error.clear();
    return true;
}
inline phoenix_ui::Context panel_context(reshade::api::effect_runtime *runtime, Session &value) {
    auto *host = active_host(value);
    auto info = processing_info(value);
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
    ui.nvidia_status = info.value("nvidia", Json::object()).value("status", std::string());
    ui.processing_enabled = &value.enabled;
    ui.timing_provenance = timing_provenance(value);
    ui.pipeline_status = info.value("pipeline_status", Json::object());
    const bool portable_services=portable_active(value)&&info.value("runtime",Json::object()).value("pipeline_frame_services",false);
    ui.portable_chain_available=portable_active(value)&&
        info.value("runtime",Json::object()).value("portable_pipeline_contract",0)>=1&&
        (portable_services||!value.settings.processing.enabled||value.settings.processing.scale==100);
    ui.portable_backend_selected=portable_active(value)||value.execution==ExecutionPreference::Portable;
    ui.inference_process = host && !resident_active(value);
    const bool native_requested = native_active(value) ||
        (value.execution == ExecutionPreference::NativeLibrary && !value.native_library_failed) ||
        (value.execution == ExecutionPreference::NativeDriver && !value.native_driver_failed && !needs_nvidia_dll(value.settings));
    // Only the local D3D12 bridge owns motion/discontinuity/age services.
    // A final color frame by itself does not establish these capabilities.
    ui.frame_services_available=bool(value.local)||portable_services;
    ui.native_chain_available=!native_requested;
    ui.final_frame_source = true;
    ui.portable_backbuffer = portable_active(value) ||
        (value.execution == ExecutionPreference::Portable && !needs_nvidia_dll(value.settings)) || native_requested;
    ui.portable_gpu_budget = (portable_active(value) || value.execution == ExecutionPreference::Portable) &&
        !needs_nvidia_dll(value.settings) && value.portable_device != "cpu" &&
        info.contains("runtime") && info.at("runtime").is_object() &&
        info.value("runtime", Json::object()).value("capabilities", Json::object()).contains("granular_controls");
    ui.fixed_native_blocks = (native_active(value) && std::string(execution_mode(value)) == "in_process_x86_driver") ||
        (value.execution == ExecutionPreference::NativeDriver && !value.native_driver_failed && !needs_nvidia_dll(value.settings));
    if (resident_active(value)) ui.portable_wall_ms = info.value("frame_wall_ms", 0.0);
    ui.input_capture = !ui.portable_backbuffer && !native_active(value);
    ui.status = value.message;
    ui.route_failed = !value.processing_error.empty();
    ui.processing_active = value.displayed && value.presented_revision == value.selection_revision;
    const std::string actual = execution_mode(value);
    ui.active_route = actual == "in_process" ? "Native D3D12 / same-device processing"
        : actual == "in_process_portable" ? "Portable runtime / same-process processing"
        : actual == "in_process_x86_driver" ? "Native x86 driver / same-process processing"
        : actual == "in_process_x64_library" ? "Native x64 library / x86 game process"
        : actual == "helper" ? "External x64 helper / graphics bridge" : "Waiting for a processing route";
    ui.route_notice = value.fallback_reason;
    if (!value.processing_error.empty()) ui.route_notice += "\n" + value.processing_error;
    if (value.settings_dirty) ui.route_notice += "\nRenderer setup change pending.";
    ui.loaded_model_id = info.value("model_source", Json::object()).value("loaded_model_id", std::string());

    ui.switches = value.switches;
    ui.mode_frames = value.mode_frames;
    ui.mode_own = value.mode_own;
    ui.mode_native = value.mode_native;
    ui.real_guides = ui.temporal_history = false;
    ui.comparison_views = true; // The helper returns the selected composed view too.
    ui.fps_budget = false;
    ui.model_restart = "The selected model loads automatically when editing is finished.";
    ui.runtime_available = info.value("schema", std::string()) == "phoenix-panel/1";
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
    ui.active_nvidia_dll_folder = value.applied_settings.nvidia_dll_folder;
    ui.active_model_id = value.applied_settings.model_id;
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
    ui.screenshot_pair_available=value.enabled && value.applied_settings.mode!=phoenix_addon::phoenix::input;
    { phoenix_pipeline::screenshots::PairExport view;view.result=value.screenshot_pair;ui.screenshot_pair_status=view.message(); }
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
    const auto processes=inference_processes(value);
    const auto validation = value.validator.poll();
    Json transports = Json::array();
    // A failed attempt is excluded from Automatic until an explicit retry. It
    // does not remove a compatible route from the user's available choices.
    for (auto preference : available_execution_preferences(value.selectable_capabilities))
        transports.push_back({{"key", execution_preference_name(preference)},
                              {"label", execution_preference_label(preference)}});
    return {{"tick_ms", GetTickCount64()}, {"api", api_name(runtime->get_device()->get_api())},
        {"window_id", uint64_t(reinterpret_cast<uintptr_t>(runtime->get_hwnd()))},
        {"enabled", value.enabled}, {"coherent_frames", value.coherent_frames},
        {"presentation_gate", {{"held",value.presentation_holds},{"loading",value.loading_presentations},
            {"policy",value.coherent_frames?"completed_output_or_loading":"experimental_unmatched_output"}}},
        {"history_policy",value.applied_settings.pipeline.motion_enabled?
            "motion_provider_owns_history; unavailable guides reset recurrence":"spatial_reset_no_engine_motion_vectors"},
        {"settings", settings_json(value.settings)}, {"applied_settings", settings_json(value.applied_settings)},
        {"restart_pending", value.restart}, {"settings_pending", value.settings_dirty},
        {"selection_revision", value.selection_revision}, {"presented_revision", value.presented_revision},
        {"status", value.message},
        {"helper_pid", host && !resident_active(value) ? host->process_id() : 0},
        {"execution_mode", execution_mode(value)},
        {"execution_preference", value.execution_name}, {"fallback_reason", value.fallback_reason},
        {"transport_options", std::move(transports)}, {"transport_failures", value.failed_routes},
        {"transport_discovery_pending", value.runtime_availability.pending},
        {"process_address_space", {{"bits", sizeof(void*) * 8},
            {"queried", value.runtime_availability.address_space.queried},
            {"free_bytes", value.runtime_availability.address_space.free_bytes},
            {"largest_free_region", value.runtime_availability.address_space.largest_free_region}}},
        {"transport_availability", {{"native_library", value.native_library_reason},
                                    {"native_driver", value.native_driver_reason},
                                    {"portable", value.portable_reason}}},
        {"portable_runtime", value.portable_root}, {"portable_device", value.portable_device},
        {"helper_transport", value.helper_transport},
        {"portable_fallback", value.portable_fallback},
        {"processing_host_pid",processes.at("processing_host_pid")},
        {"inference_pid",processes.at("inference_pid")},{"inference_pids",processes.at("inference_pids")},
        {"phase", unsigned(processing_phase(value))},
        {"delivered", delivered(value)}, {"presented", presented(value)},
        // Resource-selection lineage, not a claim that pixels reached the display.
        // The live regression runner also requests actual presented screenshots.
        {"frame_lineage", value.readback ? value.readback->transfer_timings()
                                         : value.bridge ? value.bridge->transfer_timings()
                                         : value.local ? Json{{"input_frame", value.local_composed_frame},
                                                              {"output_frame", value.local_composed_frame},
                                                              {"composed_at_frame", value.local_composed_at_frame},
                                                              {"width", value.width}, {"height", value.height}}
                                                       : Json::object()},
        {"helper_roundtrip_ms", host && !resident_active(value) ? host->roundtrip() : 0},
        {"processing_roundtrip_ms", host ? host->roundtrip() : 0},
        {"timing_provenance", timing_provenance(value)},
        {"model_validation", {{"busy", validation.busy}, {"passed", validation.passed},
                              {"key", validation.key}, {"message", validation.message}}},
        {"helper", host && !resident_active(value) ? host->ui_snapshot() : Json::object()},
        {"in_process", value.local || resident_active(value) ? processing_info(value) : Json::object()},
        {"present_fps", value.present_fps},
        {"screenshot_pair",value.screenshot_pair},
        {"screenshot", {{"pending", !value.screenshot_pending.empty()},
                        {"path", phoenix_model::source_path_string(value.screenshot_complete)},
                        {"error", value.screenshot_error}}}};
}
inline void poll_control(reshade::api::effect_runtime* runtime, Session& value) {
    try {
        value.control.poll(value.applied_settings, [&] { return control_snapshot(runtime, value); },
            [&](const phoenix_addon::phoenix::control_bridge::Command& command, Json& response) -> std::string {
                if (command.op == "set") {
                    auto draft = value.settings;
                    value.settings = command.settings;
                    if (command.enabled) {
                        if (value.enabled != *command.enabled) ++value.selection_revision;
                        value.enabled = *command.enabled;
                        value.settings.uplift = true;
                    }
                    if (command.coherent_frames) value.coherent_frames = *command.coherent_frames;
                    save(runtime, value);
                    merge_settings_patch(draft, value.applied_settings, command.settings_keys);
                    apply_live_selection(draft, value.applied_settings);
                    value.settings = std::move(draft);
                    value.settings_dirty = settings_json(value.settings) != settings_json(value.applied_settings);
                    return value.restart ? "restart_queued" : "applied";
                }
                if (command.op == "save") { save(runtime, value); return "saved"; }
                if (command.op == "validate_model") {
                    require(!value.validator.poll().busy, "Model validation is already running");
                    const auto source = phoenix_model_tools::selection(value.settings, value.configured);
                    value.validator.validate(source, {}, module);
                    return "queued";
                }
                if (command.op == "screenshot_pair") {
                    require(send_action(value, phoenix_ui::Action::SaveScreenshotPair), value.ui_error.c_str());
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
    phoenix_ui::draw_compact(ui);
    apply_ui_edits(runtime, value, ui);
}
inline void comparison_hud(reshade::api::effect_runtime* runtime) {
    std::lock_guard lock(sessions_mutex);
    auto found = sessions.find(runtime);
    if (found == sessions.end()) return;
    auto& value = *found->second;
    if (value.enabled && value.displayed && value.presented_revision == value.selection_revision) {
        auto presentation=value.displayed_settings;
        presentation.comparison_labels=value.settings.comparison_labels;
        presentation.comparison_lines=value.settings.comparison_lines;
        phoenix_ui::draw_presentation_hud(presentation);
    }
}
// The same Setup tab hosts architecture-specific transport controls.
inline void draw_transport(reshade::api::effect_runtime* runtime, Session& value) {
    if (ImGui::CollapsingHeader("Advanced transport")) {
    ImGui::TextWrapped("%s", value.local ? "Phoenix DLSS5 runs inside the game on its D3D12 device." : native_active(value) ?
        "Processing runs inside this x86 game on a native 64-bit thread; no helper process. Pixel transfers still use CPU staging." : portable_active(value) ?
        "Phoenix DLSS5 runs inside the game using the portable runtime and CPU pixel transfers." :
        "Phoenix DLSS5 selects same-device D3D12 processing when supported, otherwise an isolated 64-bit helper.");
    bool route_changed = false;
    transport_capabilities(value,value.settings,value.capabilities);
    const auto choices = available_execution_preferences(value.selectable_capabilities);
    const bool saved_available = std::find(choices.begin(), choices.end(), value.execution) != choices.end();
    const auto preview = saved_available ? execution_preference_label(value.execution) : "Saved mode unavailable - choose a route";
    if (ImGui::BeginCombo("Execution mechanism", preview)) {
        for (auto choice : choices) {
            const bool selected = choice == value.execution;
            if (ImGui::Selectable(execution_preference_label(choice), selected)) {
                value.execution = choice; value.execution_name = execution_preference_name(choice);
                route_changed = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (!saved_available) ImGui::TextWrapped("The saved execution mechanism is not available for this architecture, runtime or settings. Select Automatic to try eligible routes.");
    ImGui::TextWrapped("Automatic tries same-device native DX12, eligible x86 native routes, then portable processing. The x64 helper is last.");
    if (ImGui::Button("Recheck runtimes / retry routes")) {
        value.availability_probe.invalidate(); route_changed = true;
    }
    if (ImGui::TreeNode("Transport availability")) {
        char portable_folder[4096]{};
        strncpy_s(portable_folder, value.portable_root.c_str(), _TRUNCATE);
        if (ImGui::InputText("Portable runtime folder", portable_folder, sizeof(portable_folder))) value.portable_root = portable_folder;
        route_changed |= ImGui::IsItemDeactivatedAfterEdit();
        char portable_device[4096]{};
        strncpy_s(portable_device, value.portable_device.c_str(), _TRUNCATE);
        if (ImGui::InputText("Portable device ID", portable_device, sizeof(portable_device))) value.portable_device = portable_device;
        route_changed |= ImGui::IsItemDeactivatedAfterEdit();
        route_changed |= ImGui::Checkbox("Allow devices without a validation entry", &value.portable_allow_unvalidated);
        if constexpr (sizeof(void*) == 4) {
            char native_folder[4096]{};
            strncpy_s(native_folder, value.native_root.c_str(), _TRUNCATE);
            if (ImGui::InputText("Native runtime folder", native_folder, sizeof(native_folder))) value.native_root = native_folder;
            route_changed |= ImGui::IsItemDeactivatedAfterEdit();
        }
        ImGui::TextWrapped("Configured folders take precedence. Relative runtime folders are resolved beside this addon; Windows and driver hashes are checked away from rendering.");
        ImGui::TextWrapped("x86 library: %s", value.native_library_reason.empty() ? "Package and settings eligible" : value.native_library_reason.c_str());
        ImGui::TextWrapped("x86 driver: %s", value.native_driver_reason.empty() ? "Package and settings eligible" : value.native_driver_reason.c_str());
        ImGui::TextWrapped("Portable: %s", value.portable_reason.empty() ? "Package and settings eligible" : value.portable_reason.c_str());
        for (const auto& error : value.failed_routes) ImGui::TextWrapped("Failed: %s", error.c_str());
        ImGui::TreePop();
    }
    const char* transfers[]{"Automatic", "Shared GPU textures (D3D11 only)", "Staging / CPU pixels"};
    int transfer = value.helper_transport == "shared" ? 1 : value.helper_transport == "readback" ? 2 : 0;
    ImGui::BeginDisabled(native_active(value));
    if (ImGui::Combo("Helper transfer", &transfer, transfers, 3)) {
        value.helper_transport = transfer == 1 ? "shared" : transfer == 2 ? "readback" : "auto";
        route_changed = true;
    }
    ImGui::EndDisabled();
    if (value.execution == ExecutionPreference::Portable) {
        char root[4096]{};
        strncpy_s(root, value.portable_root.c_str(), _TRUNCATE);
        if (ImGui::InputText("Portable runtime folder", root, sizeof(root))) value.portable_root = root;
        route_changed |= ImGui::IsItemDeactivatedAfterEdit();
        char device[128]{};
        strncpy_s(device, value.portable_device.c_str(), _TRUNCATE);
        if (ImGui::BeginCombo("Portable device", value.portable_device.c_str())) {
            for (const char* id : {"auto", "cpu"})
                if (ImGui::Selectable(id, value.portable_device == id)) { value.portable_device = id; route_changed = true; }
            for (const auto& item : value.execution_devices)
                if (item.id != "native" && item.id != "cpu" && item.id != "sycl:cpu" &&
                    ImGui::Selectable(item.label.c_str(), value.portable_device == item.id)) {
                    value.portable_device = item.id; route_changed = true;
                }
            ImGui::EndCombo();
        }
        if (ImGui::InputText("Portable device ID", device, sizeof(device))) value.portable_device = device;
        route_changed |= ImGui::IsItemDeactivatedAfterEdit();
        if (value.portable_device == "cpu") {
            bool automatic_threads = value.settings.cpu_threads == 0;
            const int available = int(phoenix_threads::available());
            if (ImGui::Checkbox("Automatic portable CPU threads", &automatic_threads))
                value.settings.cpu_threads = automatic_threads ? 0 : available;
            if (!automatic_threads)
                ImGui::SliderInt("Portable CPU threads", &value.settings.cpu_threads, 1, available,
                                 "%d", ImGuiSliderFlags_AlwaysClamp);
        }
        route_changed |= ImGui::Checkbox("Allow devices without a validation entry", &value.portable_allow_unvalidated);
        route_changed |= ImGui::Checkbox("Use helper if portable processing fails", &value.portable_fallback);
    }
    if(value.execution==ExecutionPreference::NativeLibrary||value.execution==ExecutionPreference::NativeDriver) {
        char root[4096]{};strncpy_s(root,value.native_root.c_str(),_TRUNCATE);
        if(ImGui::InputText("Native runtime folder",root,sizeof(root))) value.native_root=root;
        route_changed|=ImGui::IsItemDeactivatedAfterEdit();
        route_changed|=ImGui::Checkbox("Use helper if native processing fails",&value.native_fallback);
        if(value.execution==ExecutionPreference::NativeDriver&&needs_nvidia_dll(value.settings))
            ImGui::TextWrapped("NVIDIA DLSS5 sources require the x64 library or helper; the Phoenix DLSS5 driver preference is retained.");
    }
    ImGui::Text("Active execution: %s",execution_mode(value));
    if (!value.failed_routes.empty() && ImGui::Button("Retry preferred routes")) route_changed = true;
    if (route_changed) {
        reset_route_failures(value);
        value.processing_error.clear();
        ++value.selection_revision;
        value.restart = true;
        save(runtime, value, false);
    }
    if (!value.fallback_reason.empty()) ImGui::TextWrapped("Route: %s", value.fallback_reason.c_str());
    ImGui::Text("Game API: %s", api_name(runtime->get_device()->get_api()));
    ImGui::TextUnformatted(value.applied_settings.pipeline.motion_enabled?
        "Frame history: motion provider; unavailable guides reset recurrence":
        "Frame history: spatial reset (no engine motion vectors)");
    if (ImGui::Checkbox("Keep input and output from the same frame", &value.coherent_frames))
        reshade::set_config_value(runtime, "Phoenix.Compatibility", "CoherentFrames",
                                  value.coherent_frames);
    ImGui::TextWrapped(value.coherent_frames
                           ? "Show completed output only. Hold the last result while processing; show a black loading frame before the first result. Camera movement is delayed by processing time."
                           : "Experimental: apply the old model edit to the current image on "
                             "D3D9/D3D11. Moving objects can show through older masks.");
#ifndef _WIN64
    ImGui::TextUnformatted(value.local || resident_active(value) ? "Inference inside this 32-bit game process"
                                     : "32-bit game addon / graphics helper when required");
#endif
    auto *active_host = value.bridge     ? value.bridge->host()
                        : value.readback ? value.readback->host()
                                         : nullptr;
    if (active_host) {
        auto *host = active_host;
        ImGui::Text("Completed: %llu | processing round trip: %.2f ms",
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
            ImGui::TextWrapped("Processor stopped: %s", host->error().c_str());
    }
    }
}
inline void panel(reshade::api::effect_runtime* runtime) {
    std::lock_guard lock(sessions_mutex);
    auto& value = session(runtime);
    auto ui = panel_context(runtime, value);
    ui.setup_details = [runtime, &value] { draw_transport(runtime, value); };
    ui.retry = [runtime, &value] { save(runtime, value); };
    phoenix_ui::draw_controls(ui);
    apply_ui_edits(runtime, value, ui);
}

inline void destroy(reshade::api::effect_runtime *runtime) {
    std::lock_guard lock(sessions_mutex);
    sessions.erase(runtime);
}
inline void destroy_graphics_device(reshade::api::device *device) {
    std::lock_guard lock(sessions_mutex);
    sharing_devices.erase(device);
}
inline bool is_software_render_device(reshade::api::device* device) {
    // The render adapter and inference adapter need not be the same for CPU
    // transport. WARP cannot execute the native CUDA/NVAPI network or share its
    // textures with a hardware helper; detect it before attempting either route.
    const auto api = device->get_api();
    if (api != reshade::api::device_api::d3d10 && api != reshade::api::device_api::d3d11 &&
        api != reshade::api::device_api::d3d12) return false;
    auto* native = reinterpret_cast<IUnknown*>(uintptr_t(device->get_native()));
    if (!native) return false;
    Ptr<IDXGIAdapter> adapter;
    Ptr<IDXGIDevice> dxgi;
    if (SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&dxgi)))) dxgi->GetAdapter(&adapter);
    else if (api == reshade::api::device_api::d3d12) {
        Ptr<ID3D12Device> d3d12;
        Ptr<IDXGIFactory4> factory;
        if (SUCCEEDED(native->QueryInterface(IID_PPV_ARGS(&d3d12))) &&
            SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            factory->EnumAdapterByLuid(d3d12->GetAdapterLuid(), IID_PPV_ARGS(&adapter));
    }
    Ptr<IDXGIAdapter1> adapter1;
    DXGI_ADAPTER_DESC1 description{};
    return adapter && SUCCEEDED(adapter.As(&adapter1)) &&
        SUCCEEDED(adapter1->GetDesc1(&description)) &&
        (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
}
inline void process(reshade::api::effect_runtime *runtime, reshade::api::command_list *command,
                    reshade::api::resource back_resource) {
    if (!applies(runtime))
        return;
    std::lock_guard lock(sessions_mutex);
    auto &value = session(runtime);
    poll_control(runtime, value);
    value.displayed = false;
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
                apply_live_selection(value.applied_settings, value.settings);
                ++value.selection_revision;
                value.restart = true;
                value.processing_error.clear(); reset_route_failures(value);
                value.settings_dirty = settings_json(value.settings) != settings_json(value.applied_settings);
                save(runtime, value, false);
            }
            if (index == 1)
                send_action(value, phoenix_ui::Action::ResetHistory);
            if (index == 2) {
                ++value.markers;
                send_action(value, phoenix_ui::Action::Marker);
            }
            if (index == 3 && value.applied_settings.experimental) {
                value.applied_settings.transformer_blocks = value.applied_settings.transformer_blocks == 1
                                                        ? 8
                                                        : value.applied_settings.transformer_blocks - 1;
                value.applied_settings.minimum_blocks =
                    std::min(value.applied_settings.minimum_blocks, value.applied_settings.transformer_blocks);
                value.settings.transformer_blocks = value.applied_settings.transformer_blocks;
                value.settings.minimum_blocks = value.applied_settings.minimum_blocks;
                ++value.selection_revision; value.restart = true;
                value.settings_dirty = settings_json(value.settings) != settings_json(value.applied_settings);
                save(runtime, value, false);
            }
            if (index == 4)
                send_action(value, phoenix_ui::Action::ClearTiming);
            if (index == 6) send_action(value, phoenix_ui::Action::SaveScreenshotPair);
            if (index == 5)
                send_action(value, phoenix_ui::Action::CaptureInputs,
                            value.settings.capture_burst ? value.settings.capture_burst : 8);
        }
    // A failed Vulkan recording still belongs to ReShade's pending Present;
    // finish_present submits the application's waits before we retire it.
    if (value.deferred_destroy) return;
    if (in_process_gpu_uncertain || (value.local && !value.local->safe_to_fallback())) {
        value.message = "Previous in-process GPU work did not retire; restart the game before processing again.";
        return;
    }
    auto *device = runtime->get_device();
    const auto description = device->get_resource_desc(back_resource);
    BackbufferCapabilities detected;
    detected.api = device->get_api(); detected.width = description.texture.width;
    if (!value.software_device) value.software_device = is_software_render_device(device);
    detected.software_device = *value.software_device;
    detected.height = description.texture.height; detected.format = description.texture.format;
    detected.samples = description.texture.samples; detected.layers = description.texture.depth_or_layers;
    detected.levels = description.texture.levels;
    detected.sdr = sdr_backbuffer(value.color_space, detected.format);
    if (detected.api == reshade::api::device_api::d3d12 && back_resource.handle) {
        Ptr<ID3D12Resource1> protected_image;
        auto* image = reinterpret_cast<ID3D12Resource*>(uintptr_t(back_resource.handle));
        if (SUCCEEDED(image->QueryInterface(IID_PPV_ARGS(&protected_image)))) {
            Ptr<IUnknown> protection;
            detected.protected_resource = SUCCEEDED(protected_image->GetProtectedResourceSession(IID_PPV_ARGS(&protection))) && protection;
        }
    }
    value.capabilities = transport_capabilities(value, value.applied_settings, detected);
    const auto& processing=value.applied_settings.processing;
    if(processing.enabled && processing.hook!=0 && processing.hook!=3) {
        value.message="This source is a final frame; select Present for explicit processing.";return;
    }
    if(processing.enabled && processing.hook==3 && processing.reuse_guides==2) {
        value.message="Present requires current DLSS motion and depth guides; this final-frame source has none.";return;
    }
    if (!value.enabled || (processing.enabled && processing.hook==0)) {
        value.presented_revision = value.selection_revision;
        if (value.local) value.local->reset_history();
        else { remember_timing(value); value.bridge.reset(); value.readback.reset(); }
        value.message = !value.processing_error.empty() ? value.processing_error :
            value.local ? "Phoenix DLSS5 disabled; in-process resources are kept warm." : "Phoenix DLSS5 disabled.";
        return;
    }
    if (!value.processing_error.empty()) {
        // Preserve the user's On selection. A failure suspends work until a
        // new selection/retry resumes it, without silently saving Off to disk.
        value.message = value.processing_error;
        return;
    }
    if (value.applied_settings.mode == phoenix_addon::phoenix::input) {
        value.presented_revision = value.selection_revision;
        if (value.local) value.local->reset_history();
        else { remember_timing(value); value.bridge.reset(); value.readback.reset(); }
        value.message = "Original game (model bypass).";
        return;
    }
    auto capabilities = value.capabilities;
    auto decision = choose_backbuffer_route_with_fallback(capabilities, value.execution,
                                                        value.native_fallback, value.portable_fallback);
    if ((value.execution == ExecutionPreference::NativeLibrary && value.native_library_failed && value.native_fallback) ||
        (value.execution == ExecutionPreference::NativeDriver && value.native_driver_failed && value.native_fallback) ||
        (value.execution == ExecutionPreference::Portable && value.portable_failed && value.portable_fallback))
        decision = choose_backbuffer_route(capabilities, ExecutionPreference::Automatic);
    const bool native_attempt = decision.route == BackbufferRoute::NativeLibrary || decision.route == BackbufferRoute::NativeDriver;
    const bool portable_attempt = decision.route == BackbufferRoute::Portable;
    if (value.helper_transport != "auto" && value.helper_transport != "shared" &&
        value.helper_transport != "readback") {
        value.message = "Unknown HelperTransport; use auto, shared or readback.";
        value.processing_error = value.message;
        return;
    }
    if (value.local_failed && decision.route == BackbufferRoute::InProcess) {
        capabilities.local_api = false;
        decision = choose_backbuffer_route(capabilities, value.execution);
        decision.reason = value.fallback_reason;
    }
    if (decision.route == BackbufferRoute::Unsupported) {
        value.message = decision.reason; value.fallback_reason = decision.reason;
        if(terminal_route_refusal(capabilities,decision)) value.processing_error=value.message;
        return;
    }
    if (decision.route == BackbufferRoute::InProcess) {
        try {
            if (value.bridge || value.readback) {
                remember_timing(value); value.bridge.reset(); value.readback.reset();
            }
            if (value.local && !value.local->matches(description)) {
                remember_timing(value); value.local.reset(); value.logged_outputs = 0;
            }
            if (!value.local) {
                value.local = std::make_unique<D3D12Bridge>(runtime, description);
                value.width = description.texture.width; value.height = description.texture.height;
                value.readback_format = description.texture.format;
                value.message = decision.reason;
                value.fallback_reason.clear();
                reshade::log::message(reshade::log::level::info, value.message.c_str());
            }
            const bool apply = value.restart;
            if (apply) value.local->reset_history();
            value.restart = false;
            prepare_pair_request(value,value.local.get());
            const bool composed=value.local->frame(back_resource, value.applied_settings, apply, value.coherent_frames);
            update_pair_result(value,value.local->screenshot_pair_status());
            if (composed) {
                value.local_composed_at_frame = value.frames;
                if (!value.local->held_last_frame()) {
                    value.local_composed_frame = value.frames;
                    value.presented_revision = value.selection_revision;
                    value.displayed_settings = value.applied_settings;
                }
                value.displayed = true;
                ++value.mode_own[size_t(value.applied_settings.mode)];
                value.message = value.local->held_last_frame() ? "Holding the last completed Phoenix frame while inference is deferred." : decision.reason;
            } else value.message = "In-process Phoenix DLSS5 waiting for a usable output.";
            return;
        } catch (const std::exception& error) {
            value.fallback_reason = std::string("In-process initialization/evaluation: ") + error.what();
            value.message = value.fallback_reason;
            const bool safe = !value.local || value.local->safe_to_fallback();
            if (!may_fallback_to_helper(value.execution, safe) || capabilities.width > 8192 || capabilities.height > 8192) {
                value.processing_error = value.message;
                return;
            }
            value.local.reset(); value.logged_outputs = 0;
            reject_route(value, BackbufferRoute::InProcess, value.fallback_reason);
            reshade::log::message(reshade::log::level::warning, value.fallback_reason.c_str());
            return; // Re-evaluate the ordered candidates on the next frame.
        }
    } else {
        if (value.local) { remember_timing(value); value.local.reset(); value.logged_outputs = 0; }
        if (value.failed_routes.empty()) value.fallback_reason = decision.reason;
    }
    if (value.restart) {
        remember_timing(value);
        value.bridge.reset();
        value.readback.reset();
        value.shared_unavailable = false;
        value.restart = false;
        value.logged_outputs = 0;
    }
    if (decision.route == BackbufferRoute::Helper && value.helper_transport == "shared" &&
        (device->get_api() != reshade::api::device_api::d3d11 || value.shared_unavailable || capabilities.software_device)) {
        value.message = "Shared helper transfer requires working D3D11 shared textures and fences. Select staging or automatic.";
        value.processing_error = value.message;
        return;
    }
    if (portable_attempt || native_attempt || value.helper_transport == "readback" ||
        device->get_api() != reshade::api::device_api::d3d11 || value.shared_unavailable || capabilities.software_device) {
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
                    "No model selected. Browse to your NVIDIA DLSS5 DLL or a compatible model folder.");
                std::optional<PortableOptions> portable;
                std::optional<NativeOptions> native;
                if(native_attempt) {
                    native = NativeOptions{runtime_path(value.native_root, "PhoenixRuntime/wow64"),
                                           decision.route == BackbufferRoute::NativeDriver};
                }
                if (portable_attempt) {
                    portable = PortableOptions{runtime_path(value.portable_root,
                        sizeof(void*) == 4 ? "PhoenixRuntime/win-x86" : "PhoenixRuntime/win-x64"),
                        value.portable_device, value.portable_allow_unvalidated};
                }
                ++value.helper_generation;
                value.readback = std::make_unique<ReadbackBridge>(runtime, description, module,
                                                                  selected, value.applied_settings, portable,native);
                value.message = std::string(api_name(device->get_api())) +
                                (portable_attempt ? " readback / in-process portable Phoenix DLSS5. CPU pixel transfers add " :
                                " readback / asynchronous 64-bit helper. CPU pixel transfers add ") +
                                "latency and bandwidth.";
                if(native_attempt) value.message=decision.reason+" Fenced CPU capture/return.";
                reshade::log::message(reshade::log::level::info, value.message.c_str());
            }
            if (portable_attempt && value.readback->host()->phase() == Phase::Failed &&
                (value.portable_fallback || value.execution == ExecutionPreference::Automatic)) {
                reject_route(value, decision.route, value.readback->host()->error());
                return; // Retire any queued Vulkan presentation before changing the transport.
            }
            if(native_attempt && value.readback->host()->phase() == Phase::Failed &&
               (value.native_fallback || value.execution == ExecutionPreference::Automatic)) {
                reject_route(value, decision.route, value.readback->host()->error()); return;
            }
            require(value.readback->host()->update_tuning(value.applied_settings), "Helper refused live tuning");
            prepare_pair_request(value,value.readback.get());
            value.readback->frame(command, back, value.frames, value.applied_settings,
                                  value.coherent_frames);
            update_pair_result(value,value.readback->screenshot_pair_status());
            log_progress(value);
        } catch (const std::exception &error) {
            value.message = error.what();
            if ((native_attempt && (value.native_fallback || value.execution == ExecutionPreference::Automatic)) ||
                (portable_attempt && (value.portable_fallback || value.execution == ExecutionPreference::Automatic))) {
                reject_route(value, decision.route, value.message);
            } else value.processing_error = value.message;
            reshade::log::message(reshade::log::level::error, value.message.c_str());
            if (device->get_api() == reshade::api::device_api::vulkan)
                value.deferred_destroy = true;
            else value.readback.reset();
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
                    "No model selected. Browse to your NVIDIA DLSS5 DLL or a compatible model folder.");
            auto *native_device = reinterpret_cast<ID3D11Device *>(uintptr_t(device->get_native()));
            ++value.helper_generation;
            auto &sharing = sharing_devices[device];
            value.bridge = std::make_unique<D3D11Bridge>(native_device, value.width, value.height,
                                                         value.format, sharing.Get());
            if (!sharing) sharing = value.bridge->sharing_device();
            value.bridge->start(module, selected, value.applied_settings);
            value.message = "D3D11 shared GPU textures / asynchronous 64-bit D3D12 helper (SDR).";
        }
        require(value.bridge->host()->update_tuning(value.applied_settings), "Helper refused live tuning");
        prepare_pair_request(value,value.bridge.get());
        value.bridge->frame(back.Get(), value.frames, value.applied_settings,
                            value.coherent_frames);
        update_pair_result(value,value.bridge->screenshot_pair_status());
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
        auto& value = session(runtime);
        // Publish the control endpoint before the first usable presentation.
        // Otherwise queue/backbuffer rejection also hides its diagnostic status.
        poll_control(runtime, value);
    }
}
inline bool prepare_swapchain(reshade::api::device_api api, reshade::api::swapchain_desc& description, void*) {
    if (api != reshade::api::device_api::vulkan) return false;
    // Returning processed pixels uses a transfer into the presentation image.
    // Request that usage at creation; recording a copy cannot add it later.
    const auto transfers = reshade::api::resource_usage::copy_source | reshade::api::resource_usage::copy_dest;
    if ((description.back_buffer.usage & transfers) == transfers) return false;
    description.back_buffer.usage |= transfers;
    return true;
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
            runtime->get_hwnd() != swapchain->get_hwnd())
            continue;
        if (runtime->get_current_back_buffer().handle != back.handle) {
            value->message = "Presentation and runtime backbuffers differ; waiting for a matching image.";
            poll_control(runtime, *value);
            continue;
        }
        const auto presentation_format = queue->get_device()->get_resource_desc(back).texture.format;
        value->color_space = swapchain->get_color_space();
        if ((presentation_format == reshade::api::format::r10g10b10a2_unorm ||
             presentation_format == reshade::api::format::r10g10b10a2_typeless) &&
            swapchain->get_color_space() != reshade::api::color_space::srgb_nonlinear) {
            value->message = "RGB10A2 final-frame processing requires SDR sRGB; HDR output is not supported by this bridge.";
            poll_control(runtime, *value);
            return;
        }
        auto* processing_queue = runtime->get_command_queue();
        if (processing_queue != queue && queue->get_device()->get_api() != reshade::api::device_api::vulkan) {
            value->message =
                "Presentation and graphics queues differ; resource ownership is not proven.";
            poll_control(runtime, *value);
            return;
        }
        // Vulkan can present on a secondary/compute queue. ReShade 6.8 locks
        // both queues, waits for the application's presentation semaphores,
        // then flushes the runtime graphics queue before presentation. Record
        // there and retire its work in finish_present, never flush it early.
        // See crosire/reshade v6.8.0 vulkan_hooks_swapchain.cpp::vkQueuePresentKHR.
        auto* commands = processing_queue ? processing_queue->get_immediate_command_list() : nullptr;
        if (!commands) {
            value->message = "The runtime has no graphics command list for image transport.";
            poll_control(runtime, *value);
            return;
        }
        // Present is independent of ReShade shader loading. Process before its
        // effects and overlay, so neither a dummy shader nor UI filtering is needed.
        const auto previous_presented = presented(*value);
        process(runtime, commands, back);
        // All processing exits converge here, including initialization errors,
        // required-guide refusals and route changes. A skipped inference must
        // never silently become an unprocessed display frame in coherent mode.
        const auto& settings = value->applied_settings;
        const bool processing_requested = value->enabled && value->coherent_frames &&
            settings.mode != phoenix_addon::phoenix::input &&
            (!settings.processing.enabled || settings.processing.hook != 0);
        const bool composed = value->local ? value->local_composed_at_frame == value->frames :
            presented(*value) > previous_presented;
        if (processing_requested && !composed && queue->get_device()->get_api() == reshade::api::device_api::d3d12) {
            bool held = false;
            if (value->local && value->local->safe_to_fallback() &&
                value->local->matches(queue->get_device()->get_resource_desc(back)))
                held = value->local->protect_presentation(back);
            if (held) {
                ++value->presentation_holds;
                value->local_composed_at_frame = value->frames;
                value->message += " Holding completed output.";
            } else {
                CompletedPresentation::record_loading(
                    reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(commands->get_native())),
                    reinterpret_cast<ID3D12Resource*>(uintptr_t(back.handle)));
                ++value->loading_presentations;
                value->message += " Waiting for processed output; raw game image hidden.";
            }
        }
        record_screenshot(runtime, *value, commands, back);
        return;
    }
}
inline void finish_present(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain) {
    if (!queue || !swapchain) return;
    std::lock_guard lock(sessions_mutex);
    const auto back = swapchain->get_current_back_buffer();
    for (auto &[runtime, value] : sessions) {
        if (runtime->get_device() != queue->get_device() || runtime->get_hwnd() != swapchain->get_hwnd()) continue;
        if (runtime->get_command_queue() != queue && queue->get_device()->get_api() != reshade::api::device_api::vulkan) continue;
        if (value->readback && queue->get_device()->get_api() == reshade::api::device_api::vulkan) {
            try {
                value->readback->signal_after_present();
                if (value->deferred_destroy) {
                    value->readback.reset();
                    value->deferred_destroy = false;
                }
            } catch (const std::exception &error) {
                value->message = error.what();
                value->processing_error = value->message;
                reshade::log::message(reshade::log::level::error, value->message.c_str());
                value->readback.reset();
                value->deferred_destroy = false;
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
    reshade::register_event<reshade::addon_event::create_swapchain>(prepare_swapchain);
    reshade::register_event<reshade::addon_event::init_effect_runtime>(initialize);
    reshade::register_event<reshade::addon_event::present>(present);
    reshade::register_event<reshade::addon_event::finish_present>(finish_present);
    reshade::register_event<reshade::addon_event::destroy_effect_runtime>(destroy);
    reshade::register_event<reshade::addon_event::destroy_device>(destroy_graphics_device);
}
inline void detach() {
    reshade::unregister_event<reshade::addon_event::create_swapchain>(prepare_swapchain);
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
        const auto processes=inference_processes(*value);
        auto *host = value->bridge     ? value->bridge->host()
                     : value->readback ? value->readback->host()
                                       : nullptr;
        result.push_back(
            {{"api", unsigned(runtime->get_device()->get_api())},
             {"window_id", uint64_t(reinterpret_cast<uintptr_t>(runtime->get_hwnd()))},
             {"api_name", api_name(runtime->get_device()->get_api())},
             {"enabled", value->enabled},
             {"execution_mode", execution_mode(*value)},
             {"execution_preference", value->execution_name}, {"fallback_reason", value->fallback_reason},
             {"helper_pid", host && !resident_active(*value) ? host->process_id() : 0},
             {"processing_host_pid",processes.at("processing_host_pid")},
             {"inference_pid",processes.at("inference_pid")},{"inference_pids",processes.at("inference_pids")},
             {"timing_provenance", timing_provenance(*value)},
             {"control_folder", phoenix_model::source_path_string(value->control.folder())},
             {"settings", settings_json(value->settings)},
             {"applied_settings", settings_json(value->applied_settings)},
             {"width", value->width},
             {"height", value->height},
             {"message", value->message},
             {"phase", unsigned(processing_phase(*value))},
             {"delivered", delivered(*value)}, {"presented", presented(*value)},
             {"composition",
              value->local ? "same_frame_gpu" : value->coherent_frames ? "matched_frame_hold"
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
