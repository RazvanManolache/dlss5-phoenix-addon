#pragma once
#include "../session_policy.h"
#include <filesystem>
namespace phoenix_compat {
struct NativeOptions { std::filesystem::path root; bool driver_host=false; };
inline const char* native_settings_unavailable_reason(const Settings& settings, bool driver_host) {
    if (settings.execution_device != "native")
        return "Native x86 mechanisms require the Native RTX executor; use portable or helper processing for another executor.";
    if (settings.resolution_mode != 0 || settings.host_unjitter || settings.automatic_budget)
        return "Experimental in-process x86 requires normal resolution, no unjitter and fixed budget; use the helper for these options.";
    if(driver_host) {
        if (settings.model_id == "small-fp8")
            return "The experimental x86 driver host uses the Original launch ABI; select the x64-library or helper route for Lightweight FP8.";
        if (needs_nvidia_dll(settings))
            return "The x86 driver host supports Phoenix DLSS5 only; NVIDIA DLSS5 requires the x64-library route or helper.";
        if (settings.blocks_for(settings.mode) != 8 || !std::all_of(settings.block_repeats.begin(), settings.block_repeats.end(), [](int count) { return count == 1; }))
            return "Experimental x86 driver host currently requires all eight blocks once; use the library route or helper for block overrides.";
    }
    return "";
}
inline void validate_native_settings(const Settings& settings, bool driver_host) {
    const auto reason = native_settings_unavailable_reason(settings, driver_host);
    phoenix_addon::check(!*reason, reason);
}
}
