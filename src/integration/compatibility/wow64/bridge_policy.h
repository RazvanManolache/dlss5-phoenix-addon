#pragma once
#include "../session_policy.h"
#include <filesystem>
namespace phoenix_compat {
struct NativeOptions { std::filesystem::path root; bool driver_host=false; };
inline const char* native_settings_unavailable_reason(const Settings& settings, bool driver_host,
                                                     bool library_frame_services=false, bool library_appearance=false) {
    if (settings.native_kernel_style != "auto" && phoenix_addon::phoenix::uses_ours(settings.mode))
        return "Selectable kernel styles require the D3D12 addon or x64 graphics helper; use Automatic style for the experimental x86 native bridges.";
    if (settings.execution_device != "native")
        return "Native x86 mechanisms require the Native RTX executor; use portable or helper processing for another executor.";
    if (settings.resolution_mode != 0 || settings.host_unjitter || settings.automatic_budget)
        return "Experimental in-process x86 requires normal resolution, no unjitter and fixed budget; use the helper for these options.";
    if(settings.pipeline.chain_enabled)
        return "The experimental x86 native bridges do not execute custom processing chains; use the x64 addon, portable or helper route.";
    if(settings.pipeline.priority!=0)
        return "Adaptive pipeline priority is not qualified for the x86 library bridge; use the helper.";
    if(settings.native_options && settings.ui_correction!=0)
        return "UI correction requires the helper; this x86 library packet has no UI guide.";
    if((settings.controls.appearance || (settings.native_options &&
       (settings.intensity!=1.f||settings.style!=0))) && (driver_host || !library_appearance))
        return "The selected appearance/composition controls require the helper; the x86 bridge supports strength and motion controls.";
    if((settings.pipeline.motion_enabled||settings.pipeline.scene_detection||settings.pipeline.monitoring)&&
       (driver_host||!library_frame_services))
        return "Requested motion/scene/frame services need a matching x86 library-bridge runtime; this route/package cannot provide them.";
    if((settings.pipeline.motion_enabled||settings.pipeline.scene_detection)&&settings.refinement_passes!=1)
        return "Temporal x86 library processing currently requires one network pass per captured frame; use the helper for refinements.";
    if(needs_nvidia_dll(settings))
        return "NVIDIA DLSS5/comparison requires the x64 helper in a 32-bit game; the NVIDIA x64 DLL is not qualified for in-process WOW64 execution. Phoenix supports in-process execution.";
    if(driver_host) {
        if (settings.blocks_for(settings.mode) != 8 || !std::all_of(settings.block_repeats.begin(), settings.block_repeats.end(), [](int count) { return count == 1; }))
            return "Experimental x86 driver host currently requires all eight blocks once; use the library route or helper for block overrides.";
    }
    return "";
}
inline void validate_native_settings(const Settings& settings, bool driver_host, bool library_frame_services=false, bool library_appearance=false) {
    const auto reason = native_settings_unavailable_reason(settings, driver_host,library_frame_services,library_appearance);
    phoenix_addon::check(!*reason, reason);
}
}
