#pragma once
#include "pixel_transport.h"
#include <string>
#include <vector>

namespace phoenix_compat {
enum class ExecutionPreference { Automatic, InProcess, Helper, Portable, NativeLibrary, NativeDriver, Invalid };
enum class BackbufferRoute { InProcess, Helper, Portable, NativeLibrary, NativeDriver, Unsupported };
inline ExecutionPreference execution_preference(const std::string& value) {
    if (value == "auto") return ExecutionPreference::Automatic;
    if (value == "in_process") return ExecutionPreference::InProcess;
    if (value == "helper") return ExecutionPreference::Helper;
    if (value == "portable") return ExecutionPreference::Portable;
    if (value == "wow64_library") return ExecutionPreference::NativeLibrary;
    if (value == "wow64_driver") return ExecutionPreference::NativeDriver;
    return ExecutionPreference::Invalid;
}
struct BackbufferCapabilities {
    reshade::api::device_api api = reshade::api::device_api::d3d12;
    unsigned bits = sizeof(void*) * 8, width = 0, height = 0, samples = 1, layers = 1, levels = 1;
    bool local_api = false, native_executor = true, nvidia_required = false;
    bool native_library_available = false, native_driver_available = false, portable_available = false;
    bool discovery_pending = false;
    bool sdr = true, queue_matches = true, protected_resource = false;
    bool software_device = false;
    reshade::api::format format = reshade::api::format::unknown;
};
struct BackbufferDecision {
    BackbufferRoute route;
    std::string reason;
};
inline bool local_color_format(reshade::api::format format) {
    using F = reshade::api::format;
    return format == F::r8g8b8a8_unorm || format == F::r8g8b8a8_unorm_srgb ||
           format == F::r8g8b8a8_typeless || format == F::b8g8r8a8_unorm ||
           format == F::b8g8r8a8_unorm_srgb || format == F::b8g8r8a8_typeless ||
           format == F::r10g10b10a2_unorm || format == F::r10g10b10a2_typeless;
}
inline bool sdr_backbuffer(reshade::api::color_space space,reshade::api::format format) {
    using C=reshade::api::color_space;
    using F=reshade::api::format;
    if(space==C::srgb_nonlinear)return true;
    if(space!=C::unknown)return false;
    // Legacy ReShade swapchains do not expose color-space metadata. Infer SDR
    // only for ordinary eight-bit canvases, never float/10-bit or explicit HDR.
    return format==F::r8g8b8a8_unorm||format==F::r8g8b8a8_unorm_srgb||format==F::r8g8b8a8_typeless||
        format==F::r8g8b8x8_unorm||format==F::r8g8b8x8_unorm_srgb||
        format==F::b8g8r8a8_unorm||format==F::b8g8r8a8_unorm_srgb||format==F::b8g8r8a8_typeless||
        format==F::b8g8r8x8_unorm||format==F::b8g8r8x8_unorm_srgb||format==F::b8g8r8x8_typeless;
}
inline BackbufferDecision choose_backbuffer_route(const BackbufferCapabilities& caps,
                                                 ExecutionPreference preference) {
    if (preference == ExecutionPreference::Invalid)
        return {BackbufferRoute::Unsupported, "Unknown graphics ExecutionMode; use auto, in_process, portable, helper, wow64_library or wow64_driver."};
    if (caps.bits != 32 && caps.bits != 64)
        return {BackbufferRoute::Unsupported, "Unknown process architecture."};
    if (!caps.width || !caps.height || caps.width > 16384 || caps.height > 16384)
        return {BackbufferRoute::Unsupported, "Backbuffer extent is outside the supported range."};
    if (caps.protected_resource)
        return {BackbufferRoute::Unsupported, "Protected presentation cannot be captured."};
    if (!caps.queue_matches)
        return {BackbufferRoute::Unsupported, "Presentation queue ownership is not proven."};
    if (caps.api == reshade::api::device_api::d3d12 &&
        (!caps.sdr || !local_color_format(caps.format) || caps.samples != 1 ||
         caps.layers != 1 || caps.levels != 1))
        return {BackbufferRoute::Unsupported, "Standalone DX12 requires a single-sample SDR RGB8/RGB10A2 image; use engine input for other layouts."};
    if (preference == ExecutionPreference::Automatic) {
        // Keep frame transport inside the game whenever a compatible route is
        // installed. Rejected/failed routes are removed from the capabilities
        // by the session; the helper is the final candidate, not the first.
        for (auto candidate : {ExecutionPreference::InProcess, ExecutionPreference::NativeLibrary,
                               ExecutionPreference::NativeDriver, ExecutionPreference::Portable}) {
            auto decision = choose_backbuffer_route(caps, candidate);
            if (decision.route != BackbufferRoute::Unsupported && decision.route != BackbufferRoute::Helper)
                return decision;
        }
        if (caps.discovery_pending)
            return {BackbufferRoute::Unsupported, "Checking installed same-process runtimes..."};
        auto decision = choose_backbuffer_route(caps, ExecutionPreference::Helper);
        if (decision.route == BackbufferRoute::Helper)
            decision.reason = "No eligible same-process route remains; using the x64 helper.";
        return decision;
    }
    if(preference==ExecutionPreference::NativeLibrary||preference==ExecutionPreference::NativeDriver) {
        if(caps.bits!=32) return {BackbufferRoute::Unsupported,"WOW64 execution is only available in the x86 addon."};
        // Both mixed-bitness native routes can crash the 32-bit NVIDIA OpenGL
        // driver when the renderer is disabled. Keep this unqualified pairing
        // out of automatic selection and explicit requests.
        if(caps.api==reshade::api::device_api::opengl)
            return {BackbufferRoute::Unsupported,"Cross-bitness native execution is unavailable with x86 OpenGL; use a matching portable runtime or the external helper."};
        if (!caps.native_executor)
            return {BackbufferRoute::Unsupported, "Native cross-bitness routes require the Native RTX executor."};
        if (!(preference == ExecutionPreference::NativeLibrary ? caps.native_library_available : caps.native_driver_available))
            return {BackbufferRoute::Unsupported, "The selected x86 runtime is missing, incompatible with this setup, or failed. Check transport availability."};
        if(caps.width>8192||caps.height>8192||!caps.sdr||caps.layers!=1||caps.levels!=1||
            (caps.samples!=1&&caps.api!=reshade::api::device_api::d3d9))
            return {BackbufferRoute::Unsupported,"Experimental native x86 processing requires one SDR canvas up to 8192 per axis."};
        if(preference==ExecutionPreference::NativeDriver&&caps.nvidia_required)
            return {BackbufferRoute::Helper,"NVIDIA/comparison needs its x64 library; retaining the Phoenix DLSS5 driver-host preference and using the helper."};
        return {preference==ExecutionPreference::NativeLibrary?BackbufferRoute::NativeLibrary:BackbufferRoute::NativeDriver,
            preference==ExecutionPreference::NativeLibrary?"Experimental x64 renderer inside this x86 process.":"Experimental x86 Phoenix DLSS5 host with same-process native driver."};
    }
    if (preference == ExecutionPreference::Portable && !caps.nvidia_required) {
        if (!caps.portable_available)
            return {BackbufferRoute::Unsupported, "A matching portable runtime/device is unavailable or failed for this setup."};
        if (caps.width > 8192 || caps.height > 8192 || !caps.sdr ||
            caps.layers != 1 || caps.levels != 1 ||
            (caps.samples != 1 && caps.api != reshade::api::device_api::d3d9))
            return {BackbufferRoute::Unsupported, "Portable processing requires a single SDR canvas up to 8192 per axis."};
        return {BackbufferRoute::Portable, "In-process portable Phoenix DLSS5; fenced CPU pixel transfers, no helper process."};
    }
    std::string reason;
    if (caps.bits == 32 && caps.nvidia_required)
        reason = "The selected NVIDIA DLSS5 DLL requires the 64-bit helper; Phoenix DLSS5-only processing does not.";
    else if (caps.bits == 32)
        reason = "Same-device native D3D12 requires the x64 addon. Use an available x86 same-process route or Automatic.";
    else if (caps.api != reshade::api::device_api::d3d12)
        reason = "This graphics API currently needs the D3D12 compatibility bridge.";
    else if (caps.software_device) reason = "Software render device requires staging to a hardware inference helper.";
    else if (!caps.local_api) reason = "This add-on does not expose the in-process backbuffer API.";
    else if (!caps.native_executor) reason = "The selected executor currently uses the isolated helper.";
    if (preference == ExecutionPreference::Helper) reason = "External helper explicitly selected.";
    if (preference == ExecutionPreference::Portable)
        reason = "NVIDIA DLSS5 DLL/comparison selected; using the helper while retaining the portable Phoenix DLSS5 preference.";
    if (reason.empty()) return {BackbufferRoute::InProcess, "DX12 same-device GPU processing; no helper process."};
    if (preference == ExecutionPreference::InProcess)
        return {BackbufferRoute::Unsupported, "In-process mode cannot be used: " + reason};
    if (caps.api == reshade::api::device_api::d3d12 && (caps.width > 8192 || caps.height > 8192))
        return {BackbufferRoute::Unsupported, "The helper cannot transport this DX12 extent; no automatic downscale is applied."};
    return {BackbufferRoute::Helper, reason};
}
inline bool may_fallback_to_helper(ExecutionPreference preference, bool submission_retired) {
    return preference == ExecutionPreference::Automatic && submission_retired;
}
inline const char* execution_preference_name(ExecutionPreference preference) {
    switch (preference) {
    case ExecutionPreference::Automatic: return "auto";
    case ExecutionPreference::InProcess: return "in_process";
    case ExecutionPreference::NativeLibrary: return "wow64_library";
    case ExecutionPreference::NativeDriver: return "wow64_driver";
    case ExecutionPreference::Portable: return "portable";
    case ExecutionPreference::Helper: return "helper";
    default: return "invalid";
    }
}
inline const char* execution_preference_label(ExecutionPreference preference) {
    switch (preference) {
    case ExecutionPreference::Automatic: return "Automatic (same-process first)";
    case ExecutionPreference::InProcess: return "In-process native D3D12";
    case ExecutionPreference::NativeLibrary: return "x64 library inside x86 (experimental)";
    case ExecutionPreference::NativeDriver: return "x86 Phoenix DLSS5 + native driver (experimental)";
    case ExecutionPreference::Portable: return "In-process portable (experimental)";
    case ExecutionPreference::Helper: return "External x64 helper";
    default: return "Unknown saved mechanism";
    }
}
inline std::vector<ExecutionPreference> available_execution_preferences(const BackbufferCapabilities& caps) {
    std::vector<ExecutionPreference> choices{ExecutionPreference::Automatic};
    for (auto preference : {ExecutionPreference::InProcess, ExecutionPreference::NativeLibrary,
                            ExecutionPreference::NativeDriver, ExecutionPreference::Portable,
                            ExecutionPreference::Helper}) {
        const auto route = choose_backbuffer_route(caps, preference).route;
        if (route != BackbufferRoute::Unsupported &&
            (route != BackbufferRoute::Helper || preference == ExecutionPreference::Helper))
            choices.push_back(preference);
    }
    return choices;
}
} // namespace phoenix_compat
