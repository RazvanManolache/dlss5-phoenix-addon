#pragma once
#include "pixel_transport.h"
#include <string>

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
    bool x86_native_validated = false;
    bool sdr = true, queue_matches = true, protected_resource = false;
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
    if(preference==ExecutionPreference::NativeLibrary||preference==ExecutionPreference::NativeDriver) {
        if(caps.bits!=32) return {BackbufferRoute::Unsupported,"WOW64 execution is only available in the x86 addon."};
        if(caps.width>8192||caps.height>8192||!caps.sdr||caps.layers!=1||caps.levels!=1||
            (caps.samples!=1&&caps.api!=reshade::api::device_api::d3d9))
            return {BackbufferRoute::Unsupported,"Experimental native x86 processing requires one SDR canvas up to 8192 per axis."};
        if(preference==ExecutionPreference::NativeDriver&&caps.nvidia_required)
            return {BackbufferRoute::Helper,"NVIDIA/comparison needs its x64 library; retaining the Phoenix DLSS5 driver-host preference and using the helper."};
        return {preference==ExecutionPreference::NativeLibrary?BackbufferRoute::NativeLibrary:BackbufferRoute::NativeDriver,
            preference==ExecutionPreference::NativeLibrary?"Experimental x64 renderer inside this x86 process.":"Experimental x86 Phoenix DLSS5 host with same-process native driver."};
    }
    if (preference == ExecutionPreference::Portable && !caps.nvidia_required) {
        if (caps.width > 8192 || caps.height > 8192 || !caps.sdr ||
            caps.layers != 1 || caps.levels != 1 ||
            (caps.samples != 1 && caps.api != reshade::api::device_api::d3d9))
            return {BackbufferRoute::Unsupported, "Portable processing requires a single SDR canvas up to 8192 per axis."};
        return {BackbufferRoute::Portable, "In-process portable Phoenix DLSS5; fenced CPU pixel transfers, no helper process."};
    }
    std::string reason;
    if (caps.bits == 32 && caps.nvidia_required)
        reason = "The selected NVIDIA DLSS5 DLL requires the 64-bit helper; Phoenix DLSS5-only processing does not.";
    else if (caps.bits == 32 && !caps.x86_native_validated)
        reason = "The x86 native driver execution path is not runtime-validated; using the safe graphics helper.";
    else if (caps.api != reshade::api::device_api::d3d12)
        reason = "This graphics API currently needs the D3D12 compatibility bridge.";
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
} // namespace phoenix_compat
