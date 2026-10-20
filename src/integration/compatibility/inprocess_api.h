#pragma once
#include "../reshade/feeder_api.h"
#include "session_policy.h"
#include <functional>
#include <atomic>

namespace phoenix_compat {
using Json = nlohmann::json;
// Registered by full x86/x64 add-ons. Legacy transport-only packages lack
// these callbacks and continue to select the packaged helper.
struct InProcessApi {
    std::function<void(reshade::api::effect_runtime*, const Settings&)> prepare;
    std::function<void(reshade::api::effect_runtime*)> release;
    std::function<void(reshade::api::effect_runtime*, ID3D12GraphicsCommandList*, ID3D12CommandQueue*)> observe;
    void (*forget)(ID3D12GraphicsCommandList*) noexcept = nullptr;
    PhoenixFeederEvaluateProc evaluate = nullptr;
    PhoenixFeederProgressJsonProc progress = nullptr, ui = nullptr;
    BOOL(WINAPI* action)(uint32_t, uint32_t) = nullptr;
    std::function<void(const Settings&)> update_tuning;
    std::function<bool(ID3D12GraphicsCommandList*, ID3D12CommandQueue*, PhoenixFeederFrameV1&)> present_guides;
    bool available() const { return prepare && release && observe && forget && evaluate && progress && ui && action && update_tuning; }
};
inline InProcessApi in_process_api;
// Shared across runtime recreation: a timed-out submission is not made safe
// merely by destroying its window and selecting a different processing route.
inline std::atomic<bool> in_process_gpu_uncertain{false};
inline Json local_json(PhoenixFeederProgressJsonProc read) {
    if (!read) return Json::object();
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        const auto size = read(nullptr, 0);
        if (!size || size > 1024u * 1024u) break;
        std::string text(size, '\0');
        const auto written = read(text.data(), size);
        if (written && written <= size) return Json::parse(text.c_str());
    }
    throw std::runtime_error("In-process Phoenix DLSS5 metadata is unavailable");
}
} // namespace phoenix_compat
