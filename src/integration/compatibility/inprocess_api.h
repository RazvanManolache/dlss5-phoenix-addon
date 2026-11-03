#pragma once
#include "../reshade/feeder_api.h"
#include "session_policy.h"
#include "../../../third_party/reshade/reshade_api.hpp"
#include <functional>
#include <atomic>
#include <mutex>
#include <memory>

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
    std::function<Json()> guide_status;
    std::function<void(ID3D12Resource*)> release_present_guides;
    // Resolve a ReShade-wrapped private queue to its exact observed native
    // identity before constructing driver-facing command lists/resources.
    std::function<ID3D12CommandQueue*(ID3D12CommandQueue*)> native_queue;
    // Private queues can exist without ReShade init/destroy events (OpenGL).
    // Their owning Resources object balances this lease through retirement.
    std::function<void(ID3D12CommandQueue*,bool)> local_queue_lifetime;
    bool available() const { return prepare && release && observe && forget && evaluate && progress && ui && action && update_tuning; }
};
inline InProcessApi in_process_api;
// Closing admission must not erase std::functions still used by a detached
// worker or its resource destructors. Providers are process-pinned by attach;
// this lease additionally retains their callback registry through retirement.
struct InProcessApiLifetime {
    std::mutex mutex;
    size_t leases = 0;
    bool accepting = true;
};
inline auto& in_process_api_lifetime = *new InProcessApiLifetime;
class InProcessApiOwner {
public:
    InProcessApiOwner() {
        std::lock_guard lock(in_process_api_lifetime.mutex);
        if (!in_process_api_lifetime.accepting || !in_process_api.available())
            throw std::runtime_error("In-process provider is unavailable or retiring");
        ++in_process_api_lifetime.leases;
    }
    InProcessApiOwner(const InProcessApiOwner&) = delete;
    ~InProcessApiOwner() {
        std::lock_guard lock(in_process_api_lifetime.mutex);
        if (!--in_process_api_lifetime.leases && !in_process_api_lifetime.accepting)
            in_process_api = {};
    }
};
using InProcessApiLease = std::shared_ptr<InProcessApiOwner>;
inline InProcessApiLease lease_in_process_api() { return std::make_shared<InProcessApiOwner>(); }
inline bool in_process_api_available() {
    std::lock_guard lock(in_process_api_lifetime.mutex);
    return in_process_api_lifetime.accepting && in_process_api.available();
}
inline bool can_publish_in_process_api() {
    std::lock_guard lock(in_process_api_lifetime.mutex);
    return !in_process_api_lifetime.leases;
}
inline bool publish_in_process_api(InProcessApi callbacks) {
    std::lock_guard lock(in_process_api_lifetime.mutex);
    if (in_process_api_lifetime.leases) return false;
    in_process_api = std::move(callbacks);
    in_process_api_lifetime.accepting = true;
    return true;
}
inline void retire_in_process_api() {
    std::lock_guard lock(in_process_api_lifetime.mutex);
    in_process_api_lifetime.accepting = false;
    if (!in_process_api_lifetime.leases) in_process_api = {};
}
// Native feeder callbacks share one model session. Hold this lease until every
// bridge resource/hook has retired, including asynchronous staging workers.
inline std::mutex in_process_executor_mutex;
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
