#pragma once
#include "d3d11_local_surface.h"
#include "d3d12_bridge.h"

namespace phoenix_compat {
// Reuse the native DX12 evaluator, modes, tuning, screenshots, optical flow and
// completed-frame policy. Only ownership transfer differs from a DX12 game.
class D3D11LocalBridge final : public D3D12Bridge {
    std::unique_ptr<D3D11LocalSurface> surface_;
    static std::unique_ptr<D3D11LocalSurface> create(reshade::api::effect_runtime* runtime,
                                                   const reshade::api::resource_desc& desc) {
        runtime->get_command_queue()->flush_immediate_command_list();
        return std::make_unique<D3D11LocalSurface>(
            reinterpret_cast<ID3D11Device*>(uintptr_t(runtime->get_device()->get_native())),
            desc.texture.width, desc.texture.height, typed_format(desc.texture.format));
    }
    D3D11LocalBridge(reshade::api::effect_runtime* runtime, const reshade::api::resource_desc& desc,
                     std::unique_ptr<D3D11LocalSurface> surface)
        : D3D12Bridge(runtime, desc, surface->device(), surface->queue()), surface_(std::move(surface)) {}
    reshade::api::resource image() const { return {uint64_t(uintptr_t(surface_->image()))}; }
public:
    D3D11LocalBridge(reshade::api::effect_runtime* runtime, const reshade::api::resource_desc& desc)
        : D3D11LocalBridge(runtime, desc, create(runtime, desc)) {}
    ~D3D11LocalBridge() override {
        try { surface_->retire(); } catch (...) { in_process_gpu_uncertain = true; }
        // Both APIs can retain this allocation. Match the native bridge's
        // timeout policy: pin a failed generation rather than free live work.
        if (!safe_to_fallback()) surface_.release();
    }
    bool safe_to_fallback() const override {
        return D3D12Bridge::safe_to_fallback() && surface_->retired() && !in_process_gpu_uncertain;
    }
    const char* execution_kind() const override { return "in_process_d3d11"; }
    const char* execution_description() const override { return "In-process GPU (DX11/DX12 shared textures on the game adapter)"; }
    bool frame(reshade::api::resource back, const Settings& settings, bool apply, bool coherent = true) override {
        auto* texture = reinterpret_cast<ID3D11Texture2D*>(uintptr_t(back.handle));
        surface_->capture(texture);
        const bool composed = D3D12Bridge::frame(image(), settings, apply, coherent);
        if (composed) surface_->publish(texture);
        else surface_->retire();
        return composed;
    }
    bool protect_presentation(reshade::api::resource back) override {
        if (!safe_to_fallback() || !D3D12Bridge::protect_presentation(image())) return false;
        surface_->publish(reinterpret_cast<ID3D11Texture2D*>(uintptr_t(back.handle)));
        return true;
    }
    Json ui_snapshot() override {
        auto result = D3D12Bridge::ui_snapshot();
        result["processing_location"] = execution_kind();
        result["processing_host_pid"] = result["inference_pid"] = GetCurrentProcessId();
        result["inference_pids"] = Json::array({GetCurrentProcessId()});
        result["transport"] = "same_adapter_shared_gpu";
        return result;
    }
};
} // namespace phoenix_compat
