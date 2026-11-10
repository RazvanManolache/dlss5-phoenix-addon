#pragma once
#include <d3d12.h>
#include <cstdint>
#include <stdexcept>
#include <wrl/client.h>
#include "../../../third_party/reshade/reshade_api.hpp"

namespace phoenix_compat {
// Only a retired, successfully processed image can become a presentation
// fallback. Its lifetime belongs to the bridge, not the rotating game buffer.
class CompletedPresentation {
    bool available_ = false;
  public:
    // Cold start and failed initialization have no processed pixels to hold.
    // Clear the display instead of treating the live game image as a result.
    static void record_loading(reshade::api::command_list* command, reshade::api::resource backbuffer) {
        record_loading_commands(command,backbuffer);
    }
    // The software fixture supplies the same descriptor/command operations.
    template<class Command>
    static void record_loading_commands(Command* command, reshade::api::resource backbuffer) {
        using namespace reshade::api;
        auto* device=command->get_device();
        resource_view view{};
        const auto format=format_to_default_typed(device->get_resource_desc(backbuffer).texture.format,0);
        if(!device->create_resource_view(backbuffer,resource_usage::render_target,resource_view_desc(format),&view))
            throw std::runtime_error("Cannot create the loading presentation view");
        // Use the owning API for both descriptors and commands. Besides the
        // descriptor namespace, this marks ReShade's immediate list dirty so
        // loading-only frames are submitted too.
        command->barrier(backbuffer,resource_usage::present,resource_usage::render_target);
        const float black[4] = {0, 0, 0, 1};
        command->clear_render_target_view(view,black,0,nullptr);
        command->barrier(backbuffer,resource_usage::render_target,resource_usage::present);
        device->destroy_resource_view(view);
    }
    void invalidate() { available_ = false; }
    void completed() { available_ = true; }
    bool record_hold(ID3D12GraphicsCommandList* command, ID3D12Resource* completed,
                     ID3D12Resource* backbuffer, bool coherent) const {
        if (!coherent || !available_) return false;
        const auto source = completed->GetDesc(), target = backbuffer->GetDesc();
        if (source.Dimension != target.Dimension || source.Width != target.Width ||
            source.Height != target.Height || source.Format != target.Format ||
            source.SampleDesc.Count != target.SampleDesc.Count)
            throw std::runtime_error("Completed presentation no longer matches the game buffer");
        D3D12_RESOURCE_BARRIER barriers[2]{};
        for (auto& barrier : barriers) {
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        }
        barriers[0].Transition.pResource = completed;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers[1].Transition.pResource = backbuffer;
        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        command->ResourceBarrier(2, barriers);
        command->CopyResource(backbuffer, completed);
        for (auto& barrier : barriers) {
            barrier.Transition.StateBefore = barrier.Transition.StateAfter;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        }
        command->ResourceBarrier(2, barriers);
        return true;
    }
};
}
