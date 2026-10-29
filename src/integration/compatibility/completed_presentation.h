#pragma once
#include <d3d12.h>
#include <cstdint>
#include <stdexcept>
#include <wrl/client.h>

namespace phoenix_compat {
// Only a retired, successfully processed image can become a presentation
// fallback. Its lifetime belongs to the bridge, not the rotating game buffer.
class CompletedPresentation {
    bool available_ = false;
  public:
    // Cold start and failed initialization have no processed pixels to hold.
    // Clear the display instead of treating the live game image as a result.
    static void record_loading(ID3D12GraphicsCommandList* command, ID3D12Resource* backbuffer) {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> views;
        D3D12_DESCRIPTOR_HEAP_DESC heap{};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap.NumDescriptors = 1;
        if (FAILED(backbuffer->GetDevice(IID_PPV_ARGS(&device))) ||
            FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&views))))
            throw std::runtime_error("Cannot create the loading presentation view");
        const auto view = views->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(backbuffer, nullptr, view);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {backbuffer, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                              D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET};
        command->ResourceBarrier(1, &barrier);
        const float black[4] = {0, 0, 0, 1};
        command->ClearRenderTargetView(view, black, 0, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        command->ResourceBarrier(1, &barrier);
        // ClearRenderTargetView consumes CPU descriptors while recording;
        // the view heap owns no pixels and need not survive GPU execution.
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
