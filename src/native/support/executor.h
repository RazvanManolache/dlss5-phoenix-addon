#pragma once
#include "../session/session.h"
#include "routing.h"
#include "locked_capture.h"

namespace phoenix_addon::phoenix {
// The host owns normalization, game resources, command submission and HDR
// resolve. An executor owns neural processing and returns an owned proxy image.
struct ExecutorFrame {
    ID3D12GraphicsCommandList* command = nullptr;
    ID3D12Resource* color = nullptr;  // RGBA16F model proxy; shader-read state
    ID3D12Resource* motion = nullptr; // RG16F/RG32F; shader-read state
    ID3D12Resource* depth = nullptr;  // optional, independently observed state below
    // Host-adapted RGBA32F: current->previous motion in output pixels,
    // positive linear depth, guide validity. Never pass raw device depth here.
    ID3D12Resource* guides = nullptr; // shader-read state, same extent as color
    bool device_depth = false;        // explicit normalized hardware depth [0,1], not linear
    unsigned transformer_blocks = 8;
    D3D12_RESOURCE_STATES depth_state = D3D12_RESOURCE_STATE_COMMON;
    Controls controls;
    const Settings* settings = nullptr;
    uint64_t serial = 0, completed_serial = 0;
    uint64_t locked_input_token = 0;
    direct::AppearanceTextures appearance;
    uint32_t appearance_generation = 0;
};
class NeuralExecutor {
  public:
    virtual ~NeuralExecutor() = default;
    // Attach existing transport resources to their owning game generation.
    // Inference allocations in another process remain separately reported.
    virtual void account(const std::shared_ptr<phoenix_native::memory::Generation>&) {}
    // Records GPU work, never waits for inference. Returns either the caller's
    // unchanged input or an executor-owned image, in shader-read state. The
    // caller retains both through completion of all recorded command lists.
    virtual ID3D12Resource* record(const ExecutorFrame& frame) = 0;
    virtual Json status() const = 0;
    virtual void invalidate() noexcept = 0;
};
}
