#pragma once
#include "compute_pass.h"
namespace phoenix_addon::phoenix::direct {
// Capture same-frame guides while their states and submitting queue are known.
// Present never borrows stale game pointers or guesses their final states.
class PresentGuides {
    struct Slot {
        ComPtr<ID3D12Resource> motion, depth, readable_depth;
        std::unique_ptr<Pass> motion_pass, depth_pass;
        ComPtr<ID3D12Resource> source_motion, source_depth;
        ID3D12GraphicsCommandList* pending = nullptr;
        uint64_t frame = 0, fence = 0;
        unsigned width = 0, height = 0;
        float scale_x = 1, scale_y = 1;
        bool inverted = false;
    };
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    Pass pipeline_;
    std::vector<Slot> slots_;
    uint64_t submitted_ = 0;
    static constexpr const char* shader = R"(
Texture2D<float2> Motion:register(t0);Texture2D<float> Depth:register(t1);
RWTexture2D<float4> Target:register(u0);
cbuffer C:register(b0){uint width,height,operation,a,b,c;float sx,sy;uint d;};
[numthreads(16,16,1)]void main(uint3 p:SV_DispatchThreadID){
 if(p.x>=width||p.y>=height)return;
 uint mw,mh,dw,dh;Motion.GetDimensions(mw,mh);Depth.GetDimensions(dw,dh);
 if(operation==0){uint2 q=min(uint2((float2(p.xy)+.5)*float2(mw,mh)/float2(width,height)),uint2(mw-1,mh-1));
 Target[p.xy]=float4(Motion.Load(int3(q,0))*float2(sx,sy),0,0);}
 else{uint2 q=min(uint2((float2(p.xy)+.5)*float2(dw,dh)/float2(width,height)),uint2(dw-1,dh-1));
 Target[p.xy]=Depth.Load(int3(q,0)).xxxx;}
})";

  public:
    static const char* shader_source() {
        return shader;
    }
    PresentGuides(ID3D12Device* device, ID3D12CommandQueue* queue)
        : device_(device), queue_(queue), pipeline_(device, shader) {
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        slots_.reserve(8);
    }
    bool owner(ID3D12CommandQueue* queue) const {
        return queue == queue_.Get();
    }
    bool idle() const {
        return fence_->GetCompletedValue() >= submitted_ &&
               std::none_of(slots_.begin(), slots_.end(),
                            [](const auto& s) { return s.pending != nullptr; });
    }
    bool record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* motion,
                D3D12_RESOURCE_STATES motion_state, ID3D12Resource* depth,
                D3D12_RESOURCE_STATES depth_state, unsigned width, unsigned height, uint64_t frame,
                float scale_x, float scale_y, bool inverted) {
        // Scalar depth is copied into a readable R32 texture. Refuse multi-plane families.
        if (!motion || !depth)
            return false;
        const auto md = motion->GetDesc(), dd = depth->GetDesc();
        for (const auto& d : {md, dd})
            if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 ||
                d.MipLevels != 1 || d.DepthOrArraySize != 1)
                return false;
        if ((dd.Format != DXGI_FORMAT_R32_FLOAT && dd.Format != DXGI_FORMAT_R32_TYPELESS &&
             dd.Format != DXGI_FORMAT_D32_FLOAT) ||
            (md.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            return false;
        const auto done = fence_->GetCompletedValue();
        if (done == UINT64_MAX)
            return false;
        auto found = std::find_if(slots_.begin(), slots_.end(),
                                  [&](const auto& s) { return !s.pending && s.fence <= done; });
        if (found == slots_.end()) {
            if (slots_.size() == 8)
                return false;
            slots_.emplace_back();
            found = std::prev(slots_.end());
        }
        auto& slot = *found;
        if (!slot.motion || slot.width != width || slot.height != height) {
            slot.motion =
                phoenix_native::image(device_.Get(), width, height, DXGI_FORMAT_R16G16_FLOAT,
                                      D3D12_RESOURCE_STATE_COMMON);
            slot.depth = phoenix_native::image(device_.Get(), width, height, DXGI_FORMAT_R32_FLOAT,
                                               D3D12_RESOURCE_STATE_COMMON);
            slot.motion_pass = std::make_unique<Pass>(pipeline_, nullptr);
            slot.depth_pass = std::make_unique<Pass>(pipeline_, nullptr);
            slot.width = width;
            slot.height = height;
        }
        if (!slot.readable_depth || slot.readable_depth->GetDesc().Width != dd.Width ||
            slot.readable_depth->GetDesc().Height != dd.Height)
            slot.readable_depth =
                phoenix_native::image(device_.Get(), unsigned(dd.Width), dd.Height,
                                      DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COMMON, false);
        slot.source_motion = motion;
        slot.source_depth = depth;
        slot.scale_x = scale_x;
        slot.scale_y = scale_y;
        slot.inverted = inverted;
        phoenix_native::transition(cmd, motion, motion_state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, depth, depth_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, slot.readable_depth.Get(), D3D12_RESOURCE_STATE_COMMON,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        // R32_FLOAT/R32_TYPELESS/D32_FLOAT share one 32-bit copy format family.
        cmd->CopyResource(slot.readable_depth.Get(), depth);
        phoenix_native::transition(cmd, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, depth_state);
        phoenix_native::transition(cmd, slot.readable_depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        for (unsigned op = 0; op < 2; ++op) {
            auto* target = op ? slot.depth.Get() : slot.motion.Get();
            phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            (op ? slot.depth_pass : slot.motion_pass)
                ->record(cmd, {motion, slot.readable_depth.Get(), nullptr}, target,
                         {width, height, op, 0, 0, 0, float(width) / unsigned(md.Width),
                          float(height) / md.Height, 0});
            phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON);
        }
        phoenix_native::transition(cmd, slot.readable_depth.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COMMON);
        phoenix_native::transition(cmd, motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   motion_state);
        slot.frame = frame;
        slot.pending = cmd;
        slot.fence = 0;
        return true;
    }
    void abandoned(ID3D12GraphicsCommandList* cmd) {
        for (auto& slot : slots_)
            if (slot.pending == cmd) {
                slot.pending = nullptr;
                slot.fence = 0;
            }
    }
    void submitted(ID3D12CommandQueue* queue, unsigned count, ID3D12CommandList* const* lists) {
        for (auto& slot : slots_)
            for (unsigned i = 0; i < count; ++i)
                if (slot.pending && slot.pending == lists[i]) {
                    check(owner(queue), "Present guide snapshot submitted to another queue");
                    phoenix_native::hr(queue->Signal(fence_.Get(), ++submitted_));
                    slot.fence = submitted_;
                    slot.pending = nullptr;
                }
    }
    template <class Frame>
    bool acquire(ID3D12GraphicsCommandList* cmd, ID3D12CommandQueue* queue, uint64_t frame,
                 Frame& result) {
        if (!owner(queue))
            return false;
        Slot* latest = nullptr;
        const auto output = result.output->GetDesc();
        for (auto& slot : slots_)
            if (!slot.pending && slot.fence && slot.frame == frame && slot.width == output.Width &&
                slot.height == output.Height && (!latest || slot.fence > latest->fence))
                latest = &slot;
        if (!latest)
            return false;
        for (auto* resource : {latest->motion.Get(), latest->depth.Get()}) {
            phoenix_native::transition(cmd, resource, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            phoenix_native::transition(cmd, resource, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_COMMON);
        }
        result.motion = latest->motion.Get();
        result.depth = latest->depth.Get();
        result.motion_scale_x = latest->scale_x;
        result.motion_scale_y = latest->scale_y;
        result.depth_inverted = latest->inverted ? 1u : 0u;
        return true;
    }
};
} // namespace phoenix_addon::phoenix::direct
