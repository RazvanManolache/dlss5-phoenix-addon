#pragma once
#include "compute_pass.h"
#include "../../../third_party/nlohmann/json.hpp"
namespace phoenix_addon::phoenix::direct {
// Capture same-frame guides while their states and submitting queue are known.
// Present never borrows stale game pointers or guesses their final states.
class PresentGuides {
    struct DeferredInput {
        ComPtr<ID3D12GraphicsCommandList> command;
        ComPtr<ID3D12Resource> motion, depth;
        unsigned width = 0, height = 0;
        uint64_t frame = 0;
        float scale_x = 1, scale_y = 1;
        bool inverted = false, confirmed = false, allow_motion_only = false;
    };
    struct Recorder {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> command;
        uint64_t fence = 0;
    };
    struct Slot {
        ComPtr<ID3D12Resource> motion, depth, readable_depth;
        std::unique_ptr<Pass> motion_pass, depth_pass;
        ComPtr<ID3D12Resource> source_motion, source_depth;
        ID3D12GraphicsCommandList* pending = nullptr;
        uint64_t frame = 0, fence = 0;
        unsigned width = 0, height = 0;
        float scale_x = 1, scale_y = 1;
        bool inverted = false;
        bool confirmed = false;
        bool has_depth = false;
        bool borrowed = false;
    };
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    Pass pipeline_;
    std::vector<Slot> slots_;
    std::vector<DeferredInput> deferred_;
    std::vector<Recorder> recorders_;
    uint64_t staged_ = 0, boundary_captures_ = 0, boundary_refusals_ = 0;
    uint64_t motion_only_captures_ = 0;
    std::string boundary_status_;
    D3D12_RESOURCE_DESC source_motion_description_{}, source_depth_description_{};
    uint64_t submitted_ = 0;
    uint64_t captured_ = 0, acquired_ = 0, missed_ = 0, captured_frame_ = 0, requested_frame_ = 0;
    static constexpr const char* shader = R"(
Texture2D<float2> Motion:register(t0);Texture2D<float> Depth:register(t1);
RWTexture2D<float4> Target:register(u0);
cbuffer C:register(b0){uint width,height,operation,a,b,c;float sx,sy;uint d;};
[numthreads(16,16,1)]void main(uint3 p:SV_DispatchThreadID){
 if(p.x>=width||p.y>=height)return;
 if(operation==0){uint mw,mh;Motion.GetDimensions(mw,mh);
 uint2 q=min(uint2((float2(p.xy)+.5)*float2(mw,mh)/float2(width,height)),uint2(mw-1,mh-1));
 Target[p.xy]=float4(Motion.Load(int3(q,0))*float2(sx,sy),0,0);}
 else{uint dw,dh;Depth.GetDimensions(dw,dh);
 uint2 q=min(uint2((float2(p.xy)+.5)*float2(dw,dh)/float2(width,height)),uint2(dw-1,dh-1));
 Target[p.xy]=Depth.Load(int3(q,0)).xxxx;}
})";

  public:
    nlohmann::json status() const {
        const auto readers=std::count_if(slots_.begin(),slots_.end(),[](const auto& slot){return slot.borrowed;});
        return {{"captured",captured_},{"acquired",acquired_},{"missed",missed_},
                {"active_readers",readers},
                {"captured_frame",captured_frame_},{"requested_frame",requested_frame_},
                {"staged",staged_},{"pending_submission",deferred_.size()},
                {"submission_captures",boundary_captures_},{"submission_refusals",boundary_refusals_},
                {"motion_only_captures",motion_only_captures_},
                {"submission_status",boundary_status_},
                {"source_layout",{{"motion_format",unsigned(source_motion_description_.Format)},
                    {"motion_width",source_motion_description_.Width},{"motion_height",source_motion_description_.Height},
                    {"depth_format",unsigned(source_depth_description_.Format)},
                    {"depth_width",source_depth_description_.Width},{"depth_height",source_depth_description_.Height},
                    {"depth_mips",source_depth_description_.MipLevels},
                    {"depth_samples",source_depth_description_.SampleDesc.Count}}}};
    }
    static const char* shader_source() {
        return shader;
    }
    PresentGuides(ID3D12Device* device, ID3D12CommandQueue* queue)
        : device_(device), queue_(queue), pipeline_(device, shader) {
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        slots_.reserve(8);
        deferred_.reserve(8);
        recorders_.reserve(8);
    }
    bool owner(ID3D12CommandQueue* queue) const {
        return queue == queue_.Get();
    }
    bool idle() const {
        return deferred_.empty() && fence_->GetCompletedValue() >= submitted_ &&
               std::none_of(slots_.begin(), slots_.end(),
                            [](const auto& s) { return s.pending != nullptr || s.borrowed; });
    }
    bool stage(ID3D12GraphicsCommandList* command, ID3D12Resource* motion, ID3D12Resource* depth,
               unsigned width, unsigned height, uint64_t frame, float sx, float sy, bool inverted,
               bool allow_motion_only = false) {
        if (!command || !motion || !depth || !width || !height) return false;
        source_motion_description_=motion->GetDesc(); source_depth_description_=depth->GetDesc();
        auto found = std::find_if(deferred_.begin(),deferred_.end(),[&](const auto& item) {
            return item.command.Get()==command && item.frame==frame;
        });
        if (found==deferred_.end()) {
            if (deferred_.size()==8) return false;
            deferred_.emplace_back(); found=std::prev(deferred_.end());
        }
        *found={command,motion,depth,width,height,frame,sx,sy,inverted,false,allow_motion_only};
        ++staged_;
        return true;
    }
    bool pending_before(ID3D12CommandList* command) const {
        return std::any_of(deferred_.begin(),deferred_.end(),[&](const auto& input) {
            return input.command.Get()==command && input.confirmed;
        });
    }
    void expire_before(uint64_t frame) {
        // These requests inserted no GPU commands. Inline recorded snapshots
        // retain their separate pending/fence lifetime until reset/submission.
        std::erase_if(deferred_,[&](const auto& input){return input.frame<frame;});
    }
    static bool read_only_entry(D3D12_RESOURCE_STATES state) {
        constexpr unsigned reads = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_DEPTH_READ;
        return (unsigned(state)&reads)!=0 && (unsigned(state)&~reads)==0;
    }
    // Called under the global submission lock, after the actual prefix Execute
    // and before executing the provider list. submit() must bypass our outer
    // Execute observer; this command contains only owned snapshot work.
    template<class StateAtBoundary, class Submit>
    bool capture_before(ID3D12CommandQueue* queue, ID3D12CommandList* provider,
                        StateAtBoundary state_at_boundary, Submit submit) {
        const auto found=std::find_if(deferred_.begin(),deferred_.end(),[&](const auto& input) {
            return input.command.Get()==provider && input.confirmed;
        });
        if (found==deferred_.end()) return false;
        auto input=std::move(*found);
        deferred_.erase(found); // Release callbacks cannot invalidate this iterator later.
        const auto reject=[&](const char* reason) {
            ++boundary_refusals_; boundary_status_=reason; return false;
        };
        if (!owner(queue)) return reject("Provider submitted to a different queue");
        const auto motion=state_at_boundary(input.motion.Get());
        if (!motion.known) return reject("Motion state not established by the submitted prefix");
        if (input.motion->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
            return reject("Motion resource can implicitly decay outside the tracked state");
        if (!read_only_entry(D3D12_RESOURCE_STATES(motion.state)))
            return reject("Provider motion is not read-only at list entry");
        const auto depth=state_at_boundary(input.depth.Get());
        const auto depth_description=input.depth->GetDesc();
        const auto depth_format=depth_description.Format;
        const bool scalar_depth=depth_format==DXGI_FORMAT_R32_FLOAT || depth_format==DXGI_FORMAT_R32_TYPELESS || depth_format==DXGI_FORMAT_D32_FLOAT;
        const bool readable_depth=depth.known && read_only_entry(D3D12_RESOURCE_STATES(depth.state)) && scalar_depth &&
            !(depth_description.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
        if (!readable_depth) {
            if (!input.allow_motion_only) return reject("Complete depth guides are unavailable for this consumer");
            input.depth.Reset(); // Never treat stencil or unknown depth state as scalar depth.
        }
        const auto done=fence_->GetCompletedValue();
        auto recorder=std::find_if(recorders_.begin(),recorders_.end(),[&](const auto& item){return item.fence<=done;});
        if (recorder==recorders_.end()) {
            if (recorders_.size()==8) return reject("Snapshot recording ring is full");
            recorders_.emplace_back(); recorder=std::prev(recorders_.end());
        }
        if (!recorder->allocator || !recorder->command) {
            recorder->allocator.Reset(); recorder->command.Reset();
            phoenix_native::hr(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&recorder->allocator)));
            phoenix_native::hr(device_->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,recorder->allocator.Get(),nullptr,IID_PPV_ARGS(&recorder->command)));
            phoenix_native::hr(recorder->command->Close());
        }
        phoenix_native::hr(recorder->allocator->Reset());
        phoenix_native::hr(recorder->command->Reset(recorder->allocator.Get(),nullptr));
        if (!record(recorder->command.Get(),input.motion.Get(),D3D12_RESOURCE_STATES(motion.state),
                    input.depth.Get(),D3D12_RESOURCE_STATES(depth.state),input.width,input.height,
                    input.frame,input.scale_x,input.scale_y,input.inverted)) {
            phoenix_native::hr(recorder->command->Close());
            return reject("Guide texture layout or snapshot ring is unavailable");
        }
        phoenix_native::hr(recorder->command->Close());
        ID3D12CommandList* commands[]{recorder->command.Get()};
        recorder->fence=UINT64_MAX; // Never reuse after an uncertain submission.
        submit(queue,1,commands);
        submitted(queue,1,commands);
        recorder->fence=submitted_;
        ++boundary_captures_;
        boundary_status_=readable_depth?"Captured motion/depth before provider execution":"Captured motion before provider execution; depth unavailable";
        return true;
    }
    bool record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* motion,
                D3D12_RESOURCE_STATES motion_state, ID3D12Resource* depth,
                D3D12_RESOURCE_STATES depth_state, unsigned width, unsigned height, uint64_t frame,
                float scale_x, float scale_y, bool inverted, bool wait_for_provider = false) {
        // Native Phoenix recurrence can use motion without depth. Consumers
        // needing a full guide set must not request that fallback.
        if (!motion)
            return false;
        const auto md = motion->GetDesc(), dd = depth?depth->GetDesc():D3D12_RESOURCE_DESC{};
        for (const auto& d : {md, depth?dd:md})
            if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 ||
                d.MipLevels != 1 || d.DepthOrArraySize != 1)
                return false;
        if ((depth && dd.Format != DXGI_FORMAT_R32_FLOAT && dd.Format != DXGI_FORMAT_R32_TYPELESS &&
             dd.Format != DXGI_FORMAT_D32_FLOAT) ||
            (md.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            return false;
        const auto done = fence_->GetCompletedValue();
        if (done == UINT64_MAX)
            return false;
        auto found = std::find_if(slots_.begin(), slots_.end(),
                                  [&](const auto& s) { return !s.pending && !s.borrowed && s.fence <= done; });
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
            slot.motion_pass = std::make_unique<Pass>(pipeline_, nullptr);
            slot.width = width;
            slot.height = height;
        }
        if (depth && (!slot.depth || slot.depth->GetDesc().Width!=width || slot.depth->GetDesc().Height!=height)) {
            slot.depth=phoenix_native::image(device_.Get(),width,height,DXGI_FORMAT_R32_FLOAT,D3D12_RESOURCE_STATE_COMMON);
            slot.depth_pass=std::make_unique<Pass>(pipeline_,nullptr);
        }
        if (depth && (!slot.readable_depth || slot.readable_depth->GetDesc().Width != dd.Width ||
            slot.readable_depth->GetDesc().Height != dd.Height))
            slot.readable_depth =
                phoenix_native::image(device_.Get(), unsigned(dd.Width), dd.Height,
                                      DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COMMON, false);
        slot.source_motion = motion;
        slot.source_depth = depth;
        slot.scale_x = scale_x;
        slot.scale_y = scale_y;
        slot.inverted = inverted;
        slot.has_depth = depth != nullptr;
        phoenix_native::transition(cmd, motion, motion_state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (depth) {
            phoenix_native::transition(cmd, depth, depth_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            phoenix_native::transition(cmd, slot.readable_depth.Get(), D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
            // R32_FLOAT/R32_TYPELESS/D32_FLOAT share one 32-bit copy format family.
            cmd->CopyResource(slot.readable_depth.Get(), depth);
            phoenix_native::transition(cmd, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, depth_state);
            phoenix_native::transition(cmd, slot.readable_depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        for (unsigned op = 0; op < (depth?2u:1u); ++op) {
            auto* target = op ? slot.depth.Get() : slot.motion.Get();
            phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            (op ? slot.depth_pass : slot.motion_pass)
                ->record(cmd, {motion, depth?slot.readable_depth.Get():nullptr, nullptr}, target,
                         {width, height, op, 0, 0, 0, float(width) / unsigned(md.Width),
                          float(height) / md.Height, 0});
            phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON);
        }
        if(depth)phoenix_native::transition(cmd, slot.readable_depth.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COMMON);
        phoenix_native::transition(cmd, motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   motion_state);
        slot.frame = frame;
        slot.confirmed = !wait_for_provider;
        captured_frame_ = frame; ++captured_;
        if(!depth)++motion_only_captures_;
        slot.pending = cmd;
        slot.fence = 0;
        return true;
    }
    // Capture before the vendor evaluate can change or discard its input
    // state observations; admit the snapshot only after that call succeeds.
    bool confirm(ID3D12GraphicsCommandList* command, uint64_t frame) {
        bool found = false;
        for (auto& input : deferred_)
            if (input.command.Get()==command && input.frame==frame) {input.confirmed=true;found=true;}
        for (auto& slot : slots_)
            if (slot.pending == command && slot.frame == frame) {
                slot.confirmed = true;
                found = true;
            }
        return found;
    }
    void abandoned(ID3D12GraphicsCommandList* cmd) {
        std::erase_if(deferred_,[&](const auto& input){return input.command.Get()==cmd;});
        for (auto& slot : slots_)
            if (slot.pending == cmd) {
                slot.pending = nullptr;
                slot.fence = 0;
                slot.confirmed = false;
            }
    }
    void submitted(ID3D12CommandQueue* queue, unsigned count, ID3D12CommandList* const* lists) {
        std::erase_if(deferred_,[&](const auto& input) {
            return std::find(lists,lists+count,input.command.Get())!=lists+count;
        });
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
                 Frame& result, bool require_depth = false, bool borrow = false) {
        requested_frame_ = frame;
        ++missed_;
        if (!owner(queue))
            return false;
        Slot* latest = nullptr;
        const auto output = result.output->GetDesc();
        for (auto& slot : slots_)
            if (slot.confirmed && !slot.borrowed && (!require_depth || slot.has_depth) && !slot.pending && slot.fence && slot.frame == frame && slot.width == output.Width &&
                slot.height == output.Height && (!latest || slot.fence > latest->fence))
                latest = &slot;
        if (!latest)
            return false;
        for (auto* resource : {latest->motion.Get(), latest->has_depth?latest->depth.Get():nullptr}) {
            if(!resource)continue;
            phoenix_native::transition(cmd, resource, D3D12_RESOURCE_STATE_COMMON,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            phoenix_native::transition(cmd, resource, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_COMMON);
        }
        result.motion = latest->motion.Get();
        result.depth = latest->has_depth?latest->depth.Get():nullptr;
        result.motion_scale_x = latest->scale_x;
        result.motion_scale_y = latest->scale_y;
        result.depth_inverted = latest->inverted ? 1u : 0u;
        latest->borrowed=borrow;
        --missed_; ++acquired_;
        return true;
    }
    // The frontend releases only after all GPU reads retire. A lost/uncertain
    // submission keeps the lease pinned instead of recycling its pixels.
    void release(ID3D12Resource* motion) {
        for(auto& slot:slots_)if(slot.motion.Get()==motion)slot.borrowed=false;
    }
};
} // namespace phoenix_addon::phoenix::direct
