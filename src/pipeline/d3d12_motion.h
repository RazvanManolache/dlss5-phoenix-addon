#pragma once
#include "session.h"
#include "nvof/nvof-motion-provider.hpp"
#include "../native/support/compute_pass.h"
#include <memory>
#include <optional>
#include <chrono>

namespace phoenix_pipeline {
// Runs only on a producer-ordered DIRECT queue, outside another open recording.
// The caller retains frame textures until its consumer fence completes. No CPU
// image transfer occurs unless the user enables the small scene-cut analysis.
class D3D12Motion {
    using ComPtrDevice = Microsoft::WRL::ComPtr<ID3D12Device>;
    template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    ComPtrDevice device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandQueue> compute_queue_;
    ComPtr<ID3D12Fence> fence_, consumer_;
    ComPtr<ID3D12Fence> preparation_fence_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> command_;
    ComPtr<ID3D12Resource> thumbnail_, readback_, history_motion_;
    ComPtr<ID3D12Resource> zero_motion_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    std::unique_ptr<phoenix_addon::phoenix::direct::Pass> thumbnail_pass_;
    std::unique_ptr<phoenix_addon::phoenix::direct::Pass> history_gate_;
    std::unique_ptr<phoenix_addon::phoenix::direct::Pass> zero_pass_;
    NvofMotionProvider provider_;
    NvofMotionProvider::Submission submission_{};
    Discontinuity discontinuity_;
    uint64_t sequence_ = 0, fence_value_ = 0;
    // Producer hand-offs also increment fence_value_. Only the last submitted
    // conversion list owns this allocator; waiting for the producer's newer
    // signal here would reintroduce the very CPU stall ordered mode removes.
    uint64_t allocator_fence_value_=0;
    uint64_t preparation_value_ = 0;
    unsigned width_ = 0, height_ = 0, flow_height_ = 0;
    bool attempted_ = false;
    MotionSource previous_source_ = MotionSource::Zero;
    std::string status_ = "Disabled";
    struct Timing {
        double routing=0,provider_submit=0,post_submit_sync=0,conversion_record=0,conversion_submit=0,conversion_wait=0;
    } timing_;
    bool detailed_=false;
    bool gpu_ordered_=false;
    uint64_t cpu_waits_=0;
    static void check(HRESULT h) { phoenix_native::hr(h); }
    void wait_for_conversion() {
        const auto completed=fence_->GetCompletedValue();
        if(completed==UINT64_MAX)throw std::runtime_error("Motion graphics device removed");
        if(completed>=allocator_fence_value_)return;
        ++cpu_waits_;
        const HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!event)throw std::runtime_error("Motion completion event unavailable");
        const HRESULT armed=fence_->SetEventOnCompletion(allocator_fence_value_,event);
        const DWORD waited=SUCCEEDED(armed)?WaitForSingleObject(event,10000):WAIT_FAILED;
        CloseHandle(event);
        if(waited!=WAIT_OBJECT_0)throw std::runtime_error("Motion queue did not retire");
    }
    void begin() {
        // Ordered consumers normally retire this before the next frame. Keep
        // allocator reuse safe if a caller performs another preparation early.
        if(gpu_ordered_)wait_for_conversion();
        check(allocator_->Reset());check(command_->Reset(allocator_.Get(),nullptr));
    }
    void finish(bool wait_cpu=true) {
        const auto started=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        check(command_->Close()); ID3D12CommandList* lists[]{command_.Get()};
        queue_->ExecuteCommandLists(1, lists); check(queue_->Signal(fence_.Get(), ++fence_value_));
        allocator_fence_value_=fence_value_;
        const auto submitted=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if(wait_cpu)wait_for_conversion();
        if(detailed_){
            timing_.conversion_submit=std::chrono::duration<double,std::milli>(submitted-started).count();
            timing_.conversion_wait=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-submitted).count();
        }
    }
    std::vector<uint8_t> thumbnail(ID3D12Resource* source, D3D12_RESOURCE_STATES state) {
        constexpr unsigned w = 64, h = 36;
        if (!thumbnail_pass_) {
            static constexpr char shader[] = R"hlsl(
Texture2D<float4> Source:register(t0);
RWTexture2D<float> Gray:register(u0);
cbuffer C:register(b0){uint Width,Height;uint unused[7];};
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID){
 if(p.x>=Width||p.y>=Height)return; uint w,h;Source.GetDimensions(w,h);
 uint2 q=min(uint2((float2(p.xy)+.5)*float2(w,h)/float2(Width,Height)),uint2(w-1,h-1));
 Gray[p.xy]=saturate(dot(Source.Load(int3(q,0)).rgb,float3(.2126,.7152,.0722)));
})hlsl";
            thumbnail_pass_ = std::make_unique<phoenix_addon::phoenix::direct::Pass>(device_.Get(), shader);
            thumbnail_ = phoenix_native::image(device_.Get(), w, h, DXGI_FORMAT_R32_FLOAT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            auto desc = thumbnail_->GetDesc(); UINT64 bytes = 0;
            device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint_, nullptr, nullptr, &bytes);
            readback_ = phoenix_native::buffer(device_.Get(), bytes, D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST);
        }
        begin();
        phoenix_native::transition(command_.Get(), source, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(command_.Get(), thumbnail_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        phoenix_addon::phoenix::direct::Constants constants{}; constants.width = w; constants.height = h;
        thumbnail_pass_->record(command_.Get(), {source, nullptr, nullptr}, thumbnail_.Get(), constants);
        phoenix_native::transition(command_.Get(), thumbnail_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{thumbnail_.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
        D3D12_TEXTURE_COPY_LOCATION to{readback_.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprint_};
        command_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        phoenix_native::transition(command_.Get(), thumbnail_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(command_.Get(), source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, state);
        finish();
        uint8_t* mapped = nullptr; const D3D12_RANGE range{0, size_t(readback_->GetDesc().Width)};
        check(readback_->Map(0, &range, reinterpret_cast<void**>(&mapped)));
        std::vector<uint8_t> gray(w * h);
        for (unsigned y = 0; y != h; ++y) for (unsigned x = 0; x != w; ++x) {
            float value = 0; std::memcpy(&value, mapped + footprint_.Offset + y * footprint_.Footprint.RowPitch + x * 4, 4);
            gray[y * w + x] = std::isfinite(value) ? uint8_t(std::clamp(value, 0.f, 1.f) * 255.f) : 0;
        }
        const D3D12_RANGE empty{}; readback_->Unmap(0, &empty); return gray;
    }
public:
    struct Result {
        ID3D12Resource* motion = nullptr;
        ID3D12Resource* rejection = nullptr; // R8: 0=trusted, 1=reject history
        bool reset = false;
        MotionSource source = MotionSource::Zero;
    };
    // gpu_ordered is reserved for owners that retain all resources until a
    // later consumer fence on this SAME direct queue has retired, even on error.
    D3D12Motion(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* consumer,bool gpu_ordered=false)
        : device_(device), queue_(queue), consumer_(consumer),gpu_ordered_(gpu_ordered) {
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr, IID_PPV_ARGS(&command_)));
        check(command_->Close());
    }
    Result prepare(ID3D12Resource* color, D3D12_RESOURCE_STATES state,
                   ID3D12Resource* engine_motion, const Options& p, bool reset,
                   int64_t timestamp_ns = 0, uint64_t stream = 0,
                   std::optional<uint64_t> source_sequence = std::nullopt) {
        detailed_=p.monitoring==int(Monitoring::Detailed);timing_={};
        const auto started=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if (!p.valid()) throw std::invalid_argument("Invalid motion settings");
        const auto desc = color->GetDesc();
        const bool geometry_changed = width_ != desc.Width || height_ != desc.Height;
        if (geometry_changed || flow_height_ != unsigned(p.flow_height)) {
            // This API is called only after the preceding consumer has retired.
            provider_.Shutdown(); attempted_ = false; reset = true;
            width_ = unsigned(desc.Width); height_ = desc.Height; flow_height_ = unsigned(p.flow_height);
            if(geometry_changed){history_motion_.Reset();zero_motion_.Reset();}
        }
        const auto gray = p.scene_detection ? thumbnail(color, state) : std::vector<uint8_t>{};
        ++sequence_;
        const auto discontinuity = discontinuity_.observe({stream, source_sequence.value_or(sequence_), timestamp_ns, width_, height_},
            reset, p.scene_detection, gray, gray.empty() ? 0 : 64, gray.empty() ? 0 : 36);
        reset = reset || discontinuity != ResetReason::None;
        if (reset) provider_.ResetHistory();
        const bool wants_hardware = p.motion_enabled &&
            (p.motion_source == int(MotionSource::Nvidia) ||
             (p.motion_source == int(MotionSource::Automatic) && !engine_motion));
        if (wants_hardware && !attempted_) {
            attempted_ = true; const auto extent = flow_extent(width_, height_, flow_height_);
            if (!compute_queue_) {
                D3D12_COMMAND_QUEUE_DESC description{}; description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
                check(device_->CreateCommandQueue(&description, IID_PPV_ARGS(&compute_queue_)));
                check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&preparation_fence_)));
            }
            provider_.Initialize(device_.Get(), compute_queue_.Get(), consumer_.Get(), width_, height_,
                extent[0], extent[1], desc.Format, nullptr);
        }
        const auto choice = choose_motion(p, engine_motion != nullptr, provider_.IsReady());
        if(choice.source!=previous_source_){reset=true;provider_.ResetHistory();}
        previous_source_=choice.source;
        status_ = choice.reason;
        if(wants_hardware&&!provider_.IsReady())status_=provider_.Status();
        if (choice.source == MotionSource::Engine) {submission_={};return {engine_motion, nullptr, reset, choice.source};}
        if (choice.source != MotionSource::Nvidia) {submission_={};return {nullptr, nullptr, reset, MotionSource::Zero};}
        // Provider preparation command lists are COMPUTE lists. Establish both
        // ownership directions explicitly; even a priming frame must finish
        // copying the source before its producer can overwrite it.
        check(queue_->Signal(fence_.Get(), ++fence_value_));
        check(compute_queue_->Wait(fence_.Get(), fence_value_));
        const auto submit_started=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const bool submitted = provider_.Submit(color, state, sequence_, reset, submission_);
        const auto submit_finished=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        check(compute_queue_->Signal(preparation_fence_.Get(), ++preparation_value_));
        check(queue_->Wait(preparation_fence_.Get(), preparation_value_));
        if (!submitted || !submission_.valid) {
            status_ = provider_.Status(); return {nullptr, nullptr, true, MotionSource::Zero};
        }
        check(queue_->Wait(provider_.CompletionFence(), submission_.completion_value));
        const auto conversion_started=detailed_?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        begin();
        if (!provider_.RecordConversion(command_.Get(), submission_, p.flow_consistency, p.normalized_flow_cost()))
            throw std::runtime_error("Optical flow conversion failed");
        // The recovered networks reject reprojected history outside the image.
        // Translate the provider's per-pixel rejection into that established
        // contract instead of passing unreliable vectors into recurrence.
        // Keep the raw vectors intact for diagnostics and translation tests.
        if(!history_gate_){
            static constexpr char gate[]=R"hlsl(
Texture2D<float2> Motion:register(t0);Texture2D<float> Rejection:register(t1);
RWTexture2D<float2> Output:register(u0);
cbuffer C:register(b0){uint W,H;uint unused[7];};
[numthreads(16,16,1)]void main(uint3 p:SV_DispatchThreadID){
 if(p.x>=W||p.y>=H)return;float2 v=Motion.Load(int3(p.xy,0));
 float rejection=Rejection.Load(int3(p.xy,0));
 Output[p.xy]=rejection>=.5||any(!isfinite(v))?float2(W*2,H*2):v;
})hlsl";
            history_gate_=std::make_unique<phoenix_addon::phoenix::direct::Pass>(device_.Get(),gate);
        }
        if(!history_motion_)history_motion_=phoenix_native::image(device_.Get(),width_,height_,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_addon::phoenix::direct::Constants constants{};constants.width=width_;constants.height=height_;
        phoenix_native::transition(command_.Get(),history_motion_.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        history_gate_->record(command_.Get(),{submission_.motion,submission_.history_mask,nullptr},history_motion_.Get(),constants);
        phoenix_native::transition(command_.Get(),history_motion_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if(detailed_){
            timing_.routing=std::chrono::duration<double,std::milli>(submit_started-started).count();
            timing_.provider_submit=std::chrono::duration<double,std::milli>(submit_finished-submit_started).count();
            timing_.post_submit_sync=std::chrono::duration<double,std::milli>(conversion_started-submit_finished).count();
            timing_.conversion_record=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-conversion_started).count();
        }
        finish(!gpu_ordered_);
        return {history_motion_.Get(), submission_.history_mask, reset, MotionSource::Nvidia};
    }
    void consumed(uint64_t fence) { if (submission_.valid) provider_.MarkNeuralUse(submission_, fence); }
    // A dedicated immutable zero surface avoids trusting stale shared vectors
    // when a producer stops supplying guides or the user explicitly selects zero.
    ID3D12Resource* zero_motion() {
        if(!zero_motion_){
            if(!width_||!height_)throw std::logic_error("Prepare motion geometry before requesting zero vectors");
            if(!zero_pass_){
                static constexpr char shader[]=R"hlsl(
RWTexture2D<float2> Output:register(u0);
cbuffer C:register(b0){uint W,H;uint unused[7];};
[numthreads(16,16,1)]void main(uint3 p:SV_DispatchThreadID){if(p.x<W&&p.y<H)Output[p.xy]=0;}
)hlsl";
                zero_pass_=std::make_unique<phoenix_addon::phoenix::direct::Pass>(device_.Get(),shader);
            }
            zero_motion_=phoenix_native::image(device_.Get(),width_,height_,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            begin();phoenix_addon::phoenix::direct::Constants c{};c.width=width_;c.height=height_;
            zero_pass_->record(command_.Get(),{nullptr,nullptr,nullptr},zero_motion_.Get(),c);
            phoenix_native::transition(command_.Get(),zero_motion_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
            finish();
        }
        return zero_motion_.Get();
    }
    bool visualize(ID3D12GraphicsCommandList* command, ID3D12Resource* output, unsigned mode) {
        return submission_.valid && provider_.RecordVisualization(command, submission_, output, mode, 1.f);
    }
    const std::string& status() const noexcept { return status_; }
    ID3D12Resource* raw_motion()const noexcept{return submission_.motion;}
    bool gpu_ordered()const noexcept{return gpu_ordered_;}
    uint64_t cpu_waits()const noexcept{return cpu_waits_;}
    phoenix_bundle::Json timings()const {
        if(!detailed_)return nullptr;
        return {{"routing",timing_.routing},{"provider_submit",timing_.provider_submit},
            {"post_submit_sync",timing_.post_submit_sync},{"conversion_record",timing_.conversion_record},
            {"conversion_submit",timing_.conversion_submit},{"conversion_wait",timing_.conversion_wait}};
    }
};
} // namespace phoenix_pipeline
