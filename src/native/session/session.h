#pragma once
#include "../network/network.h"
#include "../support/policy.h"
#include "../support/live_controls.h"
#include "../support/dispatch.h"
#include "../support/host_memory.h"
#include "../support/colour_bridge.h"
#include "../support/exposure_meter.h"
#include "../support/denormalize.h"
#include "../support/appearance_gpu.h"
#include <nvsdk_ngx.h>
#include <mutex>

namespace phoenix_addon {
using Microsoft::WRL::ComPtr;
using Json = phoenix_bundle::Json;
inline uint64_t completed_submission_count(uint64_t fence, uint64_t signaled,
                                           uint64_t committed) noexcept {
    if (fence == UINT64_MAX)
        return 0; // D3D12 uses this value for device removal.
    return (std::min)(fence, (std::min)(signaled, committed));
}
// A runtime that proxies D3D12 objects answers ID3D12DeviceChild::GetDevice
// with its own device wrapper, while the resources this adapter and the engine
// create report the original device underneath it. Measured on the live game:
// the queue reported a wrapper whose IUnknown differs from the device that
// owns every real resource. Everything allocated, bound and validated here has
// to name that owning device, so the documented unwrap is applied once, and
// device comparison always goes through canonical IUnknown identity.
constexpr GUID unwrapped_object_guid = {
    0x7f2c9a11, 0x3b4e, 0x4d6a, {0x81, 0x2f, 0x5e, 0x9c, 0xd3, 0x7a, 0x1b, 0x42}};
inline ComPtr<ID3D12Device> unwrap_device(ID3D12Device* device) {
    check(device != nullptr, "phoenix null device");
    ComPtr<IUnknown> original;
    if (SUCCEEDED(device->QueryInterface(unwrapped_object_guid,
                                         reinterpret_cast<void**>(original.GetAddressOf()))) &&
        original) {
        ComPtr<ID3D12Device> native;
        if (SUCCEEDED(original.As(&native)) && native)
            return native;
    }
    return ComPtr<ID3D12Device>(device);
}
inline ComPtr<ID3D12Device> owning_device(ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Device> reported;
    phoenix_native::hr(queue->GetDevice(IID_PPV_ARGS(&reported)));
    return reported;
}
inline bool same_device(ID3D12Device* left, ID3D12Device* right) {
    if (!left || !right)
        return false;
    ComPtr<IUnknown> a, b;
    return SUCCEEDED(left->QueryInterface(IID_PPV_ARGS(&a))) &&
           SUCCEEDED(right->QueryInterface(IID_PPV_ARGS(&b))) && a.Get() == b.Get();
}
// Measured on the live game: more than one object answers as "the device",
// and a queue reports a different one than every resource does. The identity
// that matters is the one device children report, because that is what every
// resource ownership check compares against. Rather than trusting a reported
// pointer or an unwrap direction, the relationship is proven: an object
// created through the reported device must report the same owner. The probe
// fence is released immediately and never submitted.
inline bool device_forwards_to(ID3D12Device* reported, ID3D12Device* owner) {
    if (!reported || !owner)
        return false;
    if (same_device(reported, owner))
        return true;
    // Probe with the same object kind the ownership rule is about: a texture.
    // Different object kinds are answered by different layers in this process.
    try {
        auto probe = phoenix_native::image(reported, 4, 4, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
        ComPtr<ID3D12Device> probe_owner;
        if (FAILED(probe->GetDevice(IID_PPV_ARGS(&probe_owner))))
            return false;
        return same_device(probe_owner.Get(), owner);
    } catch (...) {
        return false;
    }
}
// The device an engine resource reports as its owner.
inline ComPtr<ID3D12Device> resource_device(ID3D12Resource* resource) {
    check(resource != nullptr, "phoenix null resource for device resolution");
    ComPtr<ID3D12Device> reported;
    phoenix_native::hr(resource->GetDevice(IID_PPV_ARGS(&reported)));
    return reported;
}
// Several objects in this process answer as "the device", and which one a
// queue reports is not the one resources report. Instead of reasoning about
// the layering, the adapter finds it by construction: allocate a small probe,
// adopt whatever device that allocation reports as its owner, and repeat until
// the answer is stable. Every ownership comparison afterwards - including the
// network's own input validation - then uses one self-consistent identity.
// Probe textures are 4x4, never bound and released immediately.
// A layer that proxies D3D12 objects hands back wrapped descriptor heaps.
// Recording those into a command list obtained from underneath that layer
// makes the runtime dereference a pointer it does not own, which was observed
// live as the process disappearing inside SetDescriptorHeaps. The adapter
// therefore records into an unwrapped list and must allocate through a device
// whose objects come back unwrapped. That is asked of each candidate directly.
inline bool creates_unwrapped_objects(ID3D12Device* device) {
    if (!device)
        return false;
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 1;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap))) || !heap)
        return false;
    ComPtr<IUnknown> original;
    if (SUCCEEDED(heap->QueryInterface(unwrapped_object_guid,
                                       reinterpret_cast<void**>(original.GetAddressOf()))) &&
        original && static_cast<void*>(original.Get()) != static_cast<void*>(heap.Get()))
        return false;
    return true;
}
struct DeviceChoice {
    ComPtr<ID3D12Device> device;
    std::string basis;
    unsigned candidates_tried = 0;
    bool resolved = false;
};
inline DeviceChoice choose_allocation_device(ID3D12Device* reported,
                                             ID3D12Resource* engine_resource,
                                             const char* reported_name,
                                             const char* reported_unwrapped_name) {
    DeviceChoice choice;
    ComPtr<ID3D12Device> held;
    if (reported)
        reported->AddRef(), held.Attach(reported);
    const auto engine_owner = resource_device(engine_resource);
    const std::pair<const char*, ComPtr<ID3D12Device>> candidates[]{
        {reported_name, held},
        {reported_unwrapped_name, unwrap_device(held.Get())},
        {"engine_resource_owner", engine_owner},
        {"engine_resource_owner_unwrapped", unwrap_device(engine_owner.Get())},
    };
    for (const auto& candidate : candidates) {
        ++choice.candidates_tried;
        if (!candidate.second || !creates_unwrapped_objects(candidate.second.Get()))
            continue;
        // Allocations made through it must be owned by the same device the
        // engine's own resources report, or the network cannot bind both.
        if (!phoenix_native::device_owns(candidate.second.Get(), engine_owner.Get()))
            continue;
        choice.device = candidate.second;
        choice.basis = candidate.first;
        choice.resolved = true;
        return choice;
    }
    return choice;
}
inline DeviceChoice resolve_allocation_device(ID3D12CommandQueue* queue,
                                              ID3D12Resource* engine_resource) {
    ComPtr<ID3D12Device> reported;
    phoenix_native::hr(queue->GetDevice(IID_PPV_ARGS(&reported)));
    return choose_allocation_device(reported.Get(), engine_resource, "queue_reported_device",
                                    "queue_reported_device_unwrapped");
}
// List-based variant for the self-drive stage: the game drives RR through
// the sl internal path, whose lists never pass observed submission, so no
// submitting queue can be proven -- and none is needed. The output device
// is established from the list's own device plus the engine resource, the
// same candidate discipline as above.
inline DeviceChoice resolve_allocation_device_from_list(ID3D12GraphicsCommandList* cmd,
                                                        ID3D12Resource* engine_resource) {
    ComPtr<ID3D12Device> reported;
    phoenix_native::hr(cmd->GetDevice(IID_PPV_ARGS(&reported)));
    return choose_allocation_device(reported.Get(), engine_resource, "list_reported_device",
                                    "list_reported_device_unwrapped");
}
inline ID3D12Resource* parameter_resource(const NVSDK_NGX_Parameter* params, const char* key,
                                          bool required = false) {
    // Native 0x19f30 reader uses Get(void**) at vtable +0x40. MSVC groups
    // overloaded virtuals in reverse order; the typed D3D12 getter is +0x48.
    void* value = nullptr;
    if (!params || !NVSDK_NGX_SUCCEED(params->Get(key, &value)))
        value = nullptr;
    check(!required || value, "phoenix missing NR input/output parameter");
    return static_cast<ID3D12Resource*>(value);
}

// Supplied by the engine integration at the Evaluate boundary. Unknown states
// must never be converted to COMMON. All resources are single-subresource.
struct FrameEvidence {
    ID3D12CommandQueue* queue = nullptr;
    D3D12_RESOURCE_STATES color_state{}, motion_state{}, output_state{};
    bool states_verified = false;
    // True only when output_state was established by evidence: a transition in
    // the current recording, a transition already submitted on the same queue,
    // an untransitioned creation state, or a texture the caller created itself.
    bool output_state_verified = false;
    bool color_is_nr_sdr_proxy = false;
    // Restore heaps/root bindings/PSOs invalidated by the injected recorder.
    void (*restore_bindings)(void*, ID3D12GraphicsCommandList*) = nullptr;
    void* restore_context = nullptr;
    std::shared_ptr<void> binding_lease;
    std::map<std::string,D3D12_RESOURCE_STATES> optional_states;
    bool experimental_appearance_contract = false;
    uint32_t appearance_generation = 0;
};

class Session {
    phoenix_native::memory::SessionLifetime allocation_lifetime_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12Resource> color_, motion_;
    ComPtr<ID3D12Resource> pending_color_, pending_motion_, pending_output_;
    std::vector<ComPtr<ID3D12Resource>> pending_optional_;
    std::unique_ptr<phoenix::direct::AppearanceStage> appearance_;
    FrameDispatch dispatch_;
    std::unique_ptr<phoenix_native::Network> network_;
    Transaction transaction_;
    Sampling sampling_;
    bool history_ = false;
    uint64_t last_sampling_ = 0;
    uint64_t comparisons_ = 0;
    bool last_reset_ = false, last_motion_ = false, last_game_reset_ = false, native_fresh_ = false;
    ID3D12GraphicsCommandList* command_ = nullptr;
    unsigned submit_sightings_ = 0;
    mutable ComPtr<ID3D12Device> accepted_resource_device_;
    uint64_t signaled_ = 0;
    bool retirement_unknown_ = false;
    bool submission_observed_ = false;
    ID3D12GraphicsCommandList* pending_commit_ = nullptr;
    // The comparison path defers its commit exactly as the claim path does,
    // but through its own slot so the two can never be confused. Without this
    // the Network stays `pending_` after the first frame and every later one
    // fails with "invalid/lost/pending network or recording thread" -- which
    // is precisely what happened: one frame recorded, 4678 refused.
    ID3D12GraphicsCommandList* pending_comparison_commit_ = nullptr;
    struct Lease {
        uint64_t fence_value;
        ComPtr<ID3D12Resource> color, motion, output;
        std::vector<ComPtr<ID3D12Resource>> optional;
    };
    host_memory::Vector<Lease, host_memory::Tag::engine_leases> leases_;
    unsigned width_, height_;
    DXGI_FORMAT color_format_, motion_format_;
    // HDR bridge: set only when the session was constructed for a game whose
    // NR color is linear HDR rather than the SDR proxy. Meter, bridge and
    // denormalize are built together and used together; null together.
    bool hdr_ = false;
    // HDR control override (looking, not measuring): the game may ask its
    // native NR for a mode no captured plan covers (RE9 menus send
    // intensity 2, style 2, UI correction on). The kernels never see these
    // values -- resolve_controls refuses before they could -- so overriding
    // them runs OUR kernels at settings WE support while native runs at
    // the game's. Pair results record the override; acceptance stays
    // bounded-error, never byte-exact. Default off: without an explicit
    // config declaration the refusal stands.
    bool hdr_controls_override_ = false;
    std::unique_ptr<exposure_meter::Meter> meter_;
    std::unique_ptr<colour_bridge::Bridge> bridge_;
    std::unique_ptr<denormalize::Denormalize> denormalize_;

    void validate_resource(ID3D12Resource* resource, DXGI_FORMAT format) const {
        check(resource != nullptr, "phoenix missing engine resource");
        auto description = resource->GetDesc();
        check(description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                  description.Width == width_ && description.Height == height_ &&
                  description.DepthOrArraySize == 1 && description.MipLevels == 1 &&
                  description.SampleDesc.Count == 1 && description.SampleDesc.Quality == 0 &&
                  description.Format == format,
              "phoenix engine resource geometry/format changed");
        ComPtr<ID3D12Device> owner;
        phoenix_native::hr(resource->GetDevice(IID_PPV_ARGS(&owner)));
        // The engine's resources may report a different handle for the same
        // underlying device than the one this adapter allocates through, so
        // ownership is established by construction and then remembered.
        if (owner.Get() != accepted_resource_device_.Get()) {
            check(phoenix_native::device_owns(owner.Get(), device_.Get()),
                  "phoenix engine resource device changed");
            accepted_resource_device_ = owner;
        }
    }
    static ID3D12Resource* resource(const NVSDK_NGX_Parameter* params, const char* key) {
        return parameter_resource(params, key, true);
    }
    static ID3D12Resource* optional_resource(const NVSDK_NGX_Parameter* params, const char* key) {
        return parameter_resource(params, key);
    }
    static void copy_input(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source,
                           D3D12_RESOURCE_STATES state, ID3D12Resource* target) {
        phoenix_native::transition(cmd, source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(target, source);
        phoenix_native::transition(cmd, target, D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }

  public:
    // Session owns its API table through retirement. No NGX host, native modules,
    // native weights allocation, native history or original Evaluate is used.
    Session(ID3D12Device* device, ID3D12CommandQueue* queue, const Json& frame,
            const phoenix_model::WeightView& weights,
            const std::vector<phoenix_kernel::Binary>& candidates, DXGI_FORMAT color_format,
            DXGI_FORMAT motion_format, bool native_fresh = false,
            std::shared_ptr<phoenix_native::memory::Generation> accounting = {},
            bool hdr_bridge = false, bool hdr_control_override = false)
        : allocation_lifetime_{accounting}, device_(device), queue_(queue),
          native_fresh_(native_fresh), width_(frame.at("width")), height_(frame.at("height")),
          color_format_(color_format), motion_format_(motion_format), hdr_(hdr_bridge),
          hdr_controls_override_(hdr_bridge && hdr_control_override) {
        check(device && queue && queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
              "phoenix DIRECT queue required");
        dispatch_.configure_plan(frame);
        check(device_forwards_to(owning_device(queue).Get(), device_.Get()),
              "phoenix queue device mismatch");
        color_ = phoenix_native::image(device, width_, height_, color_format,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
        phoenix_native::memory::resource(device, color_.Get(), accounting,
                                         phoenix_native::memory::Kind::color);
        motion_ = phoenix_native::image(device, width_, height_, motion_format,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
        phoenix_native::memory::resource(device, motion_.Get(), accounting,
                                         phoenix_native::memory::Kind::motion);
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        phoenix_native::memory::track(fence_.Get(), accounting,
                                      phoenix_native::memory::Kind::fence);
        appearance_=std::make_unique<phoenix::direct::AppearanceStage>(device,width_,height_,accounting);
        network_ = std::make_unique<phoenix_native::Network>(device, dispatch_.api(), frame,
                                                             weights, candidates, color_.Get(),
                                                             motion_.Get(), accounting);
        const auto ends = phoenix_model::native_layout(frame.value("model_architecture", std::string("phoenix-network37/1"))).group_ends();
        check(network_->group_ends() == std::vector<unsigned>(ends.begin(), ends.end()),
              "phoenix group schedule changed; review skip mask");
        leases_.reserve(32);
        if (hdr_bridge) {
            // Fail fast: an engine format the bridge refuses must never
            // reach record(). matches() pins the format for the session's
            // life, so this construction gate covers every later frame.
            check(colour_bridge::bridge_accepts_source(color_format),
                  "phoenix HDR bridge refuses this engine color format");
            meter_ = std::make_unique<exposure_meter::Meter>(device, width_, height_);
            bridge_ = std::make_unique<colour_bridge::Bridge>(device, width_, height_);
            denormalize_ = std::make_unique<denormalize::Denormalize>(device, width_, height_);
            check(meter_->target() && bridge_->target(), "phoenix HDR bridge produced no targets");
        }
        if (accounting) {
            std::lock_guard<std::mutex> lock(accounting->shared->mutex);
            accounting->host_container_capacity_bytes =
                network_->host_container_capacity_bytes() + leases_.capacity() * sizeof(Lease);
            accounting->constructed = true;
        }
    }
    Session(const Session&) = delete;
    bool pending() const noexcept {
        return transaction_.pending();
    }
    uintptr_t pending_transaction_command() const noexcept {
        return transaction_.pending_command();
    }
    uintptr_t pending_transaction_queue() const noexcept {
        return transaction_.pending_queue();
    }
    Json host_references() const {
        uint64_t slots = 0;
        std::set<ID3D12Resource*> unique;
        auto add = [&](const auto& resource) {
            if (resource) {
                ++slots;
                unique.insert(resource.Get());
            }
        };
        add(pending_color_);
        add(pending_motion_);
        add(pending_output_);
        for (const auto& resource : pending_optional_) add(resource);
        for (const auto& lease : leases_) {
            add(lease.color);
            add(lease.motion);
            add(lease.output);
            for (const auto& resource : lease.optional) add(resource);
        }
        return {{"generation",
                 allocation_lifetime_.generation ? allocation_lifetime_.generation->id : 0},
                {"engine_resource_reference_slots", slots},
                {"distinct_engine_resource_pointers", unique.size()},
                {"fenced_lease_count", leases_.size()},
                {"lease_capacity", leases_.capacity()},
                {"pending_recording", command_ != nullptr},
                {"retirement_unknown", retirement_unknown_}};
    }
    bool failed() const noexcept {
        return transaction_.failed();
    }
    uint64_t committed() const noexcept {
        return transaction_.committed();
    }
    uint64_t completed() const noexcept {
        return completed_submission_count(fence_->GetCompletedValue(), signaled_, committed());
    }
    uint64_t last_sampling() const noexcept {
        return last_sampling_;
    }
    uint64_t signaled() const noexcept {
        return signaled_;
    }
    uint64_t fence_completed() const noexcept {
        return fence_->GetCompletedValue();
    }
    uint64_t candidate_launches() const noexcept {
        return dispatch_.launches();
    }
    unsigned expected_candidate_launches() const noexcept {
        return dispatch_.expected_launches();
    }
    Json frame_controls() const {
        const auto& controls = dispatch_.controls();
        auto result = Json{{"sampling", last_sampling_},
                           {"reset", last_reset_},
                           {"game_reset", last_game_reset_},
                           {"motion_present", last_motion_},
                           {"recurrent", controls.recurrent},
                           {"normalized_motion_x", controls.motion_x},
                           {"normalized_motion_y", controls.motion_y},
                           {"tone", controls.tone},
                           {"structure", controls.structure},
                           {"skin", controls.skin},
                           {"skin_structure", controls.skin_structure}};
        // Present only when active, so SDR pair JSONs are byte-unchanged.
        if (hdr_controls_override_)
            result["controls_overridden"] = true;
        return result;
    }
    bool matches(ID3D12CommandQueue* queue, const D3D12_RESOURCE_DESC& desc) const noexcept {
        return queue == queue_.Get() && desc.Width == width_ && desc.Height == height_ &&
               desc.Format == color_format_;
    }
    ID3D12GraphicsCommandList* command() const noexcept {
        return command_;
    }
    // True while the calling thread is the session's recorder. The engine
    // may migrate recording across threads between frames; the Transaction
    // contract (CPU-pinned) refuses that on a live session, so the add-on
    // retires and rebuilds instead of recording foreign.
    bool recording_thread() const noexcept {
        return transaction_.owner_thread();
    }
    // Submission sightings: how many submitted batches contained the open
    // recording. More than one without an observed reset between them names
    // a re-submit (or a double-firing hook), not a mismatching transaction.
    void note_submit_sighting() noexcept {
        ++submit_sightings_;
    }
    unsigned submit_sightings() const noexcept {
        return submit_sightings_;
    }
    ID3D12CommandQueue* queue() const noexcept {
        return queue_.Get();
    }
    bool submission_matches(ID3D12CommandQueue* queue,
                            ID3D12GraphicsCommandList* cmd) const noexcept {
        return queue == queue_.Get() && cmd == command_ &&
               transaction_.submission_matches(uintptr_t(cmd), uintptr_t(queue));
    }

    // The engine submits from a different thread than it records on, so the
    // reconstructed network is committed here, on its own recording thread,
    // after the submission that made it durable was observed elsewhere.
    void settle() {
        if (!pending_commit_)
            return;
        check(transaction_.owner_thread(), "phoenix commit must run on the recording thread");
        auto* cmd = pending_commit_;
        pending_commit_ = nullptr;
        try {
            network_->commit(cmd);
            sampling_.commit();
            history_ = true;
        } catch (...) {
            invalidate();
            retirement_unknown_ = true;
            throw;
        }
    }
    bool settled() const noexcept {
        return pending_commit_ == nullptr;
    }

    // Comparison only. Runs the reconstructed network into ITS OWN output and
    // returns that texture. It never reads, writes, transitions or even names
    // the engine's output, so the engine-output-state contract -- the last
    // blocker in RE Requiem -- simply does not apply here. Nothing is claimed
    // and nothing the engine submits is altered.
    //
    // `colour` must already be in the plan's own colour format. Where a game
    // hands DLSSNR something else (RE9 uses R11G11B10_FLOAT) the caller widens
    // it first; see production_game_phoenix_colour_bridge.h. `copy_input` uses
    // CopyResource, which requires identical formats, so a mismatch here is a
    // programming error rather than something to convert silently.
    //
    // This deliberately does NOT open a transaction or take a lease. Those
    // exist to make a CLAIM durable across submission, and there is no claim:
    // the caller owns `colour` and `motion` for the frame, exactly as the
    // engine already does.
    ID3D12Resource* record_comparison(ID3D12GraphicsCommandList* cmd, ID3D12Resource* colour,
                                      D3D12_RESOURCE_STATES colour_state, ID3D12Resource* motion,
                                      D3D12_RESOURCE_STATES motion_state,
                                      const NVSDK_NGX_Parameter* params,
                                      const Controls* override_controls = nullptr) {
        check(cmd != nullptr && colour != nullptr && params != nullptr,
              "phoenix comparison needs a list, colour and parameters");
        // Commit the previous frame's work first, so the Network is not
        // still pending when we record again. The any_thread forms: RE9
        // evaluates on more than one thread, and this whole call already
        // runs under the ASI's g_mutex, which is the real serialization.
        if (pending_comparison_commit_) {
            auto* previous = pending_comparison_commit_;
            pending_comparison_commit_ = nullptr;
            try {
                network_->commit_any_thread(previous);
                sampling_.commit();
                history_ = true;
            } catch (...) {
                invalidate();
                retirement_unknown_ = true;
                throw;
            }
        }
        check(
            !hdr_,
            "phoenix HDR comparison must go through the claim path; the split band has no denormalize destination");
        validate_resource(colour, color_format_);
        if (motion)
            validate_resource(motion, motion_format_);
        check(colour != motion, "phoenix comparison rejects aliased colour and motion");
        network_->validate_command_any_thread(cmd, queue_.Get());

        // `override_controls` exists for one purpose: showing our own network
        // on screen in a game whose NR settings no captured plan covers. It
        // runs OUR kernels at settings WE support -- it does not change what
        // the native DLL was asked to do, and it means the two pictures differ
        // for that reason as well as any other. A caller using it is looking,
        // not measuring, and must say so.
        const auto game_controls = override_controls
                                       ? *override_controls
                                       : read_controls(*params, [](NVSDK_NGX_Result result) {
                                             return NVSDK_NGX_SUCCEED(result);
                                         });
        // First-record here keys off comparisons, NOT the transaction counter:
        // this path deliberately never opens a transaction, so committed() is
        // 0 forever and would re-arm the takeover gate every frame (RE9 b2:
        // 1 recorded, 2159 refused). The claim path below keeps committed().
        const auto controls = effective_controls(game_controls, comparisons_ == 0, native_fresh_);
        // Validate before any GPU recording.
        resolve_controls(controls, width_, height_, motion != nullptr, history_, 0);
        try {
            last_sampling_ = sampling_.begin(controls.reset != 0);
            last_reset_ = controls.reset != 0;
            last_motion_ = motion != nullptr;
            last_game_reset_ = game_controls.reset != 0;
            const auto resolved = resolve_controls(controls, width_, height_, motion != nullptr,
                                                   history_, last_sampling_);
            FrameDispatch::Scope frame_scope(dispatch_, cmd, resolved);
            copy_input(cmd, colour, colour_state, color_.Get());
            if (motion)
                copy_input(cmd, motion, motion_state, motion_.Get());
            auto* result = network_->record(cmd, !resolved.recurrent, uint32_t(last_sampling_),
                                            queue_.Get(), true);
            frame_scope.finish();
            pending_comparison_commit_ = cmd;
            ++comparisons_;
            return result;
        } catch (...) {
            invalidate();
            retirement_unknown_ = true;
            // Recording may be partial: the caller must abandon this list.
            throw;
        }
    }
    uint64_t comparisons() const noexcept {
        return comparisons_;
    }

    void record(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* params,
                const FrameEvidence& evidence, const PhoenixLiveControls* live = nullptr,
                unsigned transformer_blocks = 8) {
        settle();
        transaction_.validate(uintptr_t(cmd), uintptr_t(evidence.queue));
        check(evidence.queue == queue_.Get() && evidence.states_verified &&
                  evidence.output_state_verified && (evidence.color_is_nr_sdr_proxy || hdr_) &&
                  evidence.restore_bindings,
              "phoenix missing verified queue/state/colour/bindings contract");
        auto* color = resource(params, "DLSSNR.Color");
        auto* motion = optional_resource(params, "DLSSNR.MVec");
        auto* output = resource(params, "DLSSNR.Output");
        validate_resource(color, color_format_);
        if (motion)
            validate_resource(motion, motion_format_);
        validate_resource(output, DXGI_FORMAT_R16G16B16A16_FLOAT);
        check(color != motion && color != output && motion != output,
              "phoenix aliased engine inputs/output rejected");
        phoenix::direct::AppearanceTextures textures;
        const char* optional_keys[]={"DLSSNR.ControlMask","DLSSNR.UI","DLSSNR.UIAlpha","DLSSNR.Backbuffer","DLSSNR.BidirectionalDistortionField"};
        ID3D12Resource** optional_targets[]={&textures.mask,&textures.ui,&textures.alpha,&textures.backbuffer,&textures.field};
        const DXGI_FORMAT optional_formats[]={DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT};
        for(unsigned i=0;i<5;++i) {
            *optional_targets[i]=optional_resource(params,optional_keys[i]);
            auto* value=*optional_targets[i];if(!value)continue;
            check(evidence.experimental_appearance_contract,"Optional appearance planes require the explicit experimental contract");
            validate_resource(value,optional_formats[i]);
            check(value!=color&&value!=motion&&value!=output,"Appearance input aliases frame resources");
            for(unsigned j=0;j<i;++j)check(value!=*optional_targets[j],"Appearance inputs alias");
            check(evidence.optional_states.count(optional_keys[i])!=0,"Missing observed appearance resource state");
        }
        check(!(textures.ui&&textures.alpha),"Supply UI or UIAlpha");
        const auto game_controls = read_controls(
            *params, [](NVSDK_NGX_Result result) { return NVSDK_NGX_SUCCEED(result); });
        auto controls = effective_controls(game_controls, committed() == 0, native_fresh_);
        if (live)
            controls = live->apply(controls);
        controls = hdr_overridden_controls(controls, hdr_controls_override_);
        const bool appearance=phoenix::direct::AppearanceStage::requested(controls,textures);
        if(appearance) {
            check(evidence.experimental_appearance_contract,"Appearance output requires the explicit experimental contract");
            check(color_->GetDesc().Format==DXGI_FORMAT_R16G16B16A16_FLOAT,"Appearance stage requires the SDR proxy format");
            check(!hdr_||(!textures.ui&&!textures.backbuffer),"HDR UI/backbuffer transfer contract is unavailable");
            check(std::isfinite(controls.intensity)&&controls.intensity>=0&&controls.intensity<=2&&controls.style>=0&&controls.style<=2,"Invalid appearance controls");
            check(!controls.ui_correction||textures.ui||textures.alpha,"UI correction needs explicit coverage");
        }
        if(appearance_->changed(textures,evidence.appearance_generation))controls.reset=1;
        auto core_controls=controls;core_controls.intensity=1;core_controls.style=0;core_controls.ui_correction=0;
        // Validate before any GPU recording or counter reservation.
        resolve_controls(core_controls, width_, height_, motion != nullptr, history_, 0);
        network_->validate_command(cmd, evidence.queue);
        const auto completed = fence_->GetCompletedValue();
        check(completed != UINT64_MAX, "phoenix completion fence reports device loss");
        leases_.erase(std::remove_if(leases_.begin(), leases_.end(),
                                     [completed](const Lease& lease) {
                                         return lease.fence_value <= completed;
                                     }),
                      leases_.end());
        check(leases_.size() < 32, "phoenix engine-resource lease limit reached");
        pending_color_ = color;
        pending_motion_ = motion;
        pending_output_ = output;
        pending_optional_.clear();
        for(auto** pointer:optional_targets)if(*pointer)pending_optional_.emplace_back(*pointer);
        transaction_.begin(uintptr_t(cmd), uintptr_t(evidence.queue));
        command_ = cmd;
        submission_observed_ = false;
        submit_sightings_ = 0;
        phoenix_native::note("record_transaction_open");
        try {
            last_sampling_ = sampling_.begin(controls.reset != 0);
            last_reset_ = controls.reset != 0;
            last_motion_ = motion != nullptr;
            last_game_reset_ = game_controls.reset != 0;
            auto resolved = resolve_controls(core_controls, width_, height_, motion != nullptr,
                                                   history_, last_sampling_);
            if (textures.mask && !network_->control_mask_handle)
                network_->bind_control_mask(appearance_->mask_target());
            resolved.control_mask_handle=textures.mask?network_->control_mask_handle:0;
            resolved.style=float(controls.style)/128.f;
            if(textures.field){resolved.motion_x=1.f/width_;resolved.motion_y=1.f/height_;}
            FrameDispatch::Scope frame_scope(dispatch_, cmd, resolved, transformer_blocks);
            phoenix_native::note("record_copy_color");
            if (hdr_) {
                // HDR bridge input: meter the engine frame, normalize it
                // into the plan domain, and feed the network that. The
                // engine color returns to its observed state; every barrier
                // is epoch-observed through the list hook, the same form
                // the copy path's barriers take.
                phoenix_native::transition(cmd, color, evidence.color_state,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                meter_->record(cmd, color);
                bridge_->record(cmd, color, meter_->target());
                phoenix_native::transition(cmd, color,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           evidence.color_state);
                copy_input(cmd, bridge_->target(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           color_.Get());
            } else
                copy_input(cmd, color, evidence.color_state, color_.Get());
            phoenix_native::note("record_copy_motion");
            if (motion)
                copy_input(cmd, motion, evidence.motion_state, motion_.Get());
            const unsigned appearance_slot=unsigned(committed()%32);
            if(appearance) {
                for(unsigned i=0;i<5;++i)if(*optional_targets[i])
                    phoenix_native::transition(cmd,*optional_targets[i],evidence.optional_states.at(optional_keys[i]),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                appearance_->prepare(cmd,color_.Get(),motion?motion_.Get():nullptr,textures,controls,appearance_slot);
            } else appearance_->absent();
            phoenix_native::note("record_network");
            // Stable motion allocation does not imply engine motion availability.
            // Passing reset here selects the parent's nonrecurrent program, but
            // sampling resets ONLY for the actual native Reset parameter.
            auto* result = network_->record(cmd, !resolved.recurrent, uint32_t(last_sampling_),
                                            evidence.queue);
            phoenix_native::note("record_network_done");
            frame_scope.finish();
            if(appearance) {
                result=appearance_->present(cmd,result,textures,controls,appearance_slot);
                appearance_->remember_mode(textures,evidence.appearance_generation);
                for(unsigned i=0;i<5;++i)if(*optional_targets[i])
                    phoenix_native::transition(cmd,*optional_targets[i],D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,evidence.optional_states.at(optional_keys[i]));
            }
            // HDR output returns through the denormalize pass, not the copy:
            // the result is already the SRV the pass samples, and the
            // engine output goes observed-state -> UAV -> observed-state.
            if (!hdr_)
                phoenix_native::transition(cmd, result,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (!hdr_)
                phoenix_native::transition(cmd, output, evidence.output_state,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
            if (hdr_) {
                phoenix_native::transition(cmd, output, evidence.output_state,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                denormalize_->record(cmd, result, meter_->target(), output, evidence.output_state);
            } else
                cmd->CopyResource(output, result);
            if (!hdr_)
                phoenix_native::transition(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST,
                                           evidence.output_state);
            if (!hdr_)
                phoenix_native::transition(cmd, result, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::note("record_restore_bindings");
            evidence.restore_bindings(evidence.restore_context, cmd);
            phoenix_native::note("record_complete");
        } catch (...) {
            invalidate();
            retirement_unknown_ = true;
            // Recording may be partial: the caller must abandon this list.
            throw;
        }
    }

    // Called AFTER the real ExecuteCommandLists. D3D12 Execute returns void;
    // successful Signal plus device status establishes accepted queue order,
    // while fence completion establishes retirement, not image correctness.
    void after_submit(ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* cmd) {
        submission_observed_ = true;
        try {
            const bool identity = submission_matches(queue, cmd);
            check(identity, "phoenix wrong submitted command/queue");
            const auto value = signaled_ + 1;
            check(value != 0, "phoenix fence exhausted");
            phoenix_native::hr(queue->Signal(fence_.Get(), value));
            signaled_ = value;
            phoenix_native::hr(device_->GetDeviceRemovedReason());
            transaction_.submitted(uintptr_t(cmd), uintptr_t(queue), true);
            // Deferred to settle(): the reconstructed network requires its own
            // recording thread and this callback runs on the engine's submitter.
            pending_commit_ = cmd;
            leases_.push_back({value, std::move(pending_color_), std::move(pending_motion_),
                               std::move(pending_output_),std::move(pending_optional_)});
            command_ = nullptr;
        } catch (...) {
            invalidate();
            retirement_unknown_ = true;
            throw;
        }
    }
    void invalidate() noexcept {
        pending_commit_ = nullptr;
        transaction_.invalidate();
        sampling_.invalidate();
        network_->invalidate();
        history_ = false;
    }
    bool cross_thread_submission() const noexcept {
        return transaction_.cross_thread_submission();
    }
    void uncertain_submission() noexcept {
        invalidate();
        retirement_unknown_ = true;
    }
    // The engine must guarantee this list can NEVER subsequently be submitted.
    void abandoned(ID3D12GraphicsCommandList* cmd) {
        check(command_ == cmd && !submission_observed_,
              "phoenix submitted command cannot be declared abandoned");
        invalidate();
        command_ = nullptr;
        retirement_unknown_ = false;
    }
    bool can_destroy() const noexcept {
        if (!transaction_.owner_thread())
            return false;
        if (FAILED(device_->GetDeviceRemovedReason()))
            return true;
        if (pending() || command_ || retirement_unknown_ || pending_commit_)
            return false;
        return fence_->GetCompletedValue() >= signaled_;
    }
};

using ResolveFrame = bool (*)(void*, ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                              const NVSDK_NGX_Parameter*, FrameEvidence*);
// Registration transfers a shared lease, not unique ownership. The sidecar
// retains it through fence completion or explicit command abandonment.
struct Endpoint {
    std::shared_ptr<Session> session;
    const NVSDK_NGX_Handle* feature = nullptr;
    ResolveFrame resolve = nullptr;
    void* context = nullptr;
};
} // namespace production_game41

#ifdef PRODUCTION_GAME_PHOENIX_EXPORTS
#define PRODUCTION_GAME_PHOENIX_API __declspec(dllexport)
#else
#define PRODUCTION_GAME_PHOENIX_API __declspec(dllimport)
#endif
extern "C" PRODUCTION_GAME_PHOENIX_API
    BOOL WINAPI CnrPHOENIXRegister(const phoenix_addon::Endpoint* endpoint);
extern "C" PRODUCTION_GAME_PHOENIX_API BOOL WINAPI
CnrPHOENIXAbandoned(ID3D12GraphicsCommandList* cmd);
extern "C" PRODUCTION_GAME_PHOENIX_API BOOL WINAPI CnrPHOENIXUnregister();
extern "C" PRODUCTION_GAME_PHOENIX_API BOOL WINAPI CnrPHOENIXSummary(const wchar_t* path);
