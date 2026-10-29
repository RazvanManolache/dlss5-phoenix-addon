#pragma once
#include "inprocess_api.h"
#include "../../pipeline/paired_screenshot.h"
#include "session_policy.h"
#include "completed_presentation.h"
#include "host/phoenix-format-adapter.h"
#include "host/phoenix-host-config.h"
#include "host/phoenix-host-safety.h"
#include <wrl/client.h>
#include "../../pipeline/session.h"
#if defined(_WIN64)
#include "../../pipeline/d3d12_motion.h"
#endif

namespace phoenix_compat {
// This route borrows the game's DIRECT queue. It owns its command list and all
// conversion/model inputs; no CPU pixel transport or fake SR feature is used.
class D3D12Bridge {
    template<class T> using Com = Microsoft::WRL::ComPtr<T>;
    struct Resources {
        // Share the failed-generation lifetime: a timeout cannot free readbacks
        // that an unretired GPU submission may still reference.
        phoenix_pipeline::screenshots::D3D12Pair screenshot_pair;
        Com<ID3D12Device> device;
        Com<ID3D12CommandQueue> queue;
        Com<ID3D12CommandAllocator> allocator;
        Com<ID3D12GraphicsCommandList> list;
        Com<ID3D12Fence> fence;
        Com<ID3D12Resource> capture, presentation, motion;
        Com<ID3D12QueryHeap> pipeline_queries;
        Com<ID3D12Resource> pipeline_readback;
        uint64_t timestamp_frequency = 0;
        Com<ID3D12DescriptorHeap> clear_cpu, clear_gpu;
        PhoenixFormatAdapter adapter;
        void (*forget)(ID3D12GraphicsCommandList*) noexcept = nullptr;
        HANDLE event = nullptr;
#if defined(_WIN64)
        std::unique_ptr<phoenix_pipeline::D3D12Motion> estimated_motion;
#endif
        ~Resources() {
            if (forget && list) forget(list.Get());
            if (event) CloseHandle(event);
        }
    };
    std::unique_ptr<Resources> resources_ = std::make_unique<Resources>();
    Settings last_tuning_;
    reshade::api::effect_runtime* runtime_;
    unsigned width_, height_;
    DXGI_FORMAT format_;
    uint64_t signal_ = 0, completed_ = 0, presented_ = 0;
    bool retired_ = true, primed_ = false, reset_ = true, owner_ = false;
    Json ui_ = Json::object();
    phoenix_pipeline::Monitor pipeline_monitor_;
    phoenix_pipeline::LatencyPolicy latency_policy_;
    Settings adaptive_, adaptive_request_;
    bool adaptive_active_ = false, reconfigure_pipeline_ = false;
    Json pipeline_report_ = Json::object();
    uint64_t ui_tick_ = 0;
    DeferredTimeout deferred_;
    CompletedPresentation presentation_;
    bool held_last_frame_ = false;
    uint64_t held_frames_ = 0, deferred_frames_ = 0, frames_with_guides_ = 0, frames_without_guides_ = 0;

    bool hold_completed(reshade::api::resource back, bool coherent) {
        auto& resources = *resources_;
        begin();
        if (!presentation_.record_hold(resources.list.Get(), resources.presentation.Get(),
                reinterpret_cast<ID3D12Resource*>(uintptr_t(back.handle)), coherent)) {
            checked(resources.list->Close(), "Cannot close unused presentation recording");
            return false;
        }
        submit();
        held_last_frame_ = true;
        ++held_frames_; ++presented_;
        return true;
    }

    static void checked(HRESULT result, const char* message) {
        if (FAILED(result)) {
            char code[32]{}; sprintf_s(code, " (0x%08X)", unsigned(result));
            throw std::runtime_error(std::string(message) + code);
        }
    }
    static D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        return barrier;
    }
    void copy(ID3D12Resource* source, ID3D12Resource* destination) {
        auto& r = *resources_;
        D3D12_RESOURCE_BARRIER barriers[]{
            transition(source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(destination, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)};
        r.list->ResourceBarrier(2, barriers);
        r.list->CopyResource(destination, source);
        for (auto& barrier : barriers) std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        r.list->ResourceBarrier(2, barriers);
    }
    void begin() {
        auto& r = *resources_;
        checked(r.allocator->Reset(), "Cannot reset the in-process allocator");
        checked(r.list->Reset(r.allocator.Get(), nullptr), "Cannot reset the in-process command list");
    }
    void submit() {
        auto& r = *resources_;
        checked(r.list->Close(), "Cannot close in-process commands");
        ID3D12CommandList* lists[]{r.list.Get()};
        retired_ = false;
        r.queue->ExecuteCommandLists(1, lists);
        const auto signaled = r.queue->Signal(r.fence.Get(), ++signal_);
        if (FAILED(signaled)) in_process_gpu_uncertain = true;
        checked(signaled, "Cannot signal in-process completion");
        const bool done = PhoenixWaitFence(signal_, 5000,
            [&] { return r.fence->GetCompletedValue(); },
            [&] { return SUCCEEDED(r.fence->SetEventOnCompletion(signal_, r.event)); },
            [&](uint32_t timeout) { return WaitForSingleObject(r.event, timeout) == WAIT_OBJECT_0; },
            [] { return GetTickCount64(); });
        if (!done) {
            in_process_gpu_uncertain = true;
            throw std::runtime_error("In-process GPU submission did not retire; processing stopped, no helper fallback is safe.");
        }
        retired_ = true;
    }
    void clear_motion() {
        auto& r = *resources_;
        auto barrier = transition(r.motion.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        r.list->ResourceBarrier(1, &barrier);
        ID3D12DescriptorHeap* heaps[]{r.clear_gpu.Get()};
        r.list->SetDescriptorHeaps(1, heaps);
        const float zero[4]{};
        r.list->ClearUnorderedAccessViewFloat(r.clear_gpu->GetGPUDescriptorHandleForHeapStart(),
            r.clear_cpu->GetCPUDescriptorHandleForHeapStart(), r.motion.Get(), zero, 0, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        r.list->ResourceBarrier(1, &barrier);
    }
public:
    bool request_screenshot_pair(const fs::path& folder) { return resources_->screenshot_pair.output.request(folder); }
    const Json& screenshot_pair_status() const { return resources_->screenshot_pair.output.result; }
    static DXGI_FORMAT typed_format(reshade::api::format format) {
        using F = reshade::api::format;
        if (format == F::r8g8b8a8_unorm || format == F::r8g8b8a8_unorm_srgb || format == F::r8g8b8a8_typeless)
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        if (format == F::b8g8r8a8_unorm || format == F::b8g8r8a8_unorm_srgb || format == F::b8g8r8a8_typeless)
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        if (format == F::r10g10b10a2_unorm || format == F::r10g10b10a2_typeless)
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        if (format == F::r16g16b16a16_float) return DXGI_FORMAT_R16G16B16A16_FLOAT;
        throw std::runtime_error("No in-process SDR conversion for this backbuffer format");
    }
    D3D12Bridge(reshade::api::effect_runtime* runtime, const reshade::api::resource_desc& desc)
        : runtime_(runtime), width_(desc.texture.width), height_(desc.texture.height),
          format_(typed_format(desc.texture.format)) {
        auto& r = *resources_;
        auto* api = runtime->get_device();
        checked(reinterpret_cast<IUnknown*>(api->get_native())->QueryInterface(IID_PPV_ARGS(&r.device)),
                "No native DX12 device");
        auto* queue = runtime->get_command_queue();
        if (!queue) throw std::runtime_error("No presentation queue");
        checked(reinterpret_cast<IUnknown*>(queue->get_native())->QueryInterface(IID_PPV_ARGS(&r.queue)),
                "No native DX12 queue");
        if (r.queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
            throw std::runtime_error("In-process processing needs a DIRECT queue");
        checked(r.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&r.allocator)),
                "Cannot create in-process allocator");
        checked(r.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, r.allocator.Get(), nullptr,
                                          IID_PPV_ARGS(&r.list)), "Cannot create in-process command list");
        // Observe this exact implementation before creating frame resources.
        // Do not assume another game command list installed matching hooks.
        in_process_api.observe(runtime_, r.list.Get(), r.queue.Get());
        r.forget = in_process_api.forget;
        checked(r.list->Close(), "Cannot initialize in-process command list");
        checked(r.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&r.fence)),
                "Cannot create in-process fence");
        r.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!r.event) throw std::runtime_error("Cannot create in-process completion event");
        checked(r.adapter.Ensure(r.device.Get(), width_, height_, format_, true),
                "Cannot prepare same-device format conversion");
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width_; texture.Height = height_; texture.DepthOrArraySize = 1;
        texture.MipLevels = 1; texture.SampleDesc.Count = 1; texture.Format = format_;
        for (auto* output : {r.capture.GetAddressOf(), r.presentation.GetAddressOf()})
            checked(r.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(output)), "Cannot create same-device color texture");
        texture.Format = DXGI_FORMAT_R16G16_FLOAT;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        checked(r.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r.motion)), "Cannot create zero-motion input");
        D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
        descriptors.NumDescriptors = 1; descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        checked(r.device->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(&r.clear_cpu)), "Cannot create motion clear descriptor");
        descriptors.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        checked(r.device->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(&r.clear_gpu)), "Cannot create GPU motion descriptor");
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = texture.Format; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        r.device->CreateUnorderedAccessView(r.motion.Get(), nullptr, &view, r.clear_cpu->GetCPUDescriptorHandleForHeapStart());
        r.device->CopyDescriptorsSimple(1, r.clear_gpu->GetCPUDescriptorHandleForHeapStart(),
            r.clear_cpu->GetCPUDescriptorHandleForHeapStart(), descriptors.Type);
    }
    ~D3D12Bridge() {
        if (!retired_ && SUCCEEDED(resources_->device->GetDeviceRemovedReason())) {
            // A timeout is not proof the GPU stopped. Pin this failed generation
            // until process exit instead of freeing resources still in flight.
            resources_.release();
            return;
        }
        if (owner_ && in_process_api.release) in_process_api.release(runtime_);
    }
    D3D12Bridge(const D3D12Bridge&) = delete;
    bool safe_to_fallback() const { return retired_; }
    uint64_t delivered() const { return completed_; }
    uint64_t presented() const { return presented_; }
    bool held_last_frame() const { return held_last_frame_; }
    bool protect_presentation(reshade::api::resource back) {
        return retired_ && hold_completed(back, true);
    }
    void reset_history() { reset_ = true; deferred_.reset(); }
    bool matches(const reshade::api::resource_desc& desc) const {
        return width_ == desc.texture.width && height_ == desc.texture.height &&
               format_ == typed_format(desc.texture.format);
    }
    Json ui_snapshot() {
        const auto now = GetTickCount64();
        if (!ui_tick_ || now - ui_tick_ >= 1000) {
            ui_ = local_json(in_process_api.ui);
            ui_tick_ = now;
        }
        auto result = ui_;
        result["pipeline_status"] = pipeline_report_;
        result["frame_inputs"] = {{"with_engine_guides",frames_with_guides_}, {"without_engine_guides",frames_without_guides_},
            {"deferred",deferred_frames_}, {"held_completed",held_frames_}};
        if (in_process_api.guide_status) result["engine_guides"] = in_process_api.guide_status();
        return result;
    }
    bool action(phoenix_ui::Action action, unsigned argument) {
        if (action == phoenix_ui::Action::ResetHistory) reset_ = true;
        const bool accepted = in_process_api.action && in_process_api.action(uint32_t(action), argument);
        if (accepted) ui_tick_ = 0;
        return accepted;
    }
    bool frame(reshade::api::resource back, const Settings& requested, bool apply, bool coherent = true) {
        auto& r = *resources_;
        held_last_frame_ = false;
        const bool adaptive = requested.pipeline.monitoring != 0 && requested.pipeline.priority != 0;
        if (adaptive && (!adaptive_active_ || apply || requested.pipeline != adaptive_request_.pipeline ||
            requested.block_repeats != adaptive_request_.block_repeats ||
            requested.transformer_blocks != adaptive_request_.transformer_blocks ||
            requested.minimum_blocks != adaptive_request_.minimum_blocks ||
            requested.automatic_budget != adaptive_request_.automatic_budget ||
            requested.budget_mode != adaptive_request_.budget_mode ||
            requested.budget_ms != adaptive_request_.budget_ms ||
            requested.desired_fps != adaptive_request_.desired_fps ||
            requested.processing != adaptive_request_.processing || requested.refinement_passes != adaptive_request_.refinement_passes)) {
            adaptive_ = requested; adaptive_request_ = requested; latency_policy_.reset();
            // The pipeline governor is the single authority while active.
            adaptive_.automatic_budget = false;
        }
        adaptive_active_ = adaptive;
        const auto& settings = adaptive ? adaptive_ : requested;
        apply = apply || reconfigure_pipeline_; reconfigure_pipeline_ = false;
        pipeline_monitor_.configure(settings.pipeline);
        const bool monitored = pipeline_monitor_.enabled();
        const auto started = monitored ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool sampled = pipeline_monitor_.gpu_sample(completed_);
        if (sampled && !r.pipeline_queries) {
            D3D12_QUERY_HEAP_DESC description{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 6, 0};
            checked(r.device->CreateQueryHeap(&description, IID_PPV_ARGS(&r.pipeline_queries)), "Cannot allocate pipeline queries");
            checked(r.queue->GetTimestampFrequency(&r.timestamp_frequency), "Cannot read pipeline timestamp frequency");
            if (!r.timestamp_frequency) throw std::runtime_error("Invalid pipeline timestamp frequency");
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = 48; buffer.Height = 1; buffer.DepthOrArraySize = 1;
            buffer.MipLevels = 1; buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            checked(r.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r.pipeline_readback)), "Cannot allocate pipeline timing readback");
        }
        auto stamp = [&](unsigned index) {
            if (sampled) r.list->EndQuery(r.pipeline_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
        };
        if (!retired_) throw std::runtime_error("Previous in-process submission has not retired");
        if (!owner_ || apply) {
            if (!owner_ || helper_settings_changed(last_tuning_, settings)) presentation_.invalidate();
            deferred_.reset(); ui_tick_ = 0;
            in_process_api.prepare(runtime_, in_process_settings(settings));
            last_tuning_ = settings;
            owner_ = true;
        } else if (phoenix_addon::phoenix::copy_live_tuning(last_tuning_, settings)) {
            in_process_api.update_tuning(settings);
            ui_tick_ = 0;
        }
        // ReShade can have pending work on its immediate list. Submit that first;
        // this list then executes on the exact same queue, before final Present.
        runtime_->get_command_queue()->flush_immediate_command_list();
        if (!primed_) {
            begin(); submit(); // Establish observed list -> queue identity.
            primed_ = true;
        }
        const auto before = local_json(in_process_api.progress);
        begin();
        stamp(0);
        auto* image = reinterpret_cast<ID3D12Resource*>(uintptr_t(back.handle));
        copy(image, r.capture.Get());
        r.adapter.RecordInput(r.device.Get(), r.list.Get(), r.capture.Get());
        copy(r.adapter.Color(), r.adapter.Output());
        clear_motion();
        // Each borrowed input needs a fresh observed COMMON state on this epoch.
        for (auto* resource : {r.adapter.Output(), r.motion.Get()}) {
            auto barrier = transition(resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            r.list->ResourceBarrier(1, &barrier);
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
            r.list->ResourceBarrier(1, &barrier);
        }
        r.list->SetDescriptorHeaps(0, nullptr);
        PhoenixFeederFrameV1 frame;
        frame.command_list = r.list.Get(); frame.queue = r.queue.Get();
        frame.output = r.adapter.Output(); frame.motion = r.motion.Get();
        // This path synthesizes zero motion, not engine motion vectors. Match
        // the helper's spatial history policy rather than accumulating stale
        // screen-space history when switching between backends or moving.
        bool have_guides = false;
        struct GuideLease {
            const bool& retired;
            ID3D12Resource* motion = nullptr;
            ~GuideLease() {
                if (motion && retired && in_process_api.release_present_guides)
                    in_process_api.release_present_guides(motion);
            }
        } borrowed_guides{retired_};
        if (settings.processing.enabled && settings.processing.hook == 3 &&
            settings.processing.reuse_guides != 0 && in_process_api.present_guides)
            have_guides = in_process_api.present_guides(r.list.Get(),r.queue.Get(),frame);
        if (have_guides) borrowed_guides.motion=frame.motion;
        const bool have_engine_guides = have_guides;
        if (settings.processing.enabled && settings.processing.hook == 3 &&
            settings.processing.reuse_guides == 2 && !have_guides) {
            checked(r.list->Close(), "Cannot discard Present without required guides");
            reset_history(); return false;
        }
        stamp(1);
        frame.reset = backbuffer_requires_reset(have_guides, reset_) ? 1u : 0u;
#if defined(_WIN64)
        bool estimated = false;
        if (settings.pipeline.motion_enabled || settings.pipeline.scene_detection) {
            // Retire the captured color and any guide copies before the optical
            // flow engine consumes them on its own driver submission.
            submit();
            if (!r.estimated_motion)
                r.estimated_motion = std::make_unique<phoenix_pipeline::D3D12Motion>(
                    r.device.Get(), r.queue.Get(), r.fence.Get());
            phoenix_pipeline::D3D12Motion::Result motion;
            try {
                motion = r.estimated_motion->prepare(r.adapter.Color(), D3D12_RESOURCE_STATE_COMMON,
                    have_guides ? frame.motion : nullptr, settings.pipeline, reset_,
                    int64_t(GetTickCount64()) * 1000000);
            } catch (...) {
                in_process_gpu_uncertain = true;
                throw;
            }
            begin();
            frame.command_list = r.list.Get();
            // The flow preparation retired the previous list. Establish an
            // explicit binding epoch on this fresh list before feeder capture.
            r.list->SetDescriptorHeaps(0, nullptr);
            if (motion.source == phoenix_pipeline::MotionSource::Nvidia) {
                frame.motion = motion.motion; estimated = true; have_guides = true;
                auto barrier = transition(frame.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON);
                r.list->ResourceBarrier(1, &barrier);
            }
            // Fresh observed COMMON barriers belong to this new recording.
            for (auto* resource : {frame.output, frame.motion}) {
                auto barrier = transition(resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
                r.list->ResourceBarrier(1, &barrier);
                std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
                r.list->ResourceBarrier(1, &barrier);
            }
            frame.reset = (motion.reset || backbuffer_requires_reset(have_guides, reset_)) ? 1u : 0u;
        }
#endif
        stamp(2);
        PhoenixFeederResultV1 result;
        const auto status = in_process_api.evaluate(&frame, &result);
#if defined(_WIN64)
        if (estimated) {
            auto barrier = transition(frame.motion, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            r.list->ResourceBarrier(1, &barrier);
        }
#endif
        stamp(3);
        if (status == S_FALSE && result.status == PhoenixFeederStatusV1::Deferred) {
            checked(r.list->Close(), "Cannot discard deferred in-process recording");
            reset_ = true;
            ++deferred_frames_;
            if (deferred_.expired(GetTickCount64()))
                throw std::runtime_error("In-process evaluation remained deferred for ten seconds");
            return hold_completed(back, coherent);
        }
        if (status != S_OK || (result.status != PhoenixFeederStatusV1::Recorded &&
                              result.status != PhoenixFeederStatusV1::Bypass)) {
            r.list->Close(); // Abandon an unsubmitted failed recording, never execute it.
            throw std::runtime_error("In-process Phoenix DLSS5 evaluation refused: " +
                local_json(in_process_api.progress).value("direct_game", Json::object()).value("status", std::string()));
        }
        if (result.status == PhoenixFeederStatusV1::Bypass) {
            submit(); reset_history();
            // Feature setup can submit work without producing an image. Hold
            // completed output then, but honor an intentional Original/Off.
            if (settings.uplift && settings.mode != phoenix_addon::phoenix::input)
                return hold_completed(back, coherent);
            presentation_.invalidate();
            return false;
        }
        // Retire inference before reading its completion proof. Conversion and
        // final writeback only follow a proven, successfully processed frame.
        submit();
#if defined(_WIN64)
        if (estimated) r.estimated_motion->consumed(signal_);
#endif
        const auto after = local_json(in_process_api.progress);
        if (!PhoenixOutputAdvanced(before, after))
            throw std::runtime_error("In-process output lacks completed Phoenix DLSS5 execution proof");
        ++completed_;
        begin();
#if defined(_WIN64)
        if (estimated && settings.pipeline.motion_view) {
            auto barrier = transition(r.adapter.Output(), D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            r.list->ResourceBarrier(1, &barrier);
            r.estimated_motion->visualize(r.list.Get(), r.adapter.Output(), unsigned(settings.pipeline.motion_view));
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
            r.list->ResourceBarrier(1, &barrier);
        }
#endif
        r.adapter.RecordOutput(r.device.Get(), r.list.Get(), r.presentation.Get());
        stamp(4);
        if(r.screenshot_pair.output.pending) r.screenshot_pair.start(r.list.Get(),r.capture.Get(),D3D12_RESOURCE_STATE_COMMON,completed_,false,{{"route","in-process D3D12"}});
        r.screenshot_pair.finish(r.list.Get(),r.presentation.Get(),D3D12_RESOURCE_STATE_COMMON,signal_+1);
        copy(r.presentation.Get(), image);
        stamp(5);
        if (sampled) r.list->ResolveQueryData(r.pipeline_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
            0, 6, r.pipeline_readback.Get(), 0);
        submit();
        r.screenshot_pair.submitted(r.list.Get());
        r.screenshot_pair.poll(r.fence->GetCompletedValue());
        ++presented_;
        presentation_.completed();
        if (have_engine_guides) ++frames_with_guides_; else ++frames_without_guides_;
        deferred_.reset();
        reset_ = false;
        if (monitored) {
            const auto age = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            std::array<double, 5> stages{};
            if (sampled) {
                uint64_t* values = nullptr; D3D12_RANGE range{0, 48}, empty{};
                checked(r.pipeline_readback->Map(0, &range, reinterpret_cast<void**>(&values)), "Cannot read pipeline timings");
                for (unsigned i = 0; i != 5; ++i)
                    stages[i] = values[i + 1] >= values[i] ? double(values[i + 1] - values[i]) * 1000. / r.timestamp_frequency : 0;
                r.pipeline_readback->Unmap(0, &empty);
            }
            pipeline_monitor_.complete(stages, age);
            pipeline_report_ = {{"monitoring", settings.pipeline.monitoring}, {"capture_to_output_ms", age},
                {"stage_ms", stages}, {"sampled_gpu_stages", sampled}, {"completed", completed_},
                {"scope", "processing queue elapsed intervals including dependency waits; not input latency"}};
#if defined(_WIN64)
            if (r.estimated_motion) pipeline_report_["motion_provider"] = r.estimated_motion->status();
#endif
            if (adaptive) {
                auto coordinated_budget=settings.pipeline;
                if(requested.automatic_budget)coordinated_budget.latency_ms=std::min(coordinated_budget.latency_ms,float(requested.budget_target_ms()));
                pipeline_report_["budget_ms"]=coordinated_budget.latency_ms;
                const auto knob = latency_policy_.sample(coordinated_budget, age, false, {
                    settings.pipeline.motion_enabled && settings.pipeline.flow_height > settings.pipeline.minimum_flow_height,
                    settings.processing.enabled && settings.processing.scale > settings.pipeline.minimum_scale,
                    settings.mode == phoenix_addon::phoenix::ours && settings.experimental && settings.transformer_blocks > settings.minimum_blocks,
                    settings.refinement_passes > settings.pipeline.minimum_passes,
                    settings.pipeline.flow_height < adaptive_request_.pipeline.flow_height,
                    settings.processing.scale < adaptive_request_.processing.scale,
                    settings.transformer_blocks < adaptive_request_.transformer_blocks,
                    settings.refinement_passes < adaptive_request_.refinement_passes});
                if (knob == phoenix_pipeline::BudgetKnob::Flow)
                    adaptive_.pipeline.flow_height = std::max(adaptive_.pipeline.minimum_flow_height, adaptive_.pipeline.flow_height / 2);
                else if (knob == phoenix_pipeline::BudgetKnob::Passes) { --adaptive_.refinement_passes; reconfigure_pipeline_ = true; }
                else if (knob == phoenix_pipeline::BudgetKnob::Blocks) { --adaptive_.transformer_blocks; reconfigure_pipeline_ = true; }
                else if (knob == phoenix_pipeline::BudgetKnob::Scale) {
                    adaptive_.processing.scale = std::max(adaptive_.pipeline.minimum_scale, adaptive_.processing.scale - 10.f);
                    reconfigure_pipeline_ = true;
                }
                else if (knob == phoenix_pipeline::BudgetKnob::RestoreFlow)
                    adaptive_.pipeline.flow_height = std::min(adaptive_request_.pipeline.flow_height, adaptive_.pipeline.flow_height * 2);
                else if (knob == phoenix_pipeline::BudgetKnob::RestorePasses) { ++adaptive_.refinement_passes;reconfigure_pipeline_=true; }
                else if (knob == phoenix_pipeline::BudgetKnob::RestoreBlocks) { ++adaptive_.transformer_blocks;reconfigure_pipeline_=true; }
                else if (knob == phoenix_pipeline::BudgetKnob::RestoreScale) {
                    adaptive_.processing.scale=std::min(adaptive_request_.processing.scale,adaptive_.processing.scale+10.f);reconfigure_pipeline_=true;
                }
                pipeline_report_["last_adjustment"] = int(knob);
            }
        } else pipeline_report_ = {{"monitoring", 0}};
        return true;
    }
};
} // namespace phoenix_compat
