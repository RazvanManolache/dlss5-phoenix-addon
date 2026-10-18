#pragma once
#include "inprocess_api.h"
#include "session_policy.h"
#include "host/phoenix-format-adapter.h"
#include "host/phoenix-host-config.h"
#include "host/phoenix-host-safety.h"
#include <wrl/client.h>

namespace phoenix_compat {
// This route borrows the game's DIRECT queue. It owns its command list and all
// conversion/model inputs; no CPU pixel transport or fake SR feature is used.
class D3D12Bridge {
    template<class T> using Com = Microsoft::WRL::ComPtr<T>;
    struct Resources {
        Com<ID3D12Device> device;
        Com<ID3D12CommandQueue> queue;
        Com<ID3D12CommandAllocator> allocator;
        Com<ID3D12GraphicsCommandList> list;
        Com<ID3D12Fence> fence;
        Com<ID3D12Resource> capture, presentation, motion;
        Com<ID3D12DescriptorHeap> clear_cpu, clear_gpu;
        PhoenixFormatAdapter adapter;
        void (*forget)(ID3D12GraphicsCommandList*) noexcept = nullptr;
        HANDLE event = nullptr;
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
    uint64_t ui_tick_ = 0;
    DeferredTimeout deferred_;

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
    static DXGI_FORMAT typed_format(reshade::api::format format) {
        using F = reshade::api::format;
        if (format == F::r8g8b8a8_unorm || format == F::r8g8b8a8_unorm_srgb || format == F::r8g8b8a8_typeless)
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        if (format == F::b8g8r8a8_unorm || format == F::b8g8r8a8_unorm_srgb || format == F::b8g8r8a8_typeless)
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        if (format == F::r10g10b10a2_unorm || format == F::r10g10b10a2_typeless)
            return DXGI_FORMAT_R10G10B10A2_UNORM;
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
        return ui_;
    }
    bool action(phoenix_ui::Action action, unsigned argument) {
        if (action == phoenix_ui::Action::ResetHistory) reset_ = true;
        const bool accepted = in_process_api.action && in_process_api.action(uint32_t(action), argument);
        if (accepted) ui_tick_ = 0;
        return accepted;
    }
    bool frame(reshade::api::resource back, const Settings& settings, bool apply) {
        auto& r = *resources_;
        if (!retired_) throw std::runtime_error("Previous in-process submission has not retired");
        if (!owner_ || apply) {
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
        frame.reset = backbuffer_requires_reset(false, reset_) ? 1u : 0u;
        PhoenixFeederResultV1 result;
        const auto status = in_process_api.evaluate(&frame, &result);
        if (status == S_FALSE && result.status == PhoenixFeederStatusV1::Deferred) {
            checked(r.list->Close(), "Cannot discard deferred in-process recording");
            reset_ = true;
            if (deferred_.expired(GetTickCount64()))
                throw std::runtime_error("In-process evaluation remained deferred for ten seconds");
            return false;
        }
        if (status != S_OK || (result.status != PhoenixFeederStatusV1::Recorded &&
                              result.status != PhoenixFeederStatusV1::Bypass)) {
            r.list->Close(); // Abandon an unsubmitted failed recording, never execute it.
            throw std::runtime_error("In-process Phoenix DLSS5 evaluation refused: " +
                local_json(in_process_api.progress).value("direct_game", Json::object()).value("status", std::string()));
        }
        if (result.status == PhoenixFeederStatusV1::Bypass) { submit(); reset_history(); return false; }
        // Retire inference before reading its completion proof. Conversion and
        // final writeback only follow a proven, successfully processed frame.
        submit();
        const auto after = local_json(in_process_api.progress);
        if (!PhoenixOutputAdvanced(before, after))
            throw std::runtime_error("In-process output lacks completed Phoenix DLSS5 execution proof");
        ++completed_;
        begin();
        r.adapter.RecordOutput(r.device.Get(), r.list.Get(), r.presentation.Get());
        copy(r.presentation.Get(), image);
        submit();
        ++presented_;
        deferred_.reset();
        reset_ = false;
        return true;
    }
};
} // namespace phoenix_compat
