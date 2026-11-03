// A same-adapter GPU mailbox between the game's DX11 queue and our DX12 queue.
// There is no helper, pixel readback, shader pass, or per-frame allocation here.
#pragma once
#include "inprocess_api.h"
#include "local_d3d12_identity.h"
#include <d3d11_4.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

namespace phoenix_compat {
class D3D11LocalSurface {
    template<class T> using Com = Microsoft::WRL::ComPtr<T>;
    Com<ID3D11Device5> game_;
    Com<ID3D11DeviceContext4> context_;
    Com<ID3D12Device> wrapper_device_;
    Com<ID3D12CommandQueue> wrapper_queue_;
    Com<ID3D12Device> device_;
    Com<ID3D12CommandQueue> queue_;
    struct QueueRegistration {
        ID3D12CommandQueue* queue = nullptr;
        std::function<void(ID3D12CommandQueue*,bool)> notify;
        void acquire(ID3D12CommandQueue* value) {
            if (!in_process_api.local_queue_lifetime) return;
            notify = in_process_api.local_queue_lifetime;
            notify(value,true); queue = value;
        }
        ~QueueRegistration() {
            if (queue) try { notify(queue,false); }
            catch (...) { in_process_gpu_uncertain = true; }
        }
    } queue_registration_;
    Com<ID3D12Resource> image_;
    Com<ID3D11Texture2D> view_;
    Com<ID3D12Fence> fence_;
    Com<ID3D11Fence> game_fence_;
    HANDLE event_ = nullptr;
    uint64_t serial_ = 0;

    static void checked(HRESULT result, const char* operation) {
        if (FAILED(result)) {
            char code[32]{}; sprintf_s(code, " (0x%08X)", unsigned(result));
            throw std::runtime_error(std::string(operation) + code);
        }
    }
    struct OwnedHandle {
        HANDLE value = nullptr;
        ~OwnedHandle() { if (value) CloseHandle(value); }
    };
    // CopyResource is subject to game predication. Disable only that state and
    // restore it, without disturbing any bound shaders, views or render targets.
    struct UnconditionalCopy {
        ID3D11DeviceContext4* context;
        Com<ID3D11Predicate> predicate;
        BOOL condition = FALSE;
        explicit UnconditionalCopy(ID3D11DeviceContext4* value) : context(value) {
            context->GetPredication(&predicate, &condition);
            context->SetPredication(nullptr, FALSE);
        }
        ~UnconditionalCopy() { context->SetPredication(predicate.Get(), condition); }
    };
    void wait() {
        if (retired()) return;
        checked(fence_->SetEventOnCompletion(serial_, event_), "Cannot watch DX11 local transfer");
        if (WaitForSingleObject(event_, 5000) != WAIT_OBJECT_0 || !retired()) {
            in_process_gpu_uncertain = true;
            throw std::runtime_error("DX11 local GPU transfer did not retire; helper fallback is unsafe");
        }
    }
public:
    // An optional explicit device makes the exact interop code testable on WARP.
    // Production always creates DX12 on the adapter that owns the game device.
    D3D11LocalSurface(ID3D11Device* game, unsigned width, unsigned height,
                     DXGI_FORMAT format, ID3D12Device* compute = nullptr) {
        checked(game->QueryInterface(IID_PPV_ARGS(&game_)), "DX11 shared fences unavailable");
        Com<ID3D11DeviceContext> immediate; game->GetImmediateContext(&immediate);
        checked(immediate.As(&context_), "DX11 fence-capable context unavailable");
        Com<IDXGIDevice> dxgi; checked(game->QueryInterface(IID_PPV_ARGS(&dxgi)), "No game adapter");
        Com<IDXGIAdapter> adapter; checked(dxgi->GetAdapter(&adapter), "No game adapter identity");
        DXGI_ADAPTER_DESC identity{}; checked(adapter->GetDesc(&identity), "No adapter LUID");
        if (compute) device_ = compute;
        else checked(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device_)), "Cannot create same-adapter DX12 compute device");
        const auto luid = device_->GetAdapterLuid();
        if (luid.LowPart != identity.AdapterLuid.LowPart || luid.HighPart != identity.AdapterLuid.HighPart)
            throw std::runtime_error("DX11 and DX12 local devices belong to different adapters");
        D3D12_COMMAND_QUEUE_DESC queue{}; queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        checked(device_->CreateCommandQueue(&queue, IID_PPV_ARGS(&queue_)), "Cannot create local DX12 queue");
        resolve_local_d3d12_identity(device_,queue_,wrapper_device_,wrapper_queue_);
        // Balance even constructor failure and queues with no ReShade lifetime
        // event. Declare this lease after queue_ so that it retires first.
        queue_registration_.acquire(queue_.Get());
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{}; texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width; texture.Height = height; texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = format; texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        checked(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &texture,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&image_)), "Cannot create local shared color");
        OwnedHandle color;
        checked(device_->CreateSharedHandle(image_.Get(), nullptr, GENERIC_ALL, nullptr, &color.value),
            "Cannot share local color");
        checked(game_->OpenSharedResource1(color.value, IID_PPV_ARGS(&view_)), "Cannot open local color in DX11");
        checked(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)), "Cannot create local transfer fence");
        OwnedHandle fence;
        checked(device_->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, &fence.value),
            "Cannot share local transfer fence");
        checked(game_->OpenSharedFence(fence.value, IID_PPV_ARGS(&game_fence_)), "Cannot open local fence in DX11");
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) throw std::runtime_error("Cannot create local transfer event");
    }
    ~D3D11LocalSurface() { if (event_) CloseHandle(event_); }
    D3D11LocalSurface(const D3D11LocalSurface&) = delete;
    ID3D12Device* device() const { return device_.Get(); }
    ID3D12CommandQueue* queue() const { return queue_.Get(); }
    ID3D12Resource* image() const { return image_.Get(); }
    bool retired() const {
        const auto completed = fence_->GetCompletedValue();
        return !serial_ || (completed != UINT64_MAX && completed >= serial_);
    }
    void retire() { wait(); }
    void capture(ID3D11Texture2D* back) {
        UnconditionalCopy state(context_.Get());
        context_->CopyResource(view_.Get(), back);
        checked(context_->Signal(game_fence_.Get(), ++serial_), "Cannot signal DX11 local input");
        context_->Flush();
        // GPU ordering, not CPU pixel transport. The native bridge's bounded
        // completion wait covers this queue dependency as well as inference.
        checked(queue_->Wait(fence_.Get(), serial_), "Cannot order local DX12 input");
    }
    void publish(ID3D11Texture2D* back) {
        checked(queue_->Signal(fence_.Get(), ++serial_), "Cannot signal local DX12 output");
        checked(context_->Wait(game_fence_.Get(), serial_), "Cannot order DX11 local output");
        UnconditionalCopy state(context_.Get());
        context_->CopyResource(back, view_.Get());
        checked(context_->Signal(game_fence_.Get(), ++serial_), "Cannot retire DX11 local output");
        context_->Flush();
        // Retire before returning: resizing/disabling the add-on immediately
        // after Present must not release a shared allocation still being used.
        wait();
    }
};
} // namespace phoenix_compat
