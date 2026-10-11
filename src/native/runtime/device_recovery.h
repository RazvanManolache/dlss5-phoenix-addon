#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace phoenix_device {
using Microsoft::WRL::ComPtr;
enum class Status { complete, pending, removed, stale, timeout };
enum class State { empty, ready, lost };
struct Extent {
    uint32_t width, height;
    void validate() const {
        if (!width || !height || width > 16384 || height > 16384)
            throw std::invalid_argument(
                "Recovery37 invalid extent; suspend zero-size windows externally");
    }
};
struct Ticket {
    uint64_t generation = 0, serial = 0, fence = 0;
    uint32_t slot = 0;
};
struct Frame {
    Ticket ticket;
    ID3D12GraphicsCommandList* commands = nullptr;
};
struct Payload {
    virtual ~Payload() = default;
};
// Rebuild all owned resources, descriptors, pipelines/modules and history here.
// The factory must not retain device objects elsewhere or submit GPU work.
using Factory = std::function<std::unique_ptr<Payload>(ID3D12Device*, Extent, uint64_t)>;

// Engine-owned-device integration. Each submitted frame must retain its Lease
// until the owning queue fence completes (or actual device loss is observed).
// No GPU submission, waits, RemoveDevice or D3D12CreateDevice happen here.
class BorrowedBinding {
  public:
    struct Epoch {
        const uint64_t generation;
        ComPtr<ID3D12Device> device;
        std::unique_ptr<Payload> resources;
        std::atomic<bool> valid{true}, history_reset{true};
        Epoch(uint64_t generation, ID3D12Device* device, std::unique_ptr<Payload> payload)
            : generation(generation), device(device), resources(std::move(payload)) {}
    };
    using Lease = std::shared_ptr<Epoch>;
    using Rebuild = std::function<std::unique_ptr<Payload>(ID3D12Device*, uint64_t)>;

  private:
    mutable std::mutex mutex_;
    Lease current_;
    uint64_t generation_ = 0;
    bool building_ = false;
    HRESULT reason_ = S_OK;
    static bool same_device(ID3D12Device* left_device, ID3D12Device* right_device) {
        if (!left_device || !right_device)
            return false;
        ComPtr<IUnknown> x, y;
        return SUCCEEDED(left_device->QueryInterface(IID_PPV_ARGS(&x))) &&
               SUCCEEDED(right_device->QueryInterface(IID_PPV_ARGS(&y))) && x.Get() == y.Get();
    }

  public:
    BorrowedBinding() = default;
    BorrowedBinding(const BorrowedBinding&) = delete;
    BorrowedBinding& operator=(const BorrowedBinding&) = delete;
    ~BorrowedBinding() {
        invalidate(E_ABORT);
    }
    // The host must externally stop new submissions first. Old GPU references
    // remain alive through outstanding frame Leases, not a guessed idle wait.
    void invalidate(HRESULT reason) {
        Lease retired;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (current_) {
                current_->valid.store(false, std::memory_order_release);
                current_->history_reset.store(true, std::memory_order_release);
            }
            retired = std::move(current_);
            ++generation_;
            reason_ = reason;
        }
        // Payload destruction can re-enter descriptor/resource callbacks.
        // It must never happen while the binding mutex is held.
    }
    Lease rebind(ID3D12Device* device, const Rebuild& rebuild) {
        if (!device || !rebuild)
            throw std::invalid_argument("Recovery37 null borrowed binding");
        if (FAILED(device->GetDeviceRemovedReason()))
            throw std::runtime_error("Recovery37 cannot bind removed engine device");
        uint64_t generation;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (current_ || building_)
                throw std::logic_error("Recovery37 invalidate old binding before explicit rebind");
            building_ = true;
            generation = ++generation_;
        }
        Lease next;
        try {
            auto payload = rebuild(device, generation);
            if (!payload)
                throw std::runtime_error("Recovery37 borrowed rebuild returned null");
            next = std::make_shared<Epoch>(generation, device, std::move(payload));
            if (FAILED(device->GetDeviceRemovedReason()))
                throw std::runtime_error("Recovery37 engine device removed during rebuild");
        } catch (...) {
            std::lock_guard<std::mutex> guard(mutex_);
            building_ = false;
            throw;
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            building_ = false;
            if (generation != generation_) {
                next->valid.store(false, std::memory_order_release);
                throw std::runtime_error("Recovery37 borrowed rebuild cancelled by invalidation");
            }
            current_ = next;
            reason_ = S_OK;
        }
        return next;
    }
    Lease acquire(ID3D12Device* device) {
        Lease lease;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            lease = current_;
        }
        if (!lease || !same_device(lease->device.Get(), device) ||
            !lease->valid.load(std::memory_order_acquire))
            return {};
        if (FAILED(device->GetDeviceRemovedReason())) {
            observe(device, DXGI_ERROR_DEVICE_REMOVED);
            return {};
        }
        return lease;
    }
    bool observe(ID3D12Device* device, HRESULT operation_result, uint64_t completed_fence = 0) {
        Lease lease;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            lease = current_;
        }
        if (!lease || !same_device(lease->device.Get(), device))
            return false;
        auto reason = device->GetDeviceRemovedReason();
        bool removed = FAILED(reason) || completed_fence == UINT64_MAX ||
                       operation_result == DXGI_ERROR_DEVICE_REMOVED ||
                       operation_result == DXGI_ERROR_DEVICE_RESET ||
                       operation_result == DXGI_ERROR_DEVICE_HUNG ||
                       operation_result == DXGI_ERROR_DRIVER_INTERNAL_ERROR;
        if (!removed)
            return false;
        Lease retired;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (current_ != lease)
                return false;
            lease->valid.store(false, std::memory_order_release);
            lease->history_reset.store(true, std::memory_order_release);
            retired = std::move(current_);
            ++generation_;
            reason_ = FAILED(reason)             ? reason
                      : FAILED(operation_result) ? operation_result
                                                 : DXGI_ERROR_DEVICE_REMOVED;
        }
        return true;
    }
    bool dispatch_allowed(const Lease& lease) const {
        if (!lease || !lease->valid.load(std::memory_order_acquire))
            return false;
        std::lock_guard<std::mutex> guard(mutex_);
        return current_ == lease;
    }
    HRESULT reason() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return reason_;
    }
};

// Single-owner-thread D3D12 DIRECT queue. Not an engine device replacement:
// external NGX/engine owners must separately release/recreate their objects.
// No automatic fallback to a different adapter, WARP, stale history or old ticket.
class Owner {
    struct Slot {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> commands;
        uint64_t fence = 0;
    };
    std::thread::id thread_ = std::this_thread::get_id();
    State state_ = State::empty;
    Extent extent_{};
    LUID adapter_id_{};
    uint64_t generation_ = 0, serial_ = 0, submitted_ = 0;
    uint32_t cursor_ = 0;
    bool recording_ = false, history_reset_ = true;
    HRESULT removal_reason_ = S_OK;
    Factory factory_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    std::array<Slot, 3> slots_;
    std::unique_ptr<Payload> payload_;
    std::vector<std::pair<HANDLE, uint64_t>> wait_events_;
    void thread_check() const;
    void check(HRESULT result);
    Status health();
    void create();
    void release() noexcept;
    void require_frame(const Frame& frame) const;

  public:
    Owner() = default;
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner();
    void initialize(LUID adapter, Extent extent, Factory factory);
    State state() const noexcept {
        return state_;
    }
    uint64_t generation() const noexcept {
        return generation_;
    }
    HRESULT removal_reason() const noexcept {
        return removal_reason_;
    }
    bool history_reset_required() const noexcept {
        return history_reset_;
    }
    void acknowledge_history_reset(uint64_t generation);
    ID3D12Device* device() const noexcept {
        return device_.Get();
    }
    ID3D12CommandQueue* queue() const noexcept {
        return queue_.Get();
    }
    Payload* payload() const noexcept {
        return payload_.get();
    }
    // begin() never waits: a still-owned ring slot produces pending via try_begin.
    Status try_begin(Frame& frame);
    Ticket submit(const Frame& frame);
    void cancel(const Frame& frame);
    Status poll(Ticket ticket);
    Status wait(Ticket ticket, DWORD timeout_ms);
    Status drain(DWORD timeout_ms);
    Status resize(Extent extent, DWORD timeout_ms);
    // Only allowed after an observed removal. Rebuild on the same adapter.
    void recover();
    Status shutdown(DWORD timeout_ms);
};
}
