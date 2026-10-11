#include "device_recovery.h"
#include <limits>
#include <string>

namespace phoenix_device {
void Owner::thread_check() const {
    if (thread_ != std::this_thread::get_id())
        throw std::logic_error("Recovery37 wrong owner thread");
}
void Owner::check(HRESULT result) {
    if (FAILED(result)) {
        if (device_ && FAILED(device_->GetDeviceRemovedReason())) {
            removal_reason_ = device_->GetDeviceRemovedReason();
            state_ = State::lost;
            history_reset_ = true;
        }
        throw std::runtime_error("Recovery37 HRESULT " + std::to_string(uint32_t(result)));
    }
}
Status Owner::health() {
    if (state_ == State::lost)
        return Status::removed;
    if (state_ != State::ready)
        return Status::stale;
    auto reason = device_->GetDeviceRemovedReason();
    if (FAILED(reason) || fence_->GetCompletedValue() == UINT64_MAX) {
        removal_reason_ = FAILED(reason) ? reason : DXGI_ERROR_DEVICE_REMOVED;
        state_ = State::lost;
        history_reset_ = true;
        return Status::removed;
    }
    const auto completed = fence_->GetCompletedValue();
    for (auto lookup_iterator = wait_events_.begin(); lookup_iterator != wait_events_.end();) {
        if (completed >= lookup_iterator->second) {
            CloseHandle(lookup_iterator->first);
            lookup_iterator = wait_events_.erase(lookup_iterator);
        } else
            ++lookup_iterator;
    }
    return Status::complete;
}
void Owner::create() {
    ++generation_;
    history_reset_ = true;
    ComPtr<IDXGIFactory4> dxgi;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)));
    ComPtr<IDXGIAdapter1> adapter;
    check(dxgi->EnumAdapterByLuid(adapter_id_, IID_PPV_ARGS(&adapter)));
    check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)));
    D3D12_COMMAND_QUEUE_DESC desc{};
    check(device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue_)));
    check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
    for (auto& slot : slots_) {
        check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              IID_PPV_ARGS(&slot.allocator)));
        check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.allocator.Get(),
                                         nullptr, IID_PPV_ARGS(&slot.commands)));
        check(slot.commands->Close());
    }
    payload_ = factory_(device_.Get(), extent_, generation_);
    if (!payload_)
        throw std::runtime_error("Recovery37 resource factory returned null");
    submitted_ = 0;
    cursor_ = 0;
    recording_ = false;
    state_ = State::ready;
}
void Owner::release() noexcept {
    for (auto item : wait_events_)
        CloseHandle(item.first);
    wait_events_.clear();
    payload_.reset();
    for (auto& slot : slots_) {
        slot.commands.Reset();
        slot.allocator.Reset();
        slot.fence = 0;
    }
    fence_.Reset();
    queue_.Reset();
    device_.Reset();
    recording_ = false;
}
Owner::~Owner() {
    // Explicit shutdown(timeout) allows a caller to defer destruction. Destruction
    // itself cannot release live GPU references on an arbitrary timeout.
    if (state_ == State::ready) {
        try {
            drain(INFINITE);
        } catch (...) {
            std::terminate();
        }
    }
    release();
}
void Owner::initialize(LUID adapter, Extent extent, Factory factory) {
    thread_check();
    extent.validate();
    if (state_ != State::empty || !factory)
        throw std::logic_error("Recovery37 invalid initialize");
    adapter_id_ = adapter;
    extent_ = extent;
    factory_ = std::move(factory);
    try {
        create();
    } catch (...) {
        release();
        state_ = State::empty;
        throw;
    }
}
void Owner::acknowledge_history_reset(uint64_t generation) {
    thread_check();
    if (generation != generation_ || health() != Status::complete)
        throw std::logic_error("Recovery37 stale history acknowledgement");
    history_reset_ = false;
}
Status Owner::try_begin(Frame& frame) {
    thread_check();
    frame = {};
    auto device_health = health();
    if (device_health != Status::complete)
        return device_health;
    if (recording_)
        throw std::logic_error("Recovery37 frame already recording");
    auto& slot = slots_[cursor_];
    if (fence_->GetCompletedValue() < slot.fence)
        return Status::pending;
    check(slot.allocator->Reset());
    check(slot.commands->Reset(slot.allocator.Get(), nullptr));
    if (++serial_ == 0)
        throw std::overflow_error("Recovery37 serial overflow");
    frame = {{generation_, serial_, 0, cursor_}, slot.commands.Get()};
    recording_ = true;
    return Status::complete;
}
void Owner::require_frame(const Frame& frame) const {
    if (!recording_ || state_ != State::ready || frame.ticket.generation != generation_ ||
        frame.ticket.serial != serial_ || frame.ticket.slot != cursor_ || frame.ticket.fence ||
        frame.commands != slots_[cursor_].commands.Get())
        throw std::logic_error("Recovery37 stale frame");
}
Ticket Owner::submit(const Frame& frame) {
    thread_check();
    require_frame(frame);
    if (health() != Status::complete)
        throw std::runtime_error("Recovery37 submit on removed device");
    if (submitted_ == UINT64_MAX - 1)
        throw std::overflow_error("Recovery37 fence exhausted");
    auto& slot = slots_[cursor_];
    check(slot.commands->Close());
    recording_ = false;
    ID3D12CommandList* list = slot.commands.Get();
    queue_->ExecuteCommandLists(1, &list);
    const uint64_t value = ++submitted_;
    check(queue_->Signal(fence_.Get(), value));
    slot.fence = value;
    Ticket result{generation_, frame.ticket.serial, value, cursor_};
    cursor_ = (cursor_ + 1) % uint32_t(slots_.size());
    return result;
}
void Owner::cancel(const Frame& frame) {
    thread_check();
    require_frame(frame);
    check(slots_[cursor_].commands->Close());
    recording_ = false;
}
Status Owner::poll(Ticket ticket) {
    thread_check();
    if (!ticket.generation || ticket.generation != generation_ || !ticket.fence ||
        ticket.fence > submitted_ || !ticket.serial || ticket.serial > serial_ ||
        ticket.slot >= slots_.size())
        return Status::stale;
    auto device_health = health();
    if (device_health != Status::complete)
        return device_health;
    // UINT64_MAX means loss, never successful completion of old image/history.
    auto completed = fence_->GetCompletedValue();
    if (completed == UINT64_MAX)
        return health();
    return completed >= ticket.fence ? Status::complete : Status::pending;
}
Status Owner::wait(Ticket ticket, DWORD timeout_ms) {
    thread_check();
    auto status = poll(ticket);
    if (status != Status::pending)
        return status;
    if (!timeout_ms)
        return Status::timeout;
    // A manual-reset event is local to this wait and cannot wake another waiter.
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!event)
        throw std::runtime_error("Recovery37 CreateEvent failed");
    HRESULT result = fence_->SetEventOnCompletion(ticket.fence, event);
    if (FAILED(result)) {
        CloseHandle(event);
        if (health() == Status::removed)
            return Status::removed;
        check(result);
    }
    DWORD waited = WaitForSingleObject(event, timeout_ms);
    status = poll(ticket);
    if (status != Status::pending) {
        CloseHandle(event);
        return status;
    }
    // Preserve a registered event after timeout until its fence completes.
    // Reserve before registration would also be valid; here allocation failure
    // waits safely before closing rather than dropping a live registration.
    try {
        wait_events_.emplace_back(event, ticket.fence);
    } catch (...) {
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        throw;
    }
    if (waited == WAIT_TIMEOUT)
        return Status::timeout;
    throw std::runtime_error("Recovery37 fence wait failed");
}
Status Owner::drain(DWORD timeout_ms) {
    thread_check();
    auto device_health = health();
    if (device_health != Status::complete)
        return device_health;
    if (!submitted_)
        return Status::complete;
    return wait({generation_, serial_, submitted_, 0}, timeout_ms);
}
Status Owner::resize(Extent extent, DWORD timeout_ms) {
    thread_check();
    extent.validate();
    if (recording_)
        throw std::logic_error("Recovery37 resize during recording");
    auto status = drain(timeout_ms);
    if (status != Status::complete)
        return status;
    if (extent.width == extent_.width && extent.height == extent_.height)
        return Status::complete;
    // Build first: an allocation failure preserves the old extent and payload.
    auto replacement = factory_(device_.Get(), extent, generation_ + 1);
    if (!replacement)
        throw std::runtime_error("Recovery37 resize factory returned null");
    payload_.swap(replacement);
    extent_ = extent;
    ++generation_;
    history_reset_ = true;
    return Status::complete;
}
void Owner::recover() {
    thread_check();
    if (health() != Status::removed)
        throw std::logic_error("Recovery37 no observed device removal");
    release();
    try {
        create();
    } catch (...) {
        release();
        state_ = State::lost;
        throw;
    }
}
Status Owner::shutdown(DWORD timeout_ms) {
    thread_check();
    if (state_ == State::empty)
        return Status::complete;
    if (recording_)
        throw std::logic_error("Recovery37 shutdown during recording");
    auto status = drain(timeout_ms);
    if (status != Status::complete && status != Status::removed)
        return status;
    release();
    state_ = State::empty;
    ++generation_;
    history_reset_ = true;
    return status;
}
}
