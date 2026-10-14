#pragma once
#include <cstdint>

// Own a reusable transfer resource without freeing or overwriting a resource
// whose GPU retirement is uncertain. The OS reclaims uncertain resources when
// the terminal helper process exits; its destructor deliberately leaves them.
template <class Resource> class PhoenixRetiredTransferBuffer {
    Resource* resource_ = nullptr;
    uint64_t capacity_ = 0;
    bool retired_ = true;

  public:
    PhoenixRetiredTransferBuffer() = default;
    PhoenixRetiredTransferBuffer(const PhoenixRetiredTransferBuffer&) = delete;
    ~PhoenixRetiredTransferBuffer() {
        release();
    }
    Resource* get() const {
        return resource_;
    }
    bool retired() const {
        return retired_;
    }
    template <class Allocate> bool ensure(uint64_t bytes, Allocate allocate) {
        if (!retired_ || !bytes)
            return false;
        if (resource_ && capacity_ >= bytes)
            return true;
        auto* next = allocate(bytes);
        if (!next)
            return false;
        release();
        resource_ = next;
        capacity_ = bytes;
        return true;
    }
    void submitted() {
        retired_ = false;
    }
    bool completed(bool proven) {
        retired_ = proven;
        return proven;
    }
    void release() {
        if (resource_ && retired_) {
            resource_->Release();
            resource_ = nullptr;
            capacity_ = 0;
        }
    }
};
