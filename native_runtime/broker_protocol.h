// Fixed-width shared-memory protocol for addon/broker communication.
// Member order is a wire ABI; rename fields without changing their sizes, alignment or buffer order.

#pragma once
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

namespace phoenix_cpp {
constexpr uint32_t broker_magic = 0x50485843, broker_version = 1;
constexpr size_t text_capacity = 65536;
// Fixed-width wire data only; no pointers, STL objects or architecture-sized
// fields. Compatible between a future x86 host and this x64 broker.
struct alignas(64) BrokerHeader {
    uint32_t magic = broker_magic, version = broker_version, width = 0, height = 0;
    uint64_t bytes = 0, sequence = 0;
    uint32_t parent = 0, ready = 0;
    int32_t result = 0;
    uint32_t reserved = 0;
    char config[text_capacity]{}, metadata[text_capacity]{}, status[text_capacity]{};
};
inline uint64_t broker_bytes(unsigned width, unsigned height) {
    if (!width || !height || width > 8192 || height > 8192)
        throw std::runtime_error("Broker extent invalid");
    return sizeof(BrokerHeader) + uint64_t(width) * height * 10 * sizeof(float);
}
inline void broker_text(char* dst, const std::string& value) {
    if (value.size() >= text_capacity)
        throw std::runtime_error("Broker JSON capacity exceeded");
    std::memcpy(dst, value.c_str(), value.size() + 1);
}
struct Handle {
    HANDLE value = nullptr;
    Handle() = default;
    explicit Handle(HANDLE height) : value(height) {}
    Handle(const Handle&) = delete;
    ~Handle() {
        if (value && value != INVALID_HANDLE_VALUE)
            CloseHandle(value);
    }
    operator HANDLE() const {
        return value;
    }
};
inline HANDLE require_handle(HANDLE height) {
    if (!height || height == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Broker Windows handle failure");
    return height;
}
}
