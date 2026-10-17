#pragma once
#include <cstdint>
#include <cstddef>
#if UINTPTR_MAX == UINT32_MAX
#include <stdexcept>
#endif

namespace phx {
#if UINTPTR_MAX == UINT32_MAX
// Device handles and graph placeholders are wider than host pointers on x86.
// Conversion to a real host pointer must never silently truncate those handles.
struct alignas(8) OperandAddress {
    uint64_t value = 0;
    OperandAddress() = default;
    OperandAddress(std::nullptr_t) {}
    template <class T> OperandAddress(T* pointer) : value(uintptr_t(pointer)) {}
    template <class T> operator T*() const {
        if (value > UINTPTR_MAX)
            throw std::overflow_error("Logical device address is not a host pointer");
        return reinterpret_cast<T*>(uintptr_t(value));
    }
    explicit operator bool() const { return value != 0; }
    bool operator==(const OperandAddress&) const = default;
};
inline uint64_t address_bits(OperandAddress address) { return address.value; }
inline OperandAddress address_from_bits(uint64_t value) {
    OperandAddress result;
    result.value = value;
    return result;
}
#else
// Retain the established native pointer representation and layout on x64/ARM64.
using OperandAddress = void*;
inline uint64_t address_bits(OperandAddress address) { return uintptr_t(address); }
inline OperandAddress address_from_bits(uint64_t value) {
    return reinterpret_cast<void*>(uintptr_t(value));
}
#endif
}
