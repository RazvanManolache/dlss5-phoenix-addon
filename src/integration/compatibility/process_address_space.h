#pragma once
#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace phoenix_compat {
struct ProcessAddressSpace {
    uint64_t free_bytes = 0, largest_free_region = 0;
    bool queried = false;
};

// These are startup safety margins, not a prediction of steady-state RAM.
// Graphics-driver initialization can transiently consume hundreds of MiB in
// the caller's low address space, even when the neural library itself is x64.
// Keep room for both the driver and the game's next texture allocation.
constexpr bool in_process_address_space_safe(unsigned bits, ProcessAddressSpace space,
                                              uint64_t minimum_region = 64 * 1024 * 1024) {
    constexpr uint64_t mib = 1024 * 1024;
    return bits != 32 || (space.queried && space.free_bytes >= 512 * mib &&
                         space.largest_free_region >= minimum_region);
}

inline ProcessAddressSpace process_address_space() {
    ProcessAddressSpace result;
#ifndef _WIN64
    uint64_t address = 0;
    constexpr uint64_t limit = uint64_t(UINT32_MAX) + 1;
    while (address < limit) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(uintptr_t(address)), &region,
                         sizeof(region)) != sizeof(region) || !region.RegionSize) break;
        result.queried = true;
        const auto end = std::min(limit, uint64_t(uintptr_t(region.BaseAddress)) + region.RegionSize);
        if (end <= address) break;
        if (region.State == MEM_FREE) {
            const auto bytes = end - address;
            result.free_bytes += bytes;
            result.largest_free_region = std::max(result.largest_free_region, bytes);
        }
        address = end;
    }
#endif
    return result;
}

inline std::string in_process_address_space_reason(ProcessAddressSpace space) {
    return "Low 32-bit address space: " + std::to_string(space.free_bytes / (1024 * 1024)) +
        " MiB free, largest region " + std::to_string(space.largest_free_region / (1024 * 1024)) +
        " MiB. Use the 64-bit helper or reduce game memory use and retry.";
}

inline void require_in_process_address_space(uint64_t minimum_region = 64 * 1024 * 1024) {
#ifndef _WIN64
    const auto space = process_address_space();
    if (!in_process_address_space_safe(32, space, minimum_region))
        throw std::runtime_error(in_process_address_space_reason(space));
#endif
}
} // namespace phoenix_compat
