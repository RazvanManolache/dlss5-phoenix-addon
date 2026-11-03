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
    uint64_t address_limit = 0;
};

constexpr uint64_t caller_address_limit(bool large_address_aware, uint64_t system_limit) {
    // A non-LAA x86 game cannot allocate the otherwise-free upper 2 GiB.
    // Native x64 code sharing its process does not extend the game's pointers.
    return std::min(system_limit, large_address_aware ? uint64_t(1)<<32 : uint64_t(1)<<31);
}

constexpr void count_free_address_region(ProcessAddressSpace& space, uint64_t begin,
                                         uint64_t bytes, uint64_t limit) {
    if (begin >= limit) return;
    bytes = std::min(bytes, limit-begin);
    space.free_bytes += bytes;
    space.largest_free_region = std::max(space.largest_free_region, bytes);
}

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
    const auto image = reinterpret_cast<const unsigned char*>(GetModuleHandleW(nullptr));
    const auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
    const auto pe = reinterpret_cast<const IMAGE_NT_HEADERS32*>(image+dos->e_lfanew);
    SYSTEM_INFO system{}; GetSystemInfo(&system);
    const auto limit = caller_address_limit(
        (pe->FileHeader.Characteristics & IMAGE_FILE_LARGE_ADDRESS_AWARE) != 0,
        uint64_t(uintptr_t(system.lpMaximumApplicationAddress))+1);
    result.address_limit = limit;
    while (address < limit) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(uintptr_t(address)), &region,
                         sizeof(region)) != sizeof(region) || !region.RegionSize) break;
        const auto end = std::min(limit, uint64_t(uintptr_t(region.BaseAddress)) + region.RegionSize);
        if (end <= address) break;
        if (region.State == MEM_FREE) {
            count_free_address_region(result, address, end-address, limit);
        }
        address = end;
    }
    result.queried = limit != 0 && address == limit;
#endif
    return result;
}

inline std::string in_process_address_space_reason(ProcessAddressSpace space) {
    return "Low 32-bit address space: " + std::to_string(space.free_bytes / (1024 * 1024)) +
        " MiB free, largest region " + std::to_string(space.largest_free_region / (1024 * 1024)) +
        " MiB, caller ceiling " + std::to_string(space.address_limit / (1024 * 1024)) +
        " MiB. Use the 64-bit helper or reduce game memory use and retry.";
}

inline ProcessAddressSpace require_in_process_address_space(uint64_t minimum_region = 64 * 1024 * 1024) {
#ifndef _WIN64
    const auto space = process_address_space();
    if (!in_process_address_space_safe(32, space, minimum_region))
        throw std::runtime_error(in_process_address_space_reason(space));
    return space;
#else
    return {};
#endif
}
} // namespace phoenix_compat
