#pragma once
#include <algorithm>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

namespace phoenix_threads {
inline unsigned available() noexcept {
#ifdef _WIN32
    const auto count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (count)
        return count;
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}
// Zero means automatic. Clamp saved settings when moved to a smaller CPU.
inline unsigned resolve(unsigned requested) noexcept {
    const auto count = available();
    return requested ? std::min(requested, count) : count;
}
}
