// One retirement value describes all copies in an immediate-queue batch.
#pragma once
#include <cstdint>

namespace phoenix_compat {
inline uint64_t reserve_retirement_serial(uint64_t& serial, uint64_t& queued_signal) {
    // DX9 emulates a timeline with a ring of event queries and scans consecutive
    // values. Skipping a value can inspect an older completed query in that slot
    // and report progress that was never signalled. Upload and capture copies
    // queued before the same signal must therefore share its retirement value.
    if (!queued_signal)
        queued_signal = ++serial;
    return queued_signal;
}
} // namespace phoenix_compat
