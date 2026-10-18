#pragma once
#include <cstdint>
namespace phoenix_addon::phoenix {
// Only a synchronous feeder may set this scope: its input textures stay
// unchanged until it receives the finished output. Engine frames leave it zero.
inline thread_local uint64_t locked_capture_token = 0;
struct LockedCaptureScope {
    uint64_t previous;
    explicit LockedCaptureScope(uint64_t token) : previous(locked_capture_token) { locked_capture_token=token; }
    ~LockedCaptureScope() { locked_capture_token=previous; }
    LockedCaptureScope(const LockedCaptureScope&)=delete;
};
inline bool capture_age_valid(uint64_t slot_token, uint64_t current_token,
                             uint64_t age_frames, uint64_t age_ms,
                             uint64_t max_frames, uint64_t max_ms) {
    // Retry submissions are not new game frames. A result for this identical,
    // still-locked input cannot become stale while its computation runs.
    if (current_token) return slot_token==current_token;
    return !slot_token && age_frames<=max_frames && age_ms<=max_ms;
}
}
