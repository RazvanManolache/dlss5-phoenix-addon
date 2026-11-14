#pragma once
#include <bit>
#include <cstdint>
#include <stdexcept>

namespace phoenix_compat {
// Depth control for the shared-mapping frame ring. Two slots is the minimum
// that overlaps producer and helper; deeper rings only absorb jitter, so the
// limit grows when the helper starves (it finished with nothing staged) and
// shrinks when queued frames wait long enough to go stale. The mapping may
// hold more slots than the current limit; the limit is the active depth.
class CpuRingDepth {
    unsigned slots_ = 1;
    unsigned limit_ = 1;

public:
    explicit CpuRingDepth(unsigned slots = 1) { reset(slots); }
    void reset(unsigned slots) {
        slots_ = slots ? slots : 1;
        limit_ = slots_ > 1 ? 2 : 1;
    }
    unsigned slots() const { return slots_; }
    unsigned limit() const { return limit_; }
    // Helper completed a frame and no staged frame was waiting for it.
    void starved() { if (limit_ < slots_) ++limit_; }
    // A staged frame waited longer than the work it was hiding.
    void backlogged() { if (limit_ > (slots_ > 1 ? 2u : 1u)) --limit_; }
    bool admits(unsigned outstanding) const { return outstanding < limit_; }
};

// Keep slot ownership separate from FIFO order. Abandoning an input can
// create a hole while older outputs are still leased.
class CpuRingSlots {
    uint32_t occupied_ = 0;

public:
    unsigned size() const { return std::popcount(occupied_); }
    unsigned acquire(unsigned slots) {
        const auto slot = unsigned(std::countr_one(occupied_));
        if (slot >= slots || slot >= 32) throw std::logic_error("No free CPU frame slot");
        occupied_ |= uint32_t{1} << slot;
        return slot;
    }
    void release(unsigned slot) {
        if (slot >= 32 || !(occupied_ & (uint32_t{1} << slot)))
            throw std::logic_error("CPU frame slot was not acquired");
        occupied_ &= ~(uint32_t{1} << slot);
    }
};
}
