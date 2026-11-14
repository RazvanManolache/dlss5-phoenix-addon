#pragma once
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <span>

namespace phoenix_compat {
// A ring of frame slots in one shared mapping. Slot ownership follows the
// frame that names it: the client writes input/depth, the host writes output,
// and neither peer touches a slot again until that frame's ack was consumed
// and the slot comes back around. A single slot is the v14 layout exactly.
class CpuFrameMapping {
    static constexpr uint64_t maximum_bytes = 128ull << 20;
    static constexpr unsigned maximum_slots = 8;
    HANDLE handle_ = nullptr;
    uint8_t* view_ = nullptr;
    size_t plane_bytes_ = 0;
    unsigned slots_ = 0;
    struct alignas(8) SlotReady {
        volatile LONG64 frame;
        uint32_t depth_present;
        uint32_t depth_dims;
    };
    SlotReady* ready_ = nullptr;
    static uint64_t mapping_size(size_t plane, unsigned slots, bool prefetch) {
        const uint64_t pixels = uint64_t(plane) * 3 * slots;
        return prefetch ? ((pixels + 7) & ~uint64_t(7)) + slots * sizeof(SlotReady) : pixels;
    }

    bool map(unsigned width, unsigned height, unsigned slots, bool prefetch) {
        const auto bytes = plane_size(width, height);
        if (!handle_ || !bytes || !slots || slots > maximum_slots ||
            uint64_t(bytes) * 3 * slots > maximum_bytes * 2) { reset(); return false; }
        view_ = static_cast<uint8_t*>(MapViewOfFile(handle_, FILE_MAP_WRITE, 0, 0, size_t(mapping_size(bytes, slots, prefetch))));
        if (!view_) { reset(); return false; }
        plane_bytes_ = bytes;
        slots_ = slots;
        if (prefetch) ready_ = reinterpret_cast<SlotReady*>(view_ + ((bytes * 3 * slots + 7) & ~size_t(7)));
        return true;
    }
    size_t slot_offset(unsigned slot) const { return size_t(slot) * plane_bytes_ * 3; }

public:
    static size_t plane_size(unsigned width, unsigned height) {
        const uint64_t pixels = uint64_t(width) * height;
        return pixels && pixels <= maximum_bytes / (3 * sizeof(float)) ? size_t(pixels * 4) : 0;
    }
    static constexpr unsigned max_slots() { return maximum_slots; }
    CpuFrameMapping() = default;
    CpuFrameMapping(const CpuFrameMapping&) = delete;
    CpuFrameMapping& operator=(const CpuFrameMapping&) = delete;
    ~CpuFrameMapping() { reset(); }
    void reset() {
        if (view_) UnmapViewOfFile(view_);
        if (handle_) CloseHandle(handle_);
        view_ = nullptr;
        handle_ = nullptr;
        plane_bytes_ = 0;
        slots_ = 0;
        ready_ = nullptr;
    }
    bool create(unsigned width, unsigned height, unsigned slots = 1, bool prefetch = false) {
        reset();
        const auto bytes = plane_size(width, height);
        if (!bytes || !slots || slots > maximum_slots ||
            uint64_t(bytes) * 3 * slots > maximum_bytes * 2) return false;
        handle_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                     DWORD(mapping_size(bytes, slots, prefetch)), nullptr);
        return map(width, height, slots, prefetch);
    }
    bool open(HANDLE peer, uint64_t handle, unsigned width, unsigned height, unsigned slots = 1, bool prefetch = false) {
        reset();
        if (!peer || !handle || handle != uintptr_t(handle) ||
            !DuplicateHandle(peer, reinterpret_cast<HANDLE>(uintptr_t(handle)), GetCurrentProcess(),
                             &handle_, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
        return map(width, height, slots, prefetch);
    }
    void publish(unsigned slot, uint64_t frame, bool has_depth, uint32_t dimensions) {
        if (!ready_ || slot >= slots_) return;
        auto& item = ready_[slot];
        InterlockedExchange64(&item.frame, 0);
        item.depth_present = has_depth;
        item.depth_dims = dimensions;
        InterlockedExchange64(&item.frame, LONG64(frame));
    }
    uint64_t ready_frame(unsigned slot) const {
        return ready_ && slot < slots_ ? uint64_t(InterlockedCompareExchange64(&ready_[slot].frame, 0, 0)) : 0;
    }
    bool depth_ready(unsigned slot) const { return ready_[slot].depth_present != 0; }
    uint32_t depth_dims(unsigned slot) const { return ready_[slot].depth_dims; }
    explicit operator bool() const { return view_ != nullptr; }
    uint64_t handle() const { return uintptr_t(handle_); }
    unsigned slots() const { return slots_; }
    size_t plane_bytes() const { return plane_bytes_; }
    std::span<uint8_t> input(unsigned slot = 0) const {
        return slot < slots_ ? std::span<uint8_t>{view_ + slot_offset(slot), plane_bytes_}
                             : std::span<uint8_t>{};
    }
    std::span<uint8_t> output(unsigned slot = 0) const {
        return slot < slots_ ? std::span<uint8_t>{view_ + slot_offset(slot) + plane_bytes_, plane_bytes_}
                             : std::span<uint8_t>{};
    }
    std::span<float> depth(unsigned slot = 0) const {
        return slot < slots_ ? std::span<float>{reinterpret_cast<float*>(view_ + slot_offset(slot) +
                                        2 * plane_bytes_), plane_bytes_ / sizeof(float)}
                             : std::span<float>{};
    }
};
}
