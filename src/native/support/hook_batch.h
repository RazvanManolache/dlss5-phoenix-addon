#pragma once
#include "../../../third_party/minhook/include/MinHook.h"
#include <array>
#include <stdexcept>

namespace phoenix_addon {
// MinHook freezes/enumerates threads when enabling hooks. Queue all related
// hooks first, then activate them in one freeze rather than once per method.
class HookBatch {
    struct Entry { void* target; void** registered; void** original; };
    std::array<Entry, 40> entries_{};
    size_t count_ = 0;
    bool committed_ = false;
public:
    HookBatch() {
        const auto status = MH_Initialize();
        if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED)
            throw std::runtime_error("Phoenix DLSS5 hook initialization failed");
    }
    HookBatch(const HookBatch&) = delete;
    HookBatch& operator=(const HookBatch&) = delete;
    void add(void* target, void* detour, void** original, void** registered) {
        if (count_ == entries_.size()) throw std::runtime_error("Phoenix DLSS5 hook batch is full");
        if (MH_CreateHook(target, detour, original) != MH_OK)
            throw std::runtime_error("Phoenix DLSS5 hook creation failed");
        // Record ownership before queueing so a queue/commit failure also rolls back.
        entries_[count_++] = {target, registered, original};
        *registered = target;
        if (MH_QueueEnableHook(target) != MH_OK)
            throw std::runtime_error("Phoenix DLSS5 hook queue failed");
    }
    void commit() {
        if (count_ && MH_ApplyQueued() != MH_OK)
            throw std::runtime_error("Phoenix DLSS5 hook batch activation failed");
        committed_ = true;
    }
    ~HookBatch() {
        if (committed_) return;
        while (count_) {
            const auto& entry = entries_[--count_];
            if (MH_RemoveHook(entry.target) == MH_OK) {
                *entry.registered = nullptr;
                *entry.original = nullptr;
            }
            // If removal itself failed, retain the trampoline pointers: an
            // active detour must never observe a cleared original function.
        }
    }
};
}
