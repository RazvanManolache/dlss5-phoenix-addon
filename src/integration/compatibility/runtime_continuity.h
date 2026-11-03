#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace phoenix_compat {
// ReShade runtime reset is not graphics-device/window destruction. A bounded
// handoff retains eligible helper sessions across that transient reset. The
// controller must drop all swapchain references before parking a session.
template<class Session> class RuntimeContinuity {
public:
    using Owner = std::pair<uintptr_t, uintptr_t>; // Device identity, HWND.
private:
    struct Entry { uint64_t order; std::unique_ptr<Session> value; };
    std::map<Owner, Entry> entries_;
    uint64_t order_ = 0;
    static constexpr size_t limit = 4;
public:
    void park(Owner owner, std::unique_ptr<Session> value) {
        if (!owner.first || !owner.second || !value) return;
        entries_.insert_or_assign(owner, Entry{++order_, std::move(value)});
        if (entries_.size() > limit) {
            auto oldest = entries_.begin();
            for (auto it = entries_.begin(); it != entries_.end(); ++it)
                if (it->second.order < oldest->second.order) oldest = it;
            entries_.erase(oldest);
        }
    }
    std::unique_ptr<Session> take(Owner owner) {
        auto found = entries_.find(owner);
        if (found == entries_.end()) return {};
        auto value = std::move(found->second.value);
        entries_.erase(found);
        return value;
    }
    template<class Alive> void prune(Alive alive) {
        for (auto it = entries_.begin(); it != entries_.end();)
            if (!alive(it->first)) it = entries_.erase(it); else ++it;
    }
    void destroy_device(uintptr_t device) {
        prune([&](Owner owner) { return owner.first != device; });
    }
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }
};
} // namespace phoenix_compat
