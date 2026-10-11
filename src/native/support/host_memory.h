#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

namespace phoenix_addon::host_memory {
enum class Tag : size_t {
    resources,
    recordings,
    transitions,
    commands,
    epoch_states,
    discarded_states,
    aliasing,
    heap_refs,
    snapshots,
    engine_leases,
    submitted,
    ownership,
    originals,
    fresh_features,
    queues,
    retired,
    command_signatures,
    committed_resources,
    count
};
inline constexpr const char* names[]{"ledger_resources",
                                     "ledger_recordings",
                                     "ledger_transitions",
                                     "observer_commands",
                                     "observer_epoch_states",
                                     "observer_discarded_states",
                                     "observer_aliasing",
                                     "binding_heap_refs",
                                     "binding_snapshots",
                                     "session_engine_leases",
                                     "submitted_session_refs",
                                     "queue_ownership",
                                     "original_list_identities",
                                     "fresh_features",
                                     "queues",
                                     "retired_session_refs",
                                     "observer_command_signatures",
                                     "observer_committed_resources"};
struct Counters {
    std::atomic<uint64_t> allocated{0}, freed{0}, live{0}, peak{0}, allocations{0},
        deallocations{0};
    void add(uint64_t bytes) noexcept {
        allocated.fetch_add(bytes, std::memory_order_relaxed);
        allocations.fetch_add(1, std::memory_order_relaxed);
        const auto value = live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        auto prior = peak.load(std::memory_order_relaxed);
        while (prior < value &&
               !peak.compare_exchange_weak(prior, value, std::memory_order_relaxed)) {
        }
    }
    void remove(uint64_t bytes) noexcept {
        freed.fetch_add(bytes, std::memory_order_relaxed);
        live.fetch_sub(bytes, std::memory_order_relaxed);
        deallocations.fetch_add(1, std::memory_order_relaxed);
    }
};
inline auto& counters() {
    static auto* values = new std::array<Counters, size_t(Tag::count)>;
    return *values;
}
template <class T, Tag Group> struct Allocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    template <class U> struct rebind {
        using other = Allocator<U, Group>;
    };
    Allocator() noexcept = default;
    template <class U> Allocator(const Allocator<U, Group>&) noexcept {}
    T* allocate(size_t count) {
        auto* result = std::allocator<T>{}.allocate(count);
        counters()[size_t(Group)].add(count * sizeof(T));
        return result;
    }
    void deallocate(T* pointer, size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
        counters()[size_t(Group)].remove(count * sizeof(T));
    }
};
template <class T, class U, Tag G> bool operator==(Allocator<T, G>, Allocator<U, G>) noexcept {
    return true;
}
template <class T, class U, Tag G> bool operator!=(Allocator<T, G>, Allocator<U, G>) noexcept {
    return false;
}
template <class K, class V, Tag G>
using Map = std::map<K, V, std::less<K>, Allocator<std::pair<const K, V>, G>>;
template <class T, Tag G> using Vector = std::vector<T, Allocator<T, G>>;
template <class T, Tag G> using Set = std::set<T, std::less<T>, Allocator<T, G>>;
}
