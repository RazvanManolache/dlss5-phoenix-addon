#pragma once
#include "backend.h"
#include <algorithm>
#include <map>
#include <numeric>

namespace phoenix_cpp {
// Logical tensor addresses exist only during graph construction. They never
// reach a backend; the immutable command list is relocated before execution.
class ScratchPlan {
    static constexpr uint64_t logical_address_base = uint64_t(1) << 62;
    static constexpr unsigned logical_address_stride_bits = 34;
    struct TensorLifetime {
        uint64_t bytes, offset = 0;
        size_t first = SIZE_MAX, last = 0;
    };
    std::vector<TensorLifetime> lifetimes_;
    size_t index(phx::OperandAddress address) const {
        return size_t((phx::address_bits(address) - logical_address_base) >> logical_address_stride_bits);
    }

  public:
    Allocation reserve(uint64_t bytes) {
        demand(bytes && bytes < (uint64_t(1) << logical_address_stride_bits),
               "Logical scratch extent");
        demand(lifetimes_.size() < (uint64_t(1) << 20), "Logical scratch capacity");
        auto address =
            logical_address_base + (uint64_t(lifetimes_.size()) << logical_address_stride_bits);
        lifetimes_.push_back({bytes});
        return {phx::address_from_bits(address), bytes, 0};
    }
    bool contains(phx::OperandAddress address) const {
        return phx::address_bits(address) >= logical_address_base && index(address) < lifetimes_.size();
    }
    void touch(phx::OperandAddress address, size_t command) {
        if (!contains(address))
            return;
        auto& range = lifetimes_[index(address)];
        range.first = std::min(range.first, command);
        range.last = std::max(range.last, command);
    }
    // Best-fit allocation over command lifetimes. An input and output used by
    // the same command cannot alias: reuse requires last_use < first_use.
    // Coalescing adjacent free ranges avoids accumulating one arena per tensor.
    uint64_t layout() {
        std::vector<size_t> order;
        for (size_t iteration_index = 0; iteration_index < lifetimes_.size(); ++iteration_index)
            if (lifetimes_[iteration_index].first != SIZE_MAX)
                order.push_back(iteration_index);
        std::stable_sort(order.begin(), order.end(), [&](size_t left_index, size_t right_index) {
            return lifetimes_[left_index].first < lifetimes_[right_index].first;
        });
        std::vector<size_t> active_lifetimes;
        std::map<uint64_t, uint64_t> free_ranges;
        uint64_t arena_extent = 0, peak_extent = 0;
        auto aligned_bytes = [](uint64_t bytes) { return (bytes + 255) & ~uint64_t(255); };
        auto release_range = [&](uint64_t offset, uint64_t bytes) {
            auto next = free_ranges.lower_bound(offset);
            if (next != free_ranges.begin()) {
                auto previous = std::prev(next);
                if (previous->first + previous->second == offset) {
                    offset = previous->first;
                    bytes += previous->second;
                    free_ranges.erase(previous);
                }
            }
            next = free_ranges.lower_bound(offset);
            if (next != free_ranges.end() && offset + bytes == next->first) {
                bytes += next->second;
                free_ranges.erase(next);
            }
            if (offset + bytes == arena_extent) {
                arena_extent = offset;
                while (!free_ranges.empty()) {
                    auto last = std::prev(free_ranges.end());
                    if (last->first + last->second != arena_extent)
                        break;
                    arena_extent = last->first;
                    free_ranges.erase(last);
                }
            } else
                demand(free_ranges.emplace(offset, bytes).second,
                       "Overlapping scratch free ranges");
        };
        for (size_t current : order) {
            auto& range = lifetimes_[current];
            for (auto iteration_index = active_lifetimes.begin();
                 iteration_index != active_lifetimes.end();)
                if (lifetimes_[*iteration_index].last < range.first) {
                    auto& old = lifetimes_[*iteration_index];
                    release_range(old.offset, aligned_bytes(old.bytes));
                    iteration_index = active_lifetimes.erase(iteration_index);
                } else
                    ++iteration_index;
            auto best = free_ranges.end();
            auto bytes = aligned_bytes(range.bytes);
            for (auto iteration_index = free_ranges.begin(); iteration_index != free_ranges.end();
                 ++iteration_index)
                if (iteration_index->second >= bytes &&
                    (best == free_ranges.end() || iteration_index->second < best->second))
                    best = iteration_index;
            if (best == free_ranges.end()) {
                range.offset = arena_extent;
                arena_extent += bytes;
                peak_extent = std::max(peak_extent, arena_extent);
            } else {
                range.offset = best->first;
                auto remainder = best->second - bytes;
                free_ranges.erase(best);
                if (remainder)
                    free_ranges.emplace(range.offset + bytes, remainder);
            }
            active_lifetimes.push_back(current);
        }
        // A chronological allocator cannot move a small live tensor out of a
        // later large tensor's way. Try packing larger buffers first, checking
        // their complete inclusive command lifetimes before sharing an address.
        // Keep the chronological layout unless this reduces the actual peak.
        // Bound the offline search so unusually large streamed graphs still
        // initialize promptly with the existing linear allocator.
        if (order.size() <= 8192) {
            std::vector<uint64_t> chronological_offsets;
            chronological_offsets.reserve(lifetimes_.size());
            for (const auto& range : lifetimes_)
                chronological_offsets.push_back(range.offset);
            auto by_size = order;
            std::stable_sort(by_size.begin(), by_size.end(), [&](size_t left, size_t right) {
                return lifetimes_[left].bytes > lifetimes_[right].bytes;
            });
            std::vector<size_t> placed;
            std::vector<std::pair<uint64_t, uint64_t>> occupied;
            uint64_t packed_peak = 0;
            for (size_t current : by_size) {
                auto& range = lifetimes_[current];
                const auto bytes = aligned_bytes(range.bytes);
                occupied.clear();
                for (auto index : placed) {
                    const auto& other = lifetimes_[index];
                    if (range.first <= other.last && other.first <= range.last)
                        occupied.emplace_back(other.offset,
                                              other.offset + aligned_bytes(other.bytes));
                }
                std::sort(occupied.begin(), occupied.end());
                uint64_t offset = 0;
                for (const auto& [begin, end] : occupied) {
                    if (offset + bytes <= begin)
                        break;
                    offset = std::max(offset, end);
                }
                range.offset = offset;
                packed_peak = std::max(packed_peak, offset + bytes);
                placed.push_back(current);
            }
            if (packed_peak < peak_extent)
                peak_extent = packed_peak;
            else
                for (size_t index = 0; index < lifetimes_.size(); ++index)
                    lifetimes_[index].offset = chronological_offsets[index];
        }
        return peak_extent;
    }
    uint64_t offset(phx::OperandAddress address) const {
        demand(contains(address), "Scratch relocation address");
        auto tensor_index = index(address);
        const auto& range = lifetimes_[tensor_index];
        demand(range.first != SIZE_MAX, "Unused scratch relocation");
        auto subview_offset = (phx::address_bits(address) - logical_address_base) &
                              ((uint64_t(1) << logical_address_stride_bits) - 1);
        demand(subview_offset < range.bytes, "Scratch relocation bounds");
        return range.offset + subview_offset;
    }
    phx::OperandAddress relocate(phx::OperandAddress address, const Allocation& arena) const {
        if (!contains(address)) return address;
        const auto byte_offset = offset(address);
        demand(byte_offset < arena.bytes && phx::address_bits(arena.address) <= UINT64_MAX - byte_offset,
               "Scratch arena relocation overflow");
        return phx::address_from_bits(phx::address_bits(arena.address) + byte_offset);
    }
};
}
