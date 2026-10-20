#pragma once
#include <array>
#include <stdexcept>

namespace phoenix_addon::phoenix {
// Query bookkeeping is independent of a GPU device. Each network pass emits
// the selected model's natural stage boundaries in order.
template <unsigned GroupCount> struct StageQueries {
    static constexpr unsigned maximum_passes = 10;
    static constexpr unsigned capacity = GroupCount * maximum_passes;
    static constexpr unsigned stamps = capacity + 9;
    std::array<unsigned, capacity> indices{};
    unsigned count = 0;
    bool accepts(unsigned group) const noexcept {
        return count < capacity && group == count % GroupCount;
    }
    void append(unsigned group, unsigned query) {
        if (!accepts(group))
            throw std::runtime_error("Phoenix DLSS5 GPU group timestamp order");
        indices[count++] = query;
    }
};
} // namespace phoenix_addon::phoenix
