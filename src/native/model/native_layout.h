#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include "registry.h"

namespace phoenix_model {
// Positions exclude the clear/copy helpers, as in FrameDispatch and Network.
struct NativeLayout {
    const NativeSchedule* schedule = &models::original.native;
    unsigned launches() const { return schedule->launches; }
    unsigned repack_in() const { return schedule->repack_in; }
    unsigned transformer_begin() const { return repack_in() + 1; }
    unsigned repack_out() const { return transformer_begin() + 40; }
    unsigned scheduled_launches(unsigned depth, const std::array<int, 8>& repeats) const {
        if (depth < 1 || depth > 8)
            throw std::runtime_error("Native transformer depth outside 1..8");
        unsigned count = launches() - 40;
        for (unsigned block = 0; block < depth; ++block) {
            if (repeats[block] < 0 || repeats[block] > 4)
                throw std::runtime_error("Native transformer repetitions outside 0..4");
            count += 5 * unsigned(repeats[block]);
        }
        return count;
    }
    std::array<unsigned, 15> group_ends() const {
        return schedule->group_ends;
    }
    unsigned group(unsigned position) const {
        const auto ends = group_ends();
        for (unsigned index = 0; index < ends.size(); ++index)
            if (position < ends[index]) return index;
        throw std::runtime_error("Native launch position outside model schedule");
    }
    unsigned skipped(unsigned mask) const {
        unsigned count = 0, begin = 0;
        const auto ends = group_ends();
        for (unsigned index = 0; index < ends.size(); ++index) {
            if (mask & (1u << index)) count += ends[index] - begin;
            begin = ends[index];
        }
        return count;
    }
};
inline NativeLayout native_layout(std::string_view architecture = "phoenix-network37/1") {
    for (const auto& model : registry().entries())
        if (model.architecture == architecture && model.native.launches)
            return {&model.native};
    throw std::runtime_error("Unsupported native RTX model architecture");
}
} // namespace phoenix_model
