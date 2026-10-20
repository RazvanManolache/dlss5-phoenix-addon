#pragma once
#include <cstddef>
#include <array>
#include <string_view>

namespace phoenix_model {
struct NativeSchedule {
    unsigned launches = 0, repack_in = 0, split_blocks = 0, first_transformer = 0;
    std::array<unsigned, 4> fused_interior_pairs{};
    std::array<unsigned, 15> group_ends{};
};
// Model identity is independent of its DLL container and execution backend.
struct Definition {
    std::string_view id, label, architecture, resource, layout_sha256;
    size_t record_count, decoded_tensor_count;
    unsigned history_gain_block;
    bool implemented;
    std::string_view unavailable_reason;
    // Reference edition identity, independent of where its resource sits in a DLL.
    std::string_view reference_dll_sha256{}, reference_resource_sha256{};
    NativeSchedule native{};
    unsigned nvidia_preset = 0;
};
} // namespace phoenix_model
