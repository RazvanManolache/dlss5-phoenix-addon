// Typed native launch-plan data and validation.
// Offsets refer to workspace/model buffers; ordering and parameter widths are runtime contracts.

#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// CPU argument materialization only. JSON parsing, digest verification, resource
// ownership, descriptors and dispatch submission belong to the consuming host.
namespace phoenix_launch_plan {
enum class BindingKind : uint32_t { workspace = 0, weights = 1, descriptor = 2, null_pointer = 3 };
struct Binding {
    uint32_t argument_offset;
    BindingKind kind;
    uint64_t value; // arena-relative byte offset, or runtime descriptor/sampler encoding
};
struct Arena {
    uint64_t base, bytes;
};
struct Launch {
    std::array<uint32_t, 3> grid, block;
    uint32_t dynamic_shared_bytes;
    std::vector<uint8_t> argument_template;
    std::vector<Binding> bindings;
};
struct Barrier {
    uint32_t after_launch; // completed launch count, never zero-based launch index
    bool global_uav;
    uint64_t owned_resource_identity; // zero for global; supplied by host otherwise
    uint32_t before, after;
};
inline std::vector<uint8_t> materialize(const Launch& launch, const std::array<Arena, 2>& arenas) {
    const auto maximum = std::numeric_limits<uint64_t>::max();
    for (const auto& arena : arenas) {
        if (!arena.base || arena.base % 512 || !arena.bytes || arena.bytes > maximum - arena.base)
            throw std::invalid_argument("arena extent/alignment");
    }
    if (!(arenas[0].base + arenas[0].bytes <= arenas[1].base ||
          arenas[1].base + arenas[1].bytes <= arenas[0].base))
        throw std::invalid_argument("arena overlap");
    auto bytes = launch.argument_template;
    if (bytes.empty() || bytes.size() > 264 || bytes.size() % 4 || launch.dynamic_shared_bytes)
        throw std::invalid_argument("argument/shared extent");
    for (unsigned axis = 0; axis != 3; ++axis)
        if (!launch.grid[axis] || !launch.block[axis])
            throw std::invalid_argument("launch shape");
    std::vector<bool> used(bytes.size());
    for (const auto& binding : launch.bindings) {
        const auto offset = binding.argument_offset;
        if (offset % 8 || offset > bytes.size() || bytes.size() - offset < 8)
            throw std::invalid_argument("binding extent");
        uint64_t value = 0;
        switch (binding.kind) {
        case BindingKind::workspace:
        case BindingKind::weights: {
            const auto& arena = arenas[static_cast<unsigned>(binding.kind)];
            if (binding.value >= arena.bytes)
                throw std::invalid_argument("relocation bounds");
            value = arena.base + binding.value;
            break;
        }
        case BindingKind::descriptor:
            value = binding.value;
            break;
        case BindingKind::null_pointer:
            if (binding.value)
                throw std::invalid_argument("null binding payload");
            break;
        default:
            throw std::invalid_argument("unknown binding kind");
        }
        for (unsigned byte = 0; byte != 8; ++byte) {
            if (used[offset + byte] || bytes[offset + byte])
                throw std::invalid_argument("binding overlap/template");
            used[offset + byte] = true;
            bytes[offset + byte] = static_cast<uint8_t>(value >> (8 * byte));
        }
    }
    return bytes;
}
} // namespace native_plan_y37
