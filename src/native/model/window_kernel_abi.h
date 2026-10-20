#pragma once
#include <cstddef>
#include <cstdint>

namespace phoenix_model::window_abi {
// Fixed-width address slots, never host pointers: the packet layout must be
// identical for x86 tooling and x64/device execution. These are ABI prefixes,
// not complete launch packets, and are used for offsetof rather than aliasing.
struct Shift { uint32_t x, y; };
template <unsigned AddressCount> struct FusedWindowPrefix {
    uint64_t addresses[AddressCount];
    uint32_t height, width;
    Shift current;
    uint64_t wait;
    Shift previous;
};
using CompactFusedWindow = FusedWindowPrefix<3>;
using WideFusedWindow = FusedWindowPrefix<4>;

struct GeometryOffsets {
    unsigned height, width, current, previous;
};
template <class Packet> constexpr GeometryOffsets geometry_offsets() {
    return {offsetof(Packet, height), offsetof(Packet, width),
            offsetof(Packet, current), offsetof(Packet, previous)};
}
inline constexpr auto compact = geometry_offsets<CompactFusedWindow>();
inline constexpr auto wide = geometry_offsets<WideFusedWindow>();
static_assert(compact.height == 24 && compact.current == 32 && compact.previous == 48);
static_assert(wide.height == 32 && wide.current == 40 && wide.previous == 56);

struct SplitQkvPrefix {
    uint64_t input, output, weights;
    uint32_t height, width;
    Shift current;
};
struct SplitPooledProjectionPrefix {
    uint64_t input, skip, encoder_skip, pooled, weights, wait, publication;
    Shift current;
};
struct SplitProjectionPrefix {
    uint64_t input, skip, output, weights;
    uint32_t height, width;
    uint32_t reserved_controls[2]; // Meaning not needed by the shift specialization.
    uint64_t publication, wait;
    Shift current;
};
inline constexpr unsigned split_qkv_shift = offsetof(SplitQkvPrefix, current);
inline constexpr unsigned split_pool_shift = offsetof(SplitPooledProjectionPrefix, current);
inline constexpr unsigned split_projection_shift = offsetof(SplitProjectionPrefix, current);
static_assert(split_qkv_shift == 32 && split_pool_shift == 56 && split_projection_shift == 64);
} // namespace phoenix_model::window_abi
