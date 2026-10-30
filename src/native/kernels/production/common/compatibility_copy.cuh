#pragma once
#include "architecture_policy.cuh"
#include <stdint.h>

namespace phoenix_architecture {
// Ordinary async copies are issued by the elected lane. Complete that lane's
// pending transfers at the existing CTA consumer barrier, after independent
// setup/compute has had a chance to overlap. Other styles retain their wait
// inside copy_compatibility_tile; native bulk-copy barriers are unchanged.
__device__ __forceinline__ void finish_compatibility_copies() {
#if PHOENIX_STYLE_ARCH == 890
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
}
// Older paths used a dependent 4-byte load/store loop in one elected lane.
// Keep that caller and its CTA synchronization contract, but transfer aligned
// tiles 16 bytes at a time. No warp collective is legal here: other lanes may
// be outside the caller's divergent leader-only branch.
__device__ __forceinline__ void copy_compatibility_tile(
    uint32_t* shared_tile, uint32_t destination_offset, uint64_t source_address,
    uint32_t byte_count) {
    const uint32_t shared_base = uint32_t(__cvta_generic_to_shared(shared_tile));
#if PHOENIX_STYLE_ARCH >= 800
    // Ampere/Ada have ordinary cp.async, but not Hopper's bulk transaction
    // form. Bypass L1 for data published by other blocks. Wait in the issuing
    // lane before returning on Ampere. Ada completes at the caller's existing
    // consumer barrier, allowing the following independent work to overlap.
    for (uint32_t offset = 0; offset < byte_count; offset += 16)
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"
                     :: "r"(shared_base + destination_offset + offset),
                        "l"(source_address + offset) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
#if PHOENIX_STYLE_ARCH != 890
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
#else
    // Turing cannot issue cp.async. Vector loads/stores still remove three
    // quarters of the loop iterations without changing the copied bits.
    for (uint32_t offset = 0; offset < byte_count; offset += 16) {
        uint32_t a, b, c, d;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                     : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                     : "l"(source_address + offset) : "memory");
        asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};"
                     :: "r"(shared_base + destination_offset + offset),
                        "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
    }
#endif
}
} // namespace phoenix_architecture
