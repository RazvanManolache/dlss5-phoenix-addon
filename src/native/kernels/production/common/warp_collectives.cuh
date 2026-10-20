#pragma once

// Collectives use the current active mask, not an assumed full warp. The
// broadcast selects the first matching active lane and retains each lane's
// previous value if none matches. Keep volatile assembly at this boundary.

__device__ __forceinline__ uint32_t broadcast_active_lane(uint32_t value) {
    const unsigned mask = __activemask();
    return __shfl_sync(mask, value, __ffs(mask) - 1);
}

__device__ __forceinline__ uint32_t broadcast_warp_word(uint32_t value, bool selected,
                                                        uint32_t previous) {

    uint32_t result;
    asm volatile("{ .reg .b32 mask,chosen,reversed,lane,tmp; .reg .pred pick,nonempty;\n"
                 "activemask.b32 mask; setp.ne.u32 pick,%2,0;\n"
                 "vote.sync.ballot.b32 chosen,pick,mask;\n"
                 "setp.ne.u32 nonempty,chosen,0;\n"
                 "brev.b32 reversed,chosen; bfind.u32 lane,reversed; sub.u32 lane,31,lane;\n"
                 "mov.b32 tmp,%3;\n"
                 "@nonempty shfl.sync.idx.b32 tmp,%1,lane,31,mask;\n"
                 "mov.b32 %0,tmp;\n}"
                 : "=r"(result)
                 : "r"(value), "r"(uint32_t(selected)), "r"(previous));
    return result;
}

template <bool All> __device__ __forceinline__ bool vote_warp_predicate(bool value) {
    uint32_t result;
    if constexpr (All) {
        asm volatile("{.reg .b32 mask; .reg .pred p,q; activemask.b32 mask; setp.ne.u32 p,%1,0; "
                     "vote.sync.all.pred q,p,mask; selp.u32 %0,1,0,q;}"
                     : "=r"(result)
                     : "r"(uint32_t(value)));
    } else {
        asm volatile("{.reg .b32 mask; .reg .pred p,q; activemask.b32 mask; setp.ne.u32 p,%1,0; "
                     "vote.sync.any.pred q,p,mask; selp.u32 %0,1,0,q;}"
                     : "=r"(result)
                     : "r"(uint32_t(value)));
    }
    return result != 0;
}
