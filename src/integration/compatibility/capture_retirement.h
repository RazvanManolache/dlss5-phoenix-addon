#pragma once
#include <cstdint>
namespace phoenix_compat {
// A short optional yield window avoids wasting an entire application frame
// when its capture fence retires immediately after Present. The caller owns
// the clock/deadline and decides whether an idle worker makes waiting useful.
// Never spin on device removal (UINT64_MAX), and never wait for inference.
template<class Query,class Expired,class Yield>
uint64_t poll_capture_retirement(uint64_t target,uint64_t completed,bool may_wait,
                                Query&& query,Expired&& expired,Yield&& yield) {
    while(may_wait && completed<target && !expired()) {
        yield();
        completed=query();
    }
    return completed;
}
}
