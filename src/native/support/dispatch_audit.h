#pragma once
#include "policy.h"

namespace phoenix_addon::dispatch_audit {
struct Counts {
    uint64_t own_attempted = 0, own_accepted = 0;
    uint64_t other_attempted = 0, other_accepted = 0;
    // Launches say how much work was issued; calls say how it was submitted.
    // The two differ exactly when an implementation chains launches together.
    uint64_t own_calls = 0, other_calls = 0;
};
inline thread_local Counts* active = nullptr;
inline thread_local bool own_call = false;
class Scope {
  public:
    explicit Scope(Counts& counts) {
        check(!active, "phoenix nested claimed Evaluate audit");
        active = &counts;
    }
    ~Scope() {
        active = nullptr;
    }
    Scope(const Scope&) = delete;
};
class OwnCall {
    bool previous_ = own_call;

  public:
    OwnCall() {
        own_call = true;
    }
    ~OwnCall() {
        own_call = previous_;
    }
    OwnCall(const OwnCall&) = delete;
};
struct Call {
    Counts* counts = active;
    bool own = own_call;
    uint64_t launches;
    explicit Call(uint64_t count) : launches(count) {
        if (!counts)
            return;
        (own ? counts->own_attempted : counts->other_attempted) += launches;
        ++(own ? counts->own_calls : counts->other_calls);
    }
    void accepted() noexcept {
        if (counts)
            (own ? counts->own_accepted : counts->other_accepted) += launches;
    }
};
} // namespace production_game41::dispatch_audit
