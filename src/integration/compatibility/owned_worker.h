#pragma once
#include <memory>
#include <thread>
#include <utility>

namespace phoenix_compat {
// The transport state is a lease held by the worker, not a raw pointer into the
// game-facing client. Its DLL is separately pinned by HostState before launch.
template <class State, class Work>
void start_owned_worker(std::shared_ptr<State> state, Work work) {
    std::thread([state = std::move(state), work = std::move(work)]() mutable {
        work(*state);
    }).detach();
}
}
