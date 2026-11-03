#pragma once
#include <memory>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <cerrno>
#include <system_error>
#endif

namespace phoenix_compat {
inline constexpr unsigned processing_worker_stack_bytes = 4 * 1024 * 1024;
// The transport state is a lease held by the worker, not a raw pointer into the
// game-facing client. Its DLL is separately pinned by HostState before launch.
template <class State, class Work>
void start_owned_worker(std::shared_ptr<State> state, Work work) {
#ifdef _WIN32
    // std::thread inherits the game's executable stack reserve. Some legacy
    // games reserve only 192 KiB, insufficient for model/driver initialization.
    // Reserve our own virtual stack; pages are committed only as it grows.
    struct Context {
        std::shared_ptr<State> state;
        Work work;
        static unsigned __stdcall run(void* pointer) noexcept {
            std::unique_ptr<Context> context(static_cast<Context*>(pointer));
            context->work(*context->state);
            return 0;
        }
    };
    auto context=std::make_unique<Context>(Context{std::move(state),std::move(work)});
    const auto thread=_beginthreadex(nullptr,processing_worker_stack_bytes,&Context::run,
        context.get(),STACK_SIZE_PARAM_IS_A_RESERVATION,nullptr);
    if(!thread)throw std::system_error(errno,std::generic_category(),"Cannot create graphics processing worker");
    context.release();
    CloseHandle(reinterpret_cast<HANDLE>(thread)); // Detached, state stays leased.
#else
    std::thread([state = std::move(state), work = std::move(work)]() mutable {
        work(*state);
    }).detach();
#endif
}
}
