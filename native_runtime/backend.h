// Backend contract for resident tensor allocations, uploads, ordered commands and timing.
// Allocation tokens identify backend resources; address may be a logical handle rather than a CPU pointer.

#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#ifndef PHX_DEVICE
#define PHX_DEVICE
#endif
#include "../src/backends/hip_sycl/compute_ops.h"
namespace phoenix_cpp {
struct Allocation {
    void* address = nullptr;
    uint64_t bytes = 0;
    uint64_t token = 0;
};
struct Event {
    uint32_t block = 0, operation = 0;
    double milliseconds = 0;
    bool gpu = false;
    uint32_t layer = 0;
};
struct Command {
    phx::Operation operation;
    uint64_t count;
    uint32_t block, layer;
};
class Backend {
  public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual Allocation allocate(uint64_t bytes) = 0;
    virtual void release(Allocation) = 0;
    // Consume/copy all source bytes before returning; callers may free the
    // source immediately. This contract also applies to temporary model tensors.
    virtual void upload(Allocation, const void*, uint64_t) = 0;
    virtual void download(Allocation, void*, uint64_t) = 0;
    virtual void execute(const std::vector<Command>&, std::vector<Event>&) = 0;
    virtual void wait() = 0;
};
using CreateBackend = Backend* (*)(const char*, const wchar_t*, char*, size_t);
using DestroyBackend = void (*)(Backend*);
inline void demand(bool yes, const char* why) {
    if (!yes)
        throw std::runtime_error(why);
}
}
