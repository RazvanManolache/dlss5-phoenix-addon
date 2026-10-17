#pragma once
#include <cstddef>
#include <stdexcept>

namespace phoenix_kernel {
template<class Query>
inline void require_native_module_capability(size_t pointer_bytes, Query&& query) {
    // The 32-bit driver can export module creation but fault when it is called.
    // Do not infer support from its function address, or affect the x64 route.
    if (pointer_bytes == 4 && !query())
        throw std::runtime_error("32-bit NVAPI CUDA module loading is unsupported; use the 64-bit graphics helper");
}
}
