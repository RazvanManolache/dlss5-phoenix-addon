#pragma once
#include <windows.h>
#include <d3d12.h>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <unordered_map>
namespace phoenix_kernel {
struct DeviceArchitectureCache {
    std::mutex lock;
    std::unordered_map<uint64_t, unsigned> values;
};
inline uint64_t adapter_identity(LUID luid) {
    return (uint64_t(uint32_t(luid.HighPart)) << 32) | luid.LowPart;
}
inline DeviceArchitectureCache& device_architecture_cache() {
    static DeviceArchitectureCache value;
    return value;
}
inline void remember_device_architecture(LUID luid, unsigned architecture) {
    auto& cache = device_architecture_cache();
    std::lock_guard guard(cache.lock);
    cache.values[adapter_identity(luid)] = architecture;
}
// Match the actual D3D12 adapter, not GPU ordinal zero. Cached per adapter;
// creates no CUDA contexts or allocations and executes no kernels.
inline unsigned device_architecture(ID3D12Device* device) {
    if (!device) return 0;
    const LUID luid = device->GetAdapterLuid();
    const uint64_t key = adapter_identity(luid);
    auto& cache = device_architecture_cache();
    std::lock_guard guard(cache.lock);
    if (const auto found = cache.values.find(key); found != cache.values.end()) return found->second;
    unsigned result = 0;
    auto library = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (library) {
        using Init = int(__stdcall*)(unsigned);
        using Count = int(__stdcall*)(int*);
        using Device = int(__stdcall*)(int*, int);
        using DeviceLuid = int(__stdcall*)(char*, unsigned*, int);
        using Capability = int(__stdcall*)(int*, int*, int);
        auto init = reinterpret_cast<Init>(GetProcAddress(library, "cuInit"));
        auto count = reinterpret_cast<Count>(GetProcAddress(library, "cuDeviceGetCount"));
        auto get = reinterpret_cast<Device>(GetProcAddress(library, "cuDeviceGet"));
        auto identity = reinterpret_cast<DeviceLuid>(GetProcAddress(library, "cuDeviceGetLuid"));
        auto capability = reinterpret_cast<Capability>(GetProcAddress(library, "cuDeviceComputeCapability"));
        int total = 0;
        if (init && count && get && identity && capability && !init(0) && !count(&total) && total >= 0 && total <= 64)
            for (int index = 0; index < total; ++index) {
                int candidate = 0, major = 0, minor = 0;
                char bytes[8]{}; unsigned mask = 0;
                if (!get(&candidate, index) && !identity(bytes, &mask, candidate) &&
                    !std::memcmp(bytes, &luid, sizeof(luid)) && !capability(&major, &minor, candidate)) {
                    result = unsigned(major * 10 + minor); break;
                }
            }
        FreeLibrary(library);
    }
    cache.values.emplace(key, result);
    return result;
}
}
