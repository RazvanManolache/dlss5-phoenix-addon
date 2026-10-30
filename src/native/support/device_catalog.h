#pragma once
#include "device_selection.h"
#include "kernel_style_device.h"
#include <windows.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include "../../../third_party/nlohmann/json.hpp"
namespace phoenix_addon::phoenix {
// Discovery here reads a validated catalogue. Optional compute SDKs load only
// in the isolated worker, never in the game's rendering process.
inline void append_native_devices(std::vector<ExecutionDevice>& result,
                                  const std::filesystem::path& build, const char* provider) {
    if (build.empty() || !build.is_absolute())
        return;
    try {
        std::ifstream stream(build / "manifest.json");
        nlohmann::json manifest;
        stream >> manifest;
        if (manifest.value("schema", std::string()) != "phoenix-native-compute/1" ||
            manifest.value("provider", std::string()) != provider ||
            manifest.value("abi", 0) != 1 || !manifest.value("passed", false) ||
            !manifest.value("inference_validated", false) ||
            !std::filesystem::is_regular_file(build / "phoenix_compute.dll"))
            return;
        for (const auto& entry : manifest.at("validated_devices")) {
            const auto device_id = entry.at("id").get<std::string>();
            if (device_id.rfind(std::string(provider) + ":", 0) != 0 ||
                !valid_execution_device(device_id))
                continue;
            bool duplicate = false;
            for (const auto& cuda_device : result)
                if (cuda_device.id == device_id)
                    duplicate = true;
            if (!duplicate)
                result.push_back({device_id, entry.at("name").get<std::string>() +
                                                 " (Phoenix DLSS5 " + provider + ")"});
        }
    } catch (const std::exception&) { /* Missing/invalid SDK catalogue stays unavailable. */
    }
}
inline std::vector<ExecutionDevice> cuda_execution_devices() {
    std::vector<ExecutionDevice> result{{"native", "Rendering GPU (optimized NVIDIA)"},
                                        {"cpu", "CPU (reference, slow)"}};
    HMODULE library = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library)
        return result;
    struct Close {
        HMODULE library_handle;
        ~Close() {
            FreeLibrary(library_handle);
        }
    } close{library};
    using Init = int(__stdcall*)(unsigned);
    using Count = int(__stdcall*)(int*);
    using Device = int(__stdcall*)(int*, int);
    using Name = int(__stdcall*)(char*, int, int);
    struct Uuid {
        unsigned char bytes[16];
    };
    using GetUuid = int(__stdcall*)(Uuid*, int);
    using DeviceLuid = int(__stdcall*)(char*, unsigned*, int);
    using Capability = int(__stdcall*)(int*, int*, int);
    auto init = reinterpret_cast<Init>(GetProcAddress(library, "cuInit"));
    auto count = reinterpret_cast<Count>(GetProcAddress(library, "cuDeviceGetCount"));
    auto device = reinterpret_cast<Device>(GetProcAddress(library, "cuDeviceGet"));
    auto name = reinterpret_cast<Name>(GetProcAddress(library, "cuDeviceGetName"));
    auto uuid = reinterpret_cast<GetUuid>(GetProcAddress(library, "cuDeviceGetUuid_v2"));
    auto luid = reinterpret_cast<DeviceLuid>(GetProcAddress(library, "cuDeviceGetLuid"));
    auto capability = reinterpret_cast<Capability>(GetProcAddress(library, "cuDeviceComputeCapability"));
    if (!uuid)
        uuid = reinterpret_cast<GetUuid>(GetProcAddress(library, "cuDeviceGetUuid"));
    int device_count = 0;
    if (!init || !count || !device || !name || !uuid || init(0) || count(&device_count) ||
        device_count < 0 || device_count > 64)
        return result;
    for (int iteration_index = 0; iteration_index < device_count; ++iteration_index) {
        int cuda_device = 0;
        char label[256]{};
        Uuid uuid_bytes{};
        if (device(&cuda_device, iteration_index) || name(label, 256, cuda_device) ||
            uuid(&uuid_bytes, cuda_device))
            continue;
        // Reuse this existing discovery pass. The first native frame need not
        // reload/reinitialize CUDA solely to identify its instruction options.
        LUID adapter{}; unsigned node_mask = 0; int major = 0, minor = 0;
        if (luid && capability && !luid(reinterpret_cast<char*>(&adapter), &node_mask, cuda_device) &&
            node_mask && !capability(&major, &minor, cuda_device))
            phoenix_kernel::remember_device_architecture(adapter, unsigned(major * 10 + minor));
        std::string device_id = "cuda:GPU-";
        for (unsigned uuid_index = 0; uuid_index < 16; ++uuid_index) {
            if (uuid_index == 4 || uuid_index == 6 || uuid_index == 8 || uuid_index == 10)
                device_id += '-';
            char pair[3]{};
            snprintf(pair, sizeof(pair), "%02x", uuid_bytes.bytes[uuid_index]);
            device_id += pair;
        }
        result.push_back({device_id, std::string(label) + " (portable reference)"});
    }
    return result;
}
inline std::vector<ExecutionDevice>
execution_devices(const std::filesystem::path& vulkan_build = {},
                  const std::filesystem::path& hip_build = {},
                  const std::filesystem::path& sycl_build = {}) {
    auto result = cuda_execution_devices();
    append_native_devices(result, hip_build, "hip");
    append_native_devices(result, sycl_build, "sycl");
    // Vulkan discovery is independent of CUDA; AMD/Intel-only systems reach here.
    if (vulkan_build.empty() || !vulkan_build.is_absolute())
        return result;
    const auto path = vulkan_build / "phoenix_vulkan.dll";
    if (!std::filesystem::is_regular_file(path))
        return result;
    HMODULE library = LoadLibraryExW(
        path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library)
        return result;
    struct Close {
        HMODULE library_handle;
        ~Close() {
            FreeLibrary(library_handle);
        }
    } close{library};
    using Count = int(__cdecl*)();
    using Info = int(__cdecl*)(unsigned, char*, char*, unsigned*, unsigned*);
    auto count = reinterpret_cast<Count>(GetProcAddress(library, "phoenix_device_count"));
    auto info = reinterpret_cast<Info>(GetProcAddress(library, "phoenix_device_info"));
    if (!count || !info)
        return result;
    const int device_count = count();
    if (device_count < 0 || device_count > 64)
        return result;
    for (int iteration_index = 0; iteration_index < device_count; ++iteration_index) {
        char uuid[33]{}, name[256]{};
        unsigned vendor = 0, capable = 0;
        if (info(unsigned(iteration_index), uuid, name, &vendor, &capable) || !capable)
            continue;
        uuid[32] = 0;
        name[255] = 0;
        const std::string device_id = "vulkan:" + std::string(uuid);
        if (valid_execution_device(device_id))
            result.push_back({device_id, std::string(name) + " (Phoenix DLSS5 Vulkan)"});
    }
    return result;
}
}
