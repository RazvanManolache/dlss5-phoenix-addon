#pragma once
#include <windows.h>
#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>
#include "phoenix.h"

namespace phoenix_cpp {
inline std::string path_utf8(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}
// Version 1 uses contiguous RGB32F / RG32F / R32F host planes. The interface
// owns no caller memory and completes all work before frame() returns.
struct RuntimeApi {
    HMODULE module = nullptr;
    void* instance = nullptr;
    void* (*create)(const char*, unsigned, unsigned) = nullptr;
    void (*destroy)(void*) = nullptr;
    int (*frame)(void*, const float*, const float*, const float*, const float*, const char*,
                 float*) = nullptr;
    int (*frame_v2)(void*, const phoenix_frame_v2*) = nullptr;
    int (*frame_v3)(void*, const phoenix_frame_v3*) = nullptr;
    const char* (*error)() = nullptr;
    size_t (*status)(void*, char*, size_t) = nullptr;
    size_t (*capabilities)(char*, size_t) = nullptr;

    RuntimeApi() = default;
    RuntimeApi(const RuntimeApi&) = delete;
    RuntimeApi& operator=(const RuntimeApi&) = delete;
    ~RuntimeApi() {
        if (instance && destroy)
            destroy(instance);
        if (module)
            FreeLibrary(module);
    }
    void open(const std::filesystem::path& root, const std::string& config, unsigned width,
              unsigned height) {
        if (module)
            throw std::runtime_error("Runtime API already loaded");
        module =
            LoadLibraryExW((root / L"phoenix_runtime_cpp.dll").c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!module)
            throw std::runtime_error("Native runtime library unavailable (Windows error " +
                std::to_string(GetLastError()) + "); use a runtime matching the game's bitness");
        create =
            reinterpret_cast<decltype(create)>(GetProcAddress(module, "phoenix_runtime_create"));
        destroy =
            reinterpret_cast<decltype(destroy)>(GetProcAddress(module, "phoenix_runtime_destroy"));
        frame = reinterpret_cast<decltype(frame)>(GetProcAddress(module, "phoenix_runtime_frame"));
        frame_v2 = reinterpret_cast<decltype(frame_v2)>(GetProcAddress(module, "phoenix_runtime_frame_v2"));
        frame_v3 = reinterpret_cast<decltype(frame_v3)>(GetProcAddress(module, "phoenix_runtime_frame_v3"));
        capabilities = reinterpret_cast<decltype(capabilities)>(GetProcAddress(module, "phoenix_runtime_capabilities"));
        error = reinterpret_cast<decltype(error)>(GetProcAddress(module, "phoenix_runtime_error"));
        status =
            reinterpret_cast<decltype(status)>(GetProcAddress(module, "phoenix_runtime_status"));
        if (!create || !destroy || !frame || !error || !status)
            throw std::runtime_error("Native runtime ABI mismatch");
        instance = create(config.c_str(), width, height);
        if (!instance)
            throw std::runtime_error(error());
    }
    std::string status_json() {
        std::vector<char> text(status(instance, nullptr, 0));
        if (text.empty() || text.size() > 1024 * 1024)
            throw std::runtime_error("Invalid runtime status size");
        status(instance, text.data(), text.size());
        return text.data();
    }
};
}
