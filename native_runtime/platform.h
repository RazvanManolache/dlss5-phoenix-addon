// OS boundary for standalone runners. The ReShade integration does not use this loader.
#pragma once
#include <algorithm>
#include <cstring>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#define PHOENIX_EXPORT __declspec(dllexport)
#else
#include <dlfcn.h>
#define PHOENIX_EXPORT __attribute__((visibility("default")))
#endif
namespace phoenix_cpp {
#ifdef _WIN32
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void *;
#endif
inline LibraryHandle open_library(const std::filesystem::path &path) {
#ifdef _WIN32
    return LoadLibraryExW(path.c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}
inline void close_library(LibraryHandle handle) {
    if (!handle)
        return;
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}
inline auto library_symbol(LibraryHandle handle, const char *name) {
#ifdef _WIN32
    return GetProcAddress(handle, name);
#else
    return dlsym(handle, name);
#endif
}
inline std::string library_name(const std::string &stem) {
#ifdef _WIN32
    return stem + ".dll";
#elif defined(__APPLE__)
    return "lib" + stem + ".dylib";
#else
    return "lib" + stem + ".so";
#endif
}
inline void copy_error(char *destination, size_t capacity, const char *message) {
    if (!destination || !capacity)
        return;
    const auto length = std::min(capacity - 1, std::strlen(message));
    std::memcpy(destination, message, length);
    destination[length] = '\0';
}
} // namespace phoenix_cpp
