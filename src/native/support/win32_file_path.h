#pragma once
#ifdef _WIN32
#include <filesystem>
#include <string>

namespace phoenix_windows {
// File access from an addon must also work in games without a long-path-aware manifest.
inline std::wstring native_path(const std::filesystem::path& path) {
    auto absolute = std::filesystem::absolute(path).lexically_normal().make_preferred().wstring();
    if (absolute.rfind(L"\\\\?\\", 0) == 0)
        return absolute;
    if (absolute.rfind(L"\\\\", 0) == 0)
        return L"\\\\?\\UNC\\" + absolute.substr(2);
    return L"\\\\?\\" + absolute;
}
}
#endif
