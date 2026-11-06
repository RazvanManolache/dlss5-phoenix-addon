#pragma once
#include <filesystem>
#include <string>
#include <string_view>

namespace phoenix_fs {
// Preserve UTF-8 semantics on Windows as well as POSIX. A narrow path(string)
// would interpret JSON and command-line paths using the Windows code page.
inline std::filesystem::path from_utf8(std::string_view bytes) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(bytes.begin(), bytes.end()));
#else
    return std::filesystem::u8path(bytes.begin(), bytes.end());
#endif
}
}
