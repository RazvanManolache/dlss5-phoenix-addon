#pragma once
#include <windows.h>
#include <compressapi.h>
#include <shlobj.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <set>
#include "../model/weight_source.h"

namespace phoenix_embedded {
namespace fs = std::filesystem;
using Json = nlohmann::json;
struct Installation {
    fs::path root, runtime, native;
    std::string identity;
    size_t files = 0;
};
inline void require(bool success, const char* message) {
    if (!success)
        throw std::runtime_error(message);
}
inline std::pair<const uint8_t*, size_t> resource(HMODULE module, unsigned resource_id) {
    auto entry = FindResourceW(module, MAKEINTRESOURCEW(resource_id), MAKEINTRESOURCEW(10));
    if (!entry)
        return {};
    const auto size = SizeofResource(module, entry);
    const auto data = static_cast<const uint8_t*>(LockResource(LoadResource(module, entry)));
    require(data && size, "Embedded runtime resource is unreadable");
    return {data, size};
}
inline fs::path relative_path(const std::string& text) {
    auto path = fs::u8path(text);
    require(!text.empty() && text.find('\0') == std::string::npos &&
                text.find(':') == std::string::npos && path.is_relative() && !path.has_root_path(),
            "Invalid embedded runtime path");
    for (const auto& part : path)
        require(part != ".." && part != ".", "Embedded runtime path traversal");
    return path;
}
// Explicit Win32 calls do not inherit std::filesystem's long-path handling.
// Normalize separators before applying the extended local/UNC path prefix.
inline std::wstring native_path(const fs::path& path) {
    auto absolute = fs::absolute(path).lexically_normal().make_preferred().wstring();
    if (absolute.rfind(L"\\\\?\\", 0) == 0)
        return absolute;
    if (absolute.rfind(L"\\\\", 0) == 0)
        return L"\\\\?\\UNC\\" + absolute.substr(2);
    return L"\\\\?\\" + absolute;
}
inline std::string file_digest(const fs::path& path) {
    HANDLE file = CreateFileW(native_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return {};
    HANDLE section = nullptr;
    const uint8_t* data = nullptr;
    struct Cleanup {
        HANDLE file;
        HANDLE& section;
        const uint8_t*& data;
        ~Cleanup() {
            if (data)
                UnmapViewOfFile(data);
            if (section)
                CloseHandle(section);
            CloseHandle(file);
        }
    } cleanup{file, section, data};
    LARGE_INTEGER bytes{};
    if (!GetFileSizeEx(file, &bytes) || bytes.QuadPart <= 0 || bytes.QuadPart > ULONG_MAX)
        return {};
    section = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (section)
        data = static_cast<const uint8_t*>(MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0));
    return data ? phoenix_model::weight_digest(data, size_t(bytes.QuadPart)) : std::string();
}
inline fs::path default_cache() {
    PWSTR path = nullptr;
    require(SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &path)),
            "DLSS5 Phoenix cannot locate the local runtime cache");
    fs::path result(path);
    CoTaskMemFree(path);
    return result / L"DLSS5Phoenix" / L"runtime";
}
inline const char* dependency_group(const std::string& path) {
    if (path == "runtime/amdhip64_7.dll" || path == "runtime/amd_comgr.dll" ||
        path == "runtime/rocm_kpack.dll")
        return "hip";
    if (path.rfind("runtime/opencl_cpu/", 0) == 0 || path == "runtime/OpenCL.dll" ||
        path == "runtime/sycl9.dll" || path == "runtime/ur_adapter_opencl.dll" ||
        path == "runtime/ur_loader.dll" || path == "runtime/ur_win_proxy_loader.dll")
        return "sycl";
    return "core";
}
inline Installation install(HMODULE module, fs::path cache = {},
                            const std::string& backend = "core") {
    const auto [data, bytes] = resource(module, 1001);
    if (!data)
        return {}; // Development builds may deliberately use an external runtime.
    auto manifest = Json::parse(data, data + bytes);
    require(manifest.at("schema") == "dlss5-phoenix-embedded-runtime/1" &&
                manifest.at("model_data_included") == false &&
                manifest.at("nvidia_dll_included") == false,
            "Unsupported embedded runtime manifest");
    const auto identity = phoenix_model::weight_digest(data, bytes);
    if (cache.empty())
        cache = default_cache();
    require(cache.is_absolute(), "Embedded cache path must be absolute");
    const auto root = cache / fs::u8path(identity);
    const auto mutex_name =
        L"Local\\DLSS5Phoenix-runtime-" + std::wstring(identity.begin(), identity.end());
    HANDLE mutex = CreateMutexW(nullptr, FALSE, mutex_name.c_str());
    require(mutex != nullptr, "Runtime cache lock unavailable");
    const auto wait = WaitForSingleObject(mutex, 60000);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) {
        CloseHandle(mutex);
        throw std::runtime_error("Runtime cache is busy");
    }
    struct Unlock {
        HANDLE mutex;
        ~Unlock() {
            ReleaseMutex(mutex);
            CloseHandle(mutex);
        }
    } unlock{mutex};
    fs::create_directories(root);
    std::set<fs::path> names;
    for (const auto& entry : manifest.at("files")) {
        const auto name = entry.at("path").get<std::string>();
        const auto group = dependency_group(name);
        if (backend != "all" && std::strcmp(group, "core") != 0 && backend != group)
            continue;
        const auto relative = relative_path(name);
        require(names.insert(relative).second, "Duplicate embedded runtime file");
        const auto path = root / relative;
        fs::create_directories(path.parent_path());
        // Reject redirected cache directories before reading or replacing a file.
        auto canonical_root = fs::canonical(root);
        auto parent = fs::canonical(path.parent_path());
        auto child = parent.begin();
        for (auto base = canonical_root.begin(); base != canonical_root.end(); ++base, ++child)
            require(child != parent.end() && _wcsicmp(base->c_str(), child->c_str()) == 0,
                    "Runtime cache directory resolves outside its version");
        const auto expected = entry.at("sha256").get<std::string>();
        std::error_code error;
        require(!fs::is_symlink(fs::symlink_status(path, error)),
                "Runtime cache file is redirected");
        if (file_digest(path) == expected)
            continue;
        const auto [packed, packed_bytes] = resource(module, entry.at("resource").get<unsigned>());
        require(packed && packed_bytes == entry.at("packed_bytes").get<size_t>(),
                "Missing embedded runtime payload");
        const auto count = entry.at("bytes").get<size_t>();
        require(count > 0 && count <= 512u * 1024u * 1024u, "Embedded runtime file extent");
        std::vector<uint8_t> output(count);
        const auto algorithm = entry.at("algorithm").get<std::string>();
        if (algorithm == "raw") {
            require(count == packed_bytes, "Embedded raw file extent");
            std::memcpy(output.data(), packed, count);
        } else {
            require(algorithm == "xpress_huff", "Unknown runtime compression");
            DECOMPRESSOR_HANDLE decoder{};
            require(CreateDecompressor(COMPRESS_ALGORITHM_XPRESS_HUFF, nullptr, &decoder) != FALSE,
                    "Runtime decompressor unavailable");
            SIZE_T written = 0;
            const auto success =
                Decompress(decoder, packed, packed_bytes, output.data(), count, &written);
            CloseDecompressor(decoder);
            require(success && written == count, "Embedded runtime decompression failed");
        }
        require(phoenix_model::weight_digest(output.data(), output.size()) == expected,
                "Embedded runtime checksum mismatch");
        auto temporary = path;
        temporary += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                     std::to_wstring(GetTickCount64());
        HANDLE file = CreateFileW(native_path(temporary).c_str(), GENERIC_WRITE, 0, nullptr,
                                  CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(file != INVALID_HANDLE_VALUE, "Runtime cache temporary file unavailable");
        DWORD written = 0;
        const bool success =
            WriteFile(file, output.data(), DWORD(output.size()), &written, nullptr) &&
            written == output.size() && FlushFileBuffers(file);
        CloseHandle(file);
        if (!success || !MoveFileExW(native_path(temporary).c_str(), native_path(path).c_str(),
                                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            DeleteFileW(native_path(temporary).c_str());
            throw std::runtime_error("Runtime cache publication failed");
        }
    }
    return {root, root / L"runtime", root / L"native", identity, names.size()};
}
} // namespace phoenix_embedded
