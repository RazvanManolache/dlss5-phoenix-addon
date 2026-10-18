#pragma once
#include <windows.h>
#include <compressapi.h>
#include <shlobj.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <atomic>
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
    return path.make_preferred();
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
inline void check_cancelled(const std::atomic<bool>* cancelled) {
    require(!cancelled || !cancelled->load(), "Runtime preparation cancelled");
}
inline DWORD wait_cache_lock(HANDLE mutex, const std::atomic<bool>* cancelled,
                             DWORD timeout = 60000) {
    const auto deadline = GetTickCount64() + timeout;
    do {
        if (cancelled && cancelled->load()) return WAIT_FAILED;
        const auto now = GetTickCount64();
        if (now >= deadline) return WAIT_TIMEOUT;
        const auto result = WaitForSingleObject(mutex, DWORD((std::min)(uint64_t(50), deadline - now)));
        if (result != WAIT_TIMEOUT) return result;
    } while (true);
}
inline std::string file_digest(const fs::path& path, const std::atomic<bool>* cancelled = nullptr) {
    check_cancelled(cancelled);
    HANDLE file = CreateFileW(native_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return {};
    struct Cleanup {
        HANDLE file;
        ~Cleanup() {
            CloseHandle(file);
        }
    } cleanup{file};
    LARGE_INTEGER bytes{};
    if (!GetFileSizeEx(file, &bytes) || bytes.QuadPart <= 0 || bytes.QuadPart > ULONG_MAX)
        return {};
    // Stream cached binaries: mapping a large core competes with a 32-bit
    // game's limited, fragmented address space even though the file is on disk.
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &hash, nullptr, 0, nullptr, 0, 0) < 0)
        return {};
    struct HashCleanup {
        BCRYPT_HASH_HANDLE value;
        ~HashCleanup() {
            BCryptDestroyHash(value);
        }
    } hash_cleanup{hash};
    std::array<uint8_t, 65536> buffer{};
    uint64_t consumed = 0;
    DWORD read = 0;
    do {
        check_cancelled(cancelled);
        if (!ReadFile(file, buffer.data(), DWORD(buffer.size()), &read, nullptr))
            return {};
        if (read && BCryptHashData(hash, buffer.data(), read, 0) < 0)
            return {};
        consumed += read;
    } while (read);
    uint8_t digest[32]{};
    if (consumed != uint64_t(bytes.QuadPart) || BCryptFinishHash(hash, digest, 32, 0) < 0)
        return {};
    constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (unsigned index = 0; index < 32; ++index) {
        result[index * 2] = hex[digest[index] >> 4];
        result[index * 2 + 1] = hex[digest[index] & 15];
    }
    return result;
}
inline fs::path default_cache() {
    PWSTR path = nullptr;
    require(SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &path)),
            "Phoenix DLSS5 cannot locate the local runtime cache");
    fs::path result(path);
    CoTaskMemFree(path);
    const auto logical = result / L"DLSS5Phoenix" / L"runtime";
    fs::create_directories(logical);
    // Packaged launchers can merge existing AppData with redirected new files.
    // Resolve a newly created file, not the merged parent, before choosing the
    // cache root. Explicit caller-supplied caches retain strict containment.
    static std::atomic<uint64_t> probe_sequence{0};
    const auto probe = logical / (L".location-" + std::to_wstring(GetCurrentProcessId()) +
        L"-" + std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(++probe_sequence));
    HANDLE file = CreateFileW(native_path(probe).c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    require(file != INVALID_HANDLE_VALUE, "Cannot resolve the writable runtime cache");
    struct CloseProbe { HANDLE file; ~CloseProbe() { CloseHandle(file); } } close{file};
    std::wstring physical(32768, L'\0');
    const auto count = GetFinalPathNameByHandleW(file, physical.data(), DWORD(physical.size()),
                                                FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    require(count && count < physical.size(), "Cannot resolve the physical runtime cache");
    physical.resize(count);
    // Keep the extended prefix: redirected package caches are substantially
    // deeper and optimized kernel filenames can exceed MAX_PATH.
    return fs::path(native_path(fs::canonical(fs::path(physical).parent_path())));
}
inline const char* dependency_group(const std::string& path) {
    if (path.rfind("compatibility/", 0) == 0)
        return "compatibility";
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
                            const std::string& backend = "core",
                            const std::atomic<bool>* cancelled = nullptr) {
    check_cancelled(cancelled);
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
    const auto wait = wait_cache_lock(mutex, cancelled);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) {
        CloseHandle(mutex);
        check_cancelled(cancelled);
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
        check_cancelled(cancelled);
        const auto name = entry.at("path").get<std::string>();
        const auto group = dependency_group(name);
        if (backend == "compatibility" && std::strcmp(group, "compatibility") != 0)
            continue;
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
        if (file_digest(path, cancelled) == expected)
            continue;
        const auto [packed, packed_bytes] = resource(module, entry.at("resource").get<unsigned>());
        require(packed && packed_bytes == entry.at("packed_bytes").get<size_t>(),
                "Missing embedded runtime payload");
        const auto count = entry.at("bytes").get<size_t>();
        require(count > 0 && count <= 512u * 1024u * 1024u, "Embedded runtime file extent");
        std::vector<uint8_t> decompressed;
        const uint8_t* output = packed;
        const auto algorithm = entry.at("algorithm").get<std::string>();
        if (algorithm == "raw") {
            require(count == packed_bytes, "Embedded raw file extent");
        } else {
            check_cancelled(cancelled);
            decompressed.resize(count);
            output = decompressed.data();
            require(algorithm == "xpress_huff", "Unknown runtime compression");
            DECOMPRESSOR_HANDLE decoder{};
            require(CreateDecompressor(COMPRESS_ALGORITHM_XPRESS_HUFF, nullptr, &decoder) != FALSE,
                    "Runtime decompressor unavailable");
            SIZE_T written = 0;
            const auto success =
                Decompress(decoder, packed, packed_bytes, decompressed.data(), count, &written);
            CloseDecompressor(decoder);
            require(success && written == count, "Embedded runtime decompression failed");
        }
        check_cancelled(cancelled);
        require(phoenix_model::weight_digest(output, count) == expected,
                "Embedded runtime checksum mismatch");
        auto temporary = path;
        temporary += L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                     std::to_wstring(GetTickCount64());
        HANDLE file = CreateFileW(native_path(temporary).c_str(), GENERIC_WRITE, 0, nullptr,
                                  CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(file != INVALID_HANDLE_VALUE, "Runtime cache temporary file unavailable");
        bool success = true;
        for (size_t offset = 0; offset < count;) {
            if (cancelled && cancelled->load()) { success = false; break; }
            const auto chunk = DWORD((std::min)(count - offset, size_t(65536)));
            DWORD written = 0;
            if (!WriteFile(file, output + offset, chunk, &written, nullptr) || written != chunk) {
                success = false;
                break;
            }
            offset += written;
        }
        success = success && FlushFileBuffers(file);
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
