// Optional UI metadata only. No network calls on the rendering thread or in DllMain.
#pragma once
#include "release_version.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "shell32.lib")
namespace phoenix_updates {
inline constexpr char current_version[] = "1.0-beta";
inline constexpr wchar_t releases_url[] = L"https://github.com/RazvanManolache/dlss5-phoenix-addon/releases";
inline std::string newer_release(std::string_view body, std::string_view current = current_version) {
    if (body.size() > 256 * 1024) return {};
    // Bound parser recursion before handing remote text to the JSON library.
    unsigned depth = 0; bool quoted = false, escaped = false;
    for (char value : body) {
        if (quoted) { if (escaped) escaped = false; else if (value == '\\') escaped = true; else if (value == '"') quoted = false; }
        else if (value == '"') quoted = true;
        else if (value == '[' || value == '{') { if (++depth > 16) return {}; }
        else if (value == ']' || value == '}') { if (!depth) return {}; --depth; }
    }
    if (quoted || depth) return {};
    const auto installed = release_updates::parse(current);
    if (!installed) return {};
    auto best = *installed;
    std::string tag;
    const auto feed = nlohmann::json::parse(body, nullptr, false);
    if (!feed.is_array() || feed.size() > 100) return {};
    for (const auto& entry : feed) {
        if (!entry.is_object() || !entry.contains("draft") || !entry["draft"].is_boolean() || entry["draft"].get<bool>() ||
            !entry.contains("published_at") || !entry["published_at"].is_string() ||
            !entry.contains("tag_name") || !entry["tag_name"].is_string()) continue;
        const auto candidate = entry["tag_name"].get<std::string>();
        const auto version = release_updates::parse(candidate);
        if (version && *version > best) { best = *version; tag = candidate; }
    }
    return tag;
}
struct InternetHandle {
    HINTERNET value{};
    ~InternetHandle() { if (value) WinHttpCloseHandle(value); }
};
class Service {
    std::mutex mutex;
    std::string latest;
    HANDLE worker{};
    std::atomic<bool> cancelled{false};
    std::atomic<HINTERNET> request{};
    std::chrono::steady_clock::time_point next{};
    struct Work { Service* service; HMODULE module; };
    void close_request() { if (auto handle = request.exchange(nullptr)) WinHttpCloseHandle(handle); }
    std::string fetch() {
        InternetHandle session{WinHttpOpen(L"DLSS5-Phoenix-update-notice", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
            WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0)};
        if (!session.value || cancelled) return {};
        if (!WinHttpSetTimeouts(session.value, 2000, 2000, 2000, 2000)) return {};
        const auto deadline = GetTickCount64() + 10000;
        InternetHandle connection{WinHttpConnect(session.value, L"api.github.com", INTERNET_DEFAULT_HTTPS_PORT, 0)};
        if (!connection.value || cancelled) return {};
        auto handle = WinHttpOpenRequest(connection.value, L"GET",
            L"/repos/RazvanManolache/dlss5-phoenix-addon/releases?per_page=10", nullptr,
            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!handle) return {};
        request.store(handle);
        struct CloseRequest { Service* owner; ~CloseRequest() { owner->close_request(); } } close{this};
        DWORD redirects = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        if (cancelled || !WinHttpSetOption(handle, WINHTTP_OPTION_REDIRECT_POLICY, &redirects, sizeof redirects) ||
            !WinHttpSendRequest(handle, L"Accept: application/vnd.github+json\r\nX-GitHub-Api-Version: 2022-11-28", DWORD(-1),
                nullptr, 0, 0, 0) || !WinHttpReceiveResponse(handle, nullptr)) return {};
        DWORD status{}, bytes = sizeof status;
        if (!WinHttpQueryHeaders(handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &bytes, WINHTTP_NO_HEADER_INDEX) || status != 200) return {};
        std::string body;
        char buffer[4096]; DWORD read{};
        while (!cancelled && GetTickCount64() < deadline) {
            if (!WinHttpReadData(handle, buffer, sizeof buffer, &read)) return {};
            if (!read) return body;
            if (body.size() + read > 256 * 1024) return {};
            body.append(buffer, read);
        }
        return {};
    }
    static DWORD WINAPI run(void* parameter) {
        auto work = std::unique_ptr<Work>(static_cast<Work*>(parameter));
        auto* self = work->service;
        try {
            const auto body = self->fetch();
            const auto tag = newer_release(body);
            std::lock_guard lock(self->mutex);
            if (!self->cancelled) {
                if (!body.empty()) self->latest = tag;
                self->next = std::chrono::steady_clock::now() +
                    (body.empty() ? std::chrono::hours(1) : std::chrono::hours(24));
            }
        } catch (...) { /* Optional metadata must never break a game. */ }
        const auto module = work->module;
        work.reset();
        // Retain the addon while this callback runs, including hosts that unload
        // without AddonUninit. Release the final reference only after all C++ cleanup.
        FreeLibraryAndExitThread(module, 0);
    }
public:
    // A host that bypasses AddonUninit can unload us when the retained worker
    // reference is released. Closing its kernel handle never joins under DllMain.
    ~Service() { if (worker) CloseHandle(worker); }
    std::string poll() {
        std::lock_guard lock(mutex);
        if (worker && WaitForSingleObject(worker, 0) == WAIT_OBJECT_0) { CloseHandle(worker); worker = nullptr; }
        const auto now = std::chrono::steady_clock::now();
        if (!worker && now >= next) {
            next = now + std::chrono::hours(1);
            HMODULE module{};
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(&run), &module)) {
                cancelled = false;
                auto work = std::make_unique<Work>(Work{this, module});
                worker = CreateThread(nullptr, 0, run, work.get(), 0, nullptr);
                if (worker) work.release(); else FreeLibrary(module);
            }
        }
        return latest;
    }
    // Called by AddonUninit, outside the loader lock. Process exit never joins.
    void stop() {
        cancelled = true;
        close_request();
        if (worker) { WaitForSingleObject(worker, INFINITE); CloseHandle(worker); worker = nullptr; }
    }
};
inline Service& service() { static Service state; return state; }
inline void open_releases() { ShellExecuteW(nullptr, L"open", releases_url, nullptr, nullptr, SW_SHOWNORMAL); }
} // namespace phoenix_updates
