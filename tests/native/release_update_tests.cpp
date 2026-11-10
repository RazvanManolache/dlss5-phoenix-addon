#include "../../src/native/support/release_updates.h"
#include <iostream>
#include <stdexcept>
int wmain(int argc, wchar_t** argv) {
    auto check = [](bool value) { if (!value) throw std::runtime_error("Release notifier contract failed"); };
    // Preload an app-local WinHTTP replacement before initializing the updater.
    // The fixture deliberately cannot open sessions; it must never be called.
    HMODULE proxy = argc > 1 ? LoadLibraryW(argv[1]) : nullptr;
    if (argc > 1) check(proxy != nullptr);
    const auto& http = phoenix_updates::winhttp();
    check(http.available && http.module != proxy);
    wchar_t system[MAX_PATH]{}, loaded[MAX_PATH]{};
    check(GetSystemDirectoryW(system, MAX_PATH) != 0);
    check(GetModuleFileNameW(http.module, loaded, MAX_PATH) != 0);
    check(_wcsicmp(loaded, (std::wstring(system) + L"\\winhttp.dll").c_str()) == 0);
    auto session = http.open(L"Phoenix updater DLL routing test", WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    check(session != nullptr);
    check(http.close(session) != FALSE);
    using release_updates::parse;
    check(parse("v1.0-beta") == parse("1.0.0-beta"));
    check(*parse("1.0-beta.10") > *parse("1.0-beta.2"));
    check(*parse("1.0") > *parse("1.0-beta.99"));
    check(!parse("1.0-beta..2") && !parse("../../evil") && !parse("1.2.3.4"));
    const auto feed = R"([{"tag_name":"v1.0-beta.2","draft":false,"published_at":"2026-10-30"},{"tag_name":"v9.0","draft":true,"published_at":"2026-10-30"}])";
    check(phoenix_updates::newer_release(feed, "v1.0-beta") == "v1.0-beta.2");
    check(phoenix_updates::newer_release(feed, "v1.0.0").empty());
    check(phoenix_updates::newer_release("{\"message\":\"API limit\"}").empty());
    check(phoenix_updates::newer_release("[null,5,{}]").empty());
    check(phoenix_updates::newer_release(std::string(20, '[')).empty());
    check(phoenix_updates::newer_release(std::string(256 * 1024 + 1, ' ')).empty());
    // Exercise real cancellation/handle retirement in a bounded Windows process.
    // No game, GPU, settings writes or downloads are involved.
    phoenix_updates::Service service;
    service.poll(); service.stop();
    if (proxy) {
        const auto calls = reinterpret_cast<LONG(*)()>(GetProcAddress(proxy, "PhoenixWinHttpProxyCalls"));
        check(calls && calls() == 0);
        FreeLibrary(proxy);
    }
    std::cout << "{\"passed\":true,\"version_and_feed_checks\":10,\"worker_retired\":true,"
        "\"system_winhttp_verified\":true,\"app_local_proxy_preloaded\":" << (proxy ? "true" : "false") << "}\n";
}
