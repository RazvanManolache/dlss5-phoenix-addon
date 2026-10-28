// Use the real addon and real MinHook, with a CPU-only NGX stand-in.
#define wmain packaged_load_main
#include "packaged_addon_load_tests.cpp"
#include <cstring>
#undef wmain
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    try {
        require(argc == 3, "Usage: ADDON NGX_STUB");
        auto ngx = LoadLibraryW(argv[2]);
        require(ngx != nullptr, "NGX stand-in failed to load");
        require(GetModuleHandleW(L"_nvngx.dll") == ngx, "NGX stand-in module identity mismatch");
        auto create = reinterpret_cast<int(__cdecl*)(void*, int, void*, void*)>(
            GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature"));
        auto calls = reinterpret_cast<unsigned(__cdecl*)()>(GetProcAddress(ngx, "ProbeCalls"));
        require(create && calls, "NGX stand-in exports missing");
        unsigned char original_code[16];
        std::memcpy(original_code, reinterpret_cast<void*>(create), sizeof(original_code));
        for (unsigned cycle = 0; cycle < 4; ++cycle) {
            auto addon = LoadLibraryW(argv[1]);
            require(addon != nullptr, "Addon failed to load");
            auto init = reinterpret_cast<bool(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonInit"));
            auto uninit = reinterpret_cast<void(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonUninit"));
            require(init && uninit && init(addon, GetModuleHandleW(nullptr)), "Addon init failed");
            require(addons == 1 && events > 0, "Probe registration unbalanced");
            require(std::memcmp(original_code, reinterpret_cast<void*>(create), sizeof(original_code)) != 0,
                    "NGX stand-in was not actually hooked");
            create(nullptr, 0, nullptr, nullptr);
            require(calls() == cycle + 1, "NGX call recursed or failed to forward exactly once");
            uninit(addon, GetModuleHandleW(nullptr));
            require(addons == 0 && events == 0 && overlays == 0, "Probe callbacks survived detach");
            require(FreeLibrary(addon) != FALSE, "Probe reference release failed");
        }
        std::cout << "{\"passed\":true,\"cpu_only\":true,\"device_probe_cycles\":4,\"forwarded_calls\":"
                  << calls() << ",\"real_addon_and_minhook\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
