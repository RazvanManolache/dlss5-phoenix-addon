#define wmain unused_load_test_main
#include "packaged_addon_load_tests.cpp"
#undef wmain
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    try {
        require(argc == 3, "Usage: ADDON SECOND_PROVIDER");
        auto second = LoadLibraryW(argv[2]);
        require(second != nullptr, "Provider DLL load failed");
        auto balanced = reinterpret_cast<bool(*)(bool)>(GetProcAddress(second, "ProviderBalanced"));
        auto addon = LoadLibraryW(argv[1]);
        require(addon != nullptr, "Addon load failed");
        auto init = reinterpret_cast<bool(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonInit"));
        auto uninit = reinterpret_cast<void(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonUninit"));
        auto first = GetModuleHandleW(nullptr);
        require(balanced && init && uninit, "Test exports missing");
        for (int cycle = 0; cycle < 3; ++cycle) {
            require(init(addon, first), "First provider registration failed");
            require(addons == 1 && events > 0 && overlays == 2 && balanced(false), "First provider imbalance");
            require(init(addon, second), "Second provider registration failed");
            require(addons == 0 && events == 0 && overlays == 0 && balanced(true), "Cached exports used wrong provider");
            uninit(addon, first); // A stale probe must not detach the live host.
            require(balanced(true), "Stale provider detached live addon");
            uninit(addon, second);
            require(balanced(false), "Second provider callbacks leaked");
        }
        FreeLibrary(addon);
        FreeLibrary(second);
        std::cout << "{\"passed\":true,\"cpu_only\":true,\"provider_switches\":6}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
