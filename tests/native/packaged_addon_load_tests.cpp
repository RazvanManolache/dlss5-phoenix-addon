// Load the real DLL against a minimal ReShade registration host. Rendering and
// widgets are tested separately; these stubs do not simulate a game runtime.
#include <windows.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
static int addons = 0, events = 0, overlays = 0;
#if defined(_M_IX86)
#pragma comment(linker, "/EXPORT:ReShadeRegisterAddon=_ReShadeRegisterAddon")
#pragma comment(linker, "/EXPORT:ReShadeUnregisterAddon=_ReShadeUnregisterAddon")
#pragma comment(linker, "/EXPORT:ReShadeLogMessage=_ReShadeLogMessage")
#pragma comment(linker, "/EXPORT:ReShadeGetImGuiFunctionTable=_ReShadeGetImGuiFunctionTable")
#pragma comment(linker, "/EXPORT:ReShadeRegisterEvent=_ReShadeRegisterEvent")
#pragma comment(linker, "/EXPORT:ReShadeUnregisterEvent=_ReShadeUnregisterEvent")
#pragma comment(linker, "/EXPORT:ReShadeRegisterOverlay=_ReShadeRegisterOverlay")
#pragma comment(linker, "/EXPORT:ReShadeUnregisterOverlay=_ReShadeUnregisterOverlay")
#endif
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void*, uint32_t) { ++addons; return true; }
extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void*) { --addons; }
extern "C" __declspec(dllexport) void ReShadeLogMessage(void*, int, const char*) {}
extern "C" __declspec(dllexport) const void* ReShadeGetImGuiFunctionTable(uint32_t) {
    // Addon attach only checks this pointer. No drawing callbacks are invoked.
    static void* table[4096]{}; return table;
}
extern "C" __declspec(dllexport) void ReShadeRegisterEvent(uint32_t, void*) { ++events; }
extern "C" __declspec(dllexport) void ReShadeUnregisterEvent(uint32_t, void*) { --events; }
extern "C" __declspec(dllexport) void ReShadeRegisterOverlay(const char*, void*) { ++overlays; }
extern "C" __declspec(dllexport) void ReShadeUnregisterOverlay(const char*, void*) { --overlays; }
static void require(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE addon = nullptr;
    try {
        require(argc == 2 || argc == 3, "Usage: ADDON [stable|experimental]");
        const std::wstring channel = argc == 3 ? argv[2] : L"experimental";
        require(channel == L"stable" || channel == L"experimental", "Unknown expected release channel");
        const bool experimental = channel == L"experimental";
        addon = LoadLibraryW(argv[1]);
        require(addon != nullptr, "Experimental addon did not attach");
        require(addons == 1 && events > 0 && overlays == 2, "ReShade callbacks were not registered");
        auto name = reinterpret_cast<const char**>(GetProcAddress(addon, "NAME"));
        require(name && (std::string(*name).find("experimental") != std::string::npos) == experimental,
                "Addon identity does not match its release channel");
        using Version = uint32_t(WINAPI*)();
        auto version = reinterpret_cast<Version>(GetProcAddress(addon, "CnrPHOENIXFeederVersion"));
        auto version2 = reinterpret_cast<Version>(GetProcAddress(addon, "CnrPHOENIXFeederVersionV2"));
        require(version && version() == 1 && version2 && version2() == 2, "Feeder ABI version mismatch");
        auto init = reinterpret_cast<bool(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonInit"));
        auto uninit = reinterpret_cast<void(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonUninit"));
        require(init && uninit && init(addon, GetModuleHandleW(nullptr)) && addons == 1,
                "Explicit initialization duplicated registration");
        const int registered_events = events;
        uninit(addon, GetModuleHandleW(nullptr));
        require(addons == 0 && events == 0 && overlays == 0, "Addon left registered callbacks behind");
        require(FreeLibrary(addon) != FALSE, "Addon did not unload"); addon = nullptr;
        require(addons == 0 && events == 0 && overlays == 0, "DLL detach repeated unregister");
        std::cout << "{\"passed\":true,\"actual_dll_loaded\":true,\"experimental_identity\":"
                  << (experimental ? "true" : "false") << ",\"release_channel_checked\":true,"
                     "\"feeder_abi_checked\":true,\"balanced_attach_detach\":true,\"registered_events\":"
                  << registered_events << ",\"minimal_reshade_host\":true,\"game_presentation_validated\":false}\n";
    } catch (const std::exception& error) {
        if (addon) FreeLibrary(addon);
        std::cerr << error.what() << '\n'; return 1;
    }
}
