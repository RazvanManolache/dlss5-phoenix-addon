// Minimal ReShade host for the optional input companion. No game, graphics
// device or input macro is started. The runner checks the mailbox separately.
#include <windows.h>
#include <iostream>
#include <string>
#if defined(_M_IX86)
#pragma comment(linker, "/EXPORT:ReShadeRegisterAddon=_ReShadeRegisterAddon")
#pragma comment(linker, "/EXPORT:ReShadeUnregisterAddon=_ReShadeUnregisterAddon")
#pragma comment(linker, "/EXPORT:ReShadeLogMessage=_ReShadeLogMessage")
#pragma comment(linker, "/EXPORT:ReShadeRegisterEvent=_ReShadeRegisterEvent")
#pragma comment(linker, "/EXPORT:ReShadeUnregisterEvent=_ReShadeUnregisterEvent")
#endif
static int addons = 0, events = 0;
extern "C" __declspec(dllexport) bool ReShadeRegisterAddon(void*, unsigned) { ++addons; return true; }
extern "C" __declspec(dllexport) void ReShadeUnregisterAddon(void*) { --addons; }
extern "C" __declspec(dllexport) void ReShadeLogMessage(void*, int, const char*) {}
extern "C" __declspec(dllexport) void ReShadeRegisterEvent(unsigned, void*) { ++events; }
extern "C" __declspec(dllexport) void ReShadeUnregisterEvent(unsigned, void*) { --events; }
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc != 2) return 2;
    // Existing window, no swapchain callback: model an addon hot-loaded after
    // the renderer started. Offscreen and non-activating; no desktop input.
    HWND window = CreateWindowExW(WS_EX_NOACTIVATE, L"STATIC", L"Phoenix input load probe",
        WS_POPUP, -30000, -30000, 64, 64, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (!window) return 6;
    ShowWindow(window, SW_SHOWNOACTIVATE);
    HMODULE module = LoadLibraryW(argv[1]);
    if (!module || addons != 1 || events != 1) return 3;
    auto name = reinterpret_cast<const char**>(GetProcAddress(module, "NAME"));
    using Init = bool (*)(HMODULE, HMODULE);
    using Uninit = void (*)(HMODULE, HMODULE);
    auto init = reinterpret_cast<Init>(GetProcAddress(module, "AddonInit"));
    auto uninit = reinterpret_cast<Uninit>(GetProcAddress(module, "AddonUninit"));
    if (!name || std::string(*name) != "Phoenix Input" || !init || !uninit ||
        !init(module, GetModuleHandleW(nullptr)) || addons != 1 || events != 1) return 4;
    std::cout << "ready" << std::endl;
    std::string command;
    std::getline(std::cin, command);
    uninit(module, GetModuleHandleW(nullptr));
    if (addons || events || !FreeLibrary(module)) return 5;
    DestroyWindow(window);
    std::cout << "{\"passed\":true,\"actual_dll_loaded\":true,"
                 "\"balanced_attach_detach\":true,\"games_launched\":false}" << std::endl;
    return 0;
}
