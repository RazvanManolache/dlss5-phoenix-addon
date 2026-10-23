// Repeated real companion load/unload, with no game, GPU or input injection.
#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <cwchar>
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
    if (argc < 2 || argc > 3) return 2;
    const int cycles = argc == 3 ? _wtoi(argv[2]) : 20;
    if (cycles < 1 || cycles > 1000) return 2;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        HMODULE module = LoadLibraryW(argv[1]);
        if (!module || addons != 1 || events != 1) return 3;
        using Uninit = void (*)(HMODULE, HMODULE);
        auto uninit = reinterpret_cast<Uninit>(GetProcAddress(module, "AddonUninit"));
        if (!uninit) return 4;
        // Allow the actual worker to initialize hooks outside the loader lock.
        Sleep(150);
        uninit(module, GetModuleHandleW(nullptr));
        if (addons || events || !FreeLibrary(module)) return 5;
        PROCESS_MEMORY_COUNTERS_EX memory{};
        memory.cb = sizeof(memory);
        DWORD handles = 0;
        if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)) ||
            !GetProcessHandleCount(GetCurrentProcess(), &handles)) return 6;
        std::cout << "{\"cycle\":" << cycle << ",\"private\":" << memory.PrivateUsage
                  << ",\"heaps\":" << GetProcessHeaps(0, nullptr)
                  << ",\"handles\":" << handles << "}" << std::endl;
    }
    return 0;
}
