#pragma once
#include <windows.h>

namespace phoenix_process {
// ReShade may invoke AddonUninit from its own process-detach callback, before
// this DLL receives DLL_PROCESS_DETACH. The reserved argument is unavailable
// there. Query the loader's shutdown state without loading another module.
inline bool shutting_down() noexcept {
    using Query = BOOLEAN(NTAPI*)();
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto query = ntdll ? reinterpret_cast<Query>(
        GetProcAddress(ntdll, "RtlDllShutdownInProgress")) : nullptr;
    return query && query() != FALSE;
}
}
