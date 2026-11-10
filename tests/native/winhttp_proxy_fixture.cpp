// A game-local WinHTTP replacement used by release_update_tests. No networking.
#include <windows.h>
static LONG calls{};
extern "C" __declspec(dllexport) void* WINAPI WinHttpOpen(
    const wchar_t*, DWORD, const wchar_t*, const wchar_t*, DWORD) {
    InterlockedIncrement(&calls);
    return nullptr;
}
extern "C" __declspec(dllexport) LONG PhoenixWinHttpProxyCalls() { return calls; }
