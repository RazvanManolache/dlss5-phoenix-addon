// CPU-only NGX stand-in for addon unload/reload lifetime regression tests.
#include <windows.h>
#if defined(_M_IX86)
#pragma comment(linker, "/EXPORT:NVSDK_NGX_D3D12_CreateFeature=_NVSDK_NGX_D3D12_CreateFeature")
#pragma comment(linker, "/EXPORT:NVSDK_NGX_D3D12_EvaluateFeature=_NVSDK_NGX_D3D12_EvaluateFeature")
#pragma comment(linker, "/EXPORT:NVSDK_NGX_D3D12_ReleaseFeature=_NVSDK_NGX_D3D12_ReleaseFeature")
#pragma comment(linker, "/EXPORT:ProbeCalls=_ProbeCalls")
#endif
static unsigned calls;
extern "C" __declspec(dllexport) int __cdecl NVSDK_NGX_D3D12_CreateFeature(void*, int, void*, void*) {
    ++calls;
    return static_cast<int>(0xBAD00005u);
}
extern "C" __declspec(dllexport) int __cdecl NVSDK_NGX_D3D12_EvaluateFeature(void*, void*, void*, void*) {
    return static_cast<int>(0xBAD00006u);
}
extern "C" __declspec(dllexport) int __cdecl NVSDK_NGX_D3D12_ReleaseFeature(void*) {
    return static_cast<int>(0xBAD00007u);
}
extern "C" __declspec(dllexport) unsigned __cdecl ProbeCalls() { return calls; }
