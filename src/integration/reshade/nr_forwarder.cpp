#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <nvsdk_ngx.h>

namespace {
using Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D12Device*,
                                         NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using Create = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
                                           NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                                             const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
HMODULE runtime = nullptr;
Init initialize = nullptr;
Create create = nullptr;
Evaluate evaluate = nullptr;
Release release = nullptr;
volatile LONG last_result = 0;
// Keep a real call frame in this named module; an optimized tail jump would
// restore the caller identity that the NR runtime refuses with PlatformError.
NVSDK_NGX_Result completed(NVSDK_NGX_Result value) {
    InterlockedExchange(&last_result, LONG(value));
    return value;
}
}
extern "C" __declspec(dllexport) BOOL WINAPI PhoenixNrLoad(const wchar_t* path) {
    if (runtime) return TRUE;
    if (!path || !*path) return FALSE;
    auto module = LoadLibraryExW(path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) return FALSE;
    initialize = reinterpret_cast<Init>(GetProcAddress(module, "NVSDK_NGX_D3D12_Init_Ext"));
    create = reinterpret_cast<Create>(GetProcAddress(module, "NVSDK_NGX_D3D12_CreateFeature"));
    evaluate = reinterpret_cast<Evaluate>(GetProcAddress(module, "NVSDK_NGX_D3D12_EvaluateFeature"));
    release = reinterpret_cast<Release>(GetProcAddress(module, "NVSDK_NGX_D3D12_ReleaseFeature"));
    if (!initialize || !create || !evaluate || !release) {
        initialize = nullptr; create = nullptr; evaluate = nullptr; release = nullptr;
        FreeLibrary(module);
        SetLastError(ERROR_PROC_NOT_FOUND);
        return FALSE;
    }
    runtime = module; // Its features own driver objects until process teardown.
    return TRUE;
}
extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV PhoenixNrInit(
    unsigned long long app, const wchar_t* data, ID3D12Device* device, NVSDK_NGX_Version version,
    const NVSDK_NGX_Parameter* parameters) {
    return completed(initialize ? initialize(app, data, device, version, parameters) : NVSDK_NGX_Result_FAIL_NotInitialized);
}
extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV PhoenixNrCreate(
    ID3D12GraphicsCommandList* list, NVSDK_NGX_Feature feature, NVSDK_NGX_Parameter* parameters,
    NVSDK_NGX_Handle** handle) {
    return completed(create ? create(list, feature, parameters, handle) : NVSDK_NGX_Result_FAIL_NotInitialized);
}
extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV PhoenixNrEvaluate(
    ID3D12GraphicsCommandList* list, const NVSDK_NGX_Handle* feature,
    const NVSDK_NGX_Parameter* parameters, PFN_NVSDK_NGX_ProgressCallback progress) {
    return completed(evaluate ? evaluate(list, feature, parameters, progress) : NVSDK_NGX_Result_FAIL_NotInitialized);
}
extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV PhoenixNrRelease(NVSDK_NGX_Handle* handle) {
    return completed(release ? release(handle) : NVSDK_NGX_Result_FAIL_NotInitialized);
}
