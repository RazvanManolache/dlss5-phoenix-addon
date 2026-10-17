// An isolated x64 DLL: no ReShade, model allocations, hooks, or GPU dispatch.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include "../../third_party/nvapi/nvapi.h"
#include "wow64_probe_contract.h"
#ifdef PHX_NATIVE_DRIVER_BRIDGE
#include <cstdio>
#endif
#ifdef PHX_WOW64_UMD_PROBE
#include "wow64_umd_probe.h"
static const wchar_t* native_cuda_library=nullptr;
extern "C" __declspec(dllexport) void PhoenixSetCudaLibrary64(const wchar_t* path) { native_cuda_library=path; }
extern "C" __declspec(dllexport) unsigned PhoenixUmdRedirectCount64() { return wow64_umd_probe::redirects; }
extern "C" __declspec(dllexport) void PhoenixSetGdiRpc64(PhoenixGdiRpc* rpc) { wow64_umd_probe::gdi = rpc; }
#endif

extern "C" void* __cdecl memset(void* memory, int value, size_t bytes);
#pragma function(memset)
extern "C" void* __cdecl memset(void* memory, int value, size_t bytes) {
    auto* output = static_cast<volatile unsigned char*>(memory);
    while (bytes--) *output++ = static_cast<unsigned char>(value);
    return memory;
}

extern "C" __declspec(dllexport) int __cdecl PhoenixWow64Probe64(
    const wchar_t* path, PhoenixWow64ProbeResult* result) {
    if (!result || !path) return 1;
#ifdef PHX_WOW64_UMD_PROBE
#ifdef PHX_NATIVE_DRIVER_BRIDGE
    wow64_runtime_diagnostics::retry_native_reservation=true;
    wow64_umd_probe::bridge_kmt_escape=true;
    wow64_umd_probe::bridge_dc_creation=true;
    wow64_umd_probe::bridge_display_gdi=true;
    if(!wow64_umd_probe::gdi || !wow64_umd_probe::install_gdi_export() ||
        !wow64_umd_probe::install_display_exports()) return 3;
#endif
    if (!wow64_umd_probe::install()) return 3;
#ifdef PHX_NATIVE_DRIVER_BRIDGE
    if(native_cuda_library) {
        auto cuda=LoadLibraryW(native_cuda_library);
        if(!cuda) return 3;
        wow64_umd_probe::patch(cuda);
        auto initialize=reinterpret_cast<int(*)(unsigned)>(GetProcAddress(cuda,"cuInit"));
        if(!initialize) return 3;
        const auto status=initialize(0);
        std::printf("{\"event\":\"native_driver_cuda_init\",\"status\":%d,\"pid\":%lu}\n",status,GetCurrentProcessId());
        std::fflush(stdout);
        if(status!=0) return 3;
    }
#endif
#endif
    *result = {};
    result->stage = 1;
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE;
    void* blob = nullptr;
    HMODULE nvapi = nullptr;
    NVDX_ObjectHandle module = nullptr;
    DWORD bytes = 0, read = 0;
    DXGI_ADAPTER_DESC1 desc{};
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    decltype(&NvAPI_D3D12_DestroyCuModule) destroy = nullptr;
    decltype(&NvAPI_D3D12_CreateCuModule) create = nullptr;
#ifdef PHX_NATIVE_DRIVER_ARITHMETIC
    decltype(&NvAPI_D3D12_CreateCubinComputeShaderWithName) legacy_create = nullptr;
#endif
    decltype(&NvAPI_D3D12_IsFatbinPTXSupported) supported = nullptr;
    void*(__cdecl* query)(unsigned) = nullptr;
    NvAPI_Status(__cdecl* initialize)() = nullptr;
    bool available = false;
    result->stage = 2;
    result->status = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(result->status)) goto cleanup;
    for (UINT index = 0;; ++index) {
        result->stage = 3;
        result->status = factory->EnumAdapters1(index, &adapter);
        if (FAILED(result->status)) goto cleanup;
        result->status = adapter->GetDesc1(&desc);
        if (FAILED(result->status)) goto cleanup;
        if (desc.VendorId == 0x10de && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
        adapter->Release(); adapter = nullptr;
    }
    result->stage = 4;
    result->status = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(result->status)) goto cleanup;
    result->stage = 5;
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    result->status = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (FAILED(result->status)) goto cleanup;
    result->stage = 6;
    nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!nvapi) { result->win32_error = GetLastError(); result->status = -1; goto cleanup; }
#ifdef PHX_WOW64_UMD_PROBE
    wow64_umd_probe::patch(nvapi);
#endif
    query = reinterpret_cast<decltype(query)>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
    if (!query) { result->status = -1; goto cleanup; }
    initialize = reinterpret_cast<decltype(initialize)>(query(0x0150e828));
    create = reinterpret_cast<decltype(create)>(query(0xad1a677d));
    destroy = reinterpret_cast<decltype(destroy)>(query(0x41c65285));
    supported = reinterpret_cast<decltype(supported)>(query(0x70c07832));
#ifdef PHX_NATIVE_DRIVER_ARITHMETIC
    legacy_create=reinterpret_cast<decltype(legacy_create)>(query(0x1dc7261f));
    destroy=reinterpret_cast<decltype(destroy)>(query(0x7fb785ba));
    if(!legacy_create) { result->status=-1; goto cleanup; }
#endif
    if (!initialize || !create || !destroy || !supported) { result->status = -1; goto cleanup; }
    result->stage = 7;
#ifdef PHX_WOW64_UMD_PROBE
    wow64_umd_probe::patch(GetModuleHandleW(L"nvapi64_impl.dll"));
#endif
    result->status = initialize();
    if (result->status != NVAPI_OK) goto cleanup;
    result->stage = 8;
    result->status = supported(device, &available);
    result->fatbin_supported = available;
    if (result->status != NVAPI_OK || !available) { result->status = -1; goto cleanup; }
    result->stage = 9;
    file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { result->win32_error = GetLastError(); result->status = -1; goto cleanup; }
    bytes = GetFileSize(file, nullptr);
    if (!bytes || bytes > 16 * 1024 * 1024) { result->status = -1; goto cleanup; }
    blob = HeapAlloc(GetProcessHeap(), 0, bytes);
    if (!blob || !ReadFile(file, blob, bytes, &read, nullptr) || read != bytes) {
        result->win32_error = GetLastError(); result->status = -1; goto cleanup;
    }
    result->blob_bytes = bytes;
    result->stage = 10;
#ifdef PHX_NATIVE_DRIVER_ARITHMETIC
    result->status = legacy_create(device,blob,bytes,64,1,1,"phoenix_arithmetic",&module);
#else
    result->status = create(device, blob, bytes, &module);
#endif
    result->module_created = module != nullptr;
    if (result->status != NVAPI_OK || !module) goto cleanup;
    result->stage = 11;
    result->status = destroy(device, module);
    module = nullptr;
    if (result->status != NVAPI_OK) goto cleanup;
    result->stage = 12;
cleanup:
#ifdef PHX_NATIVE_DRIVER_BRIDGE
    std::printf("{\"event\":\"native_driver_bridge\",\"pid\":%lu,\"driver_pointer_bits\":64,"
        "\"phoenix_runtime_loaded\":%s,\"bridged_escape_count\":%ld,\"module_created\":%u,\"status\":%ld}\n",
        GetCurrentProcessId(),GetModuleHandleW(L"phoenix_native.dll")?"true":"false",
        LONG(wow64_umd_probe::bridged_escape_count),result->module_created,LONG(result->status));
    std::fflush(stdout);
#endif
#ifdef PHX_WOW64_UMD_PROBE
    wow64_umd_probe::restore();
#ifdef PHX_NATIVE_DRIVER_BRIDGE
    wow64_umd_probe::restore_gdi_export();
    wow64_umd_probe::restore_display_exports();
#endif
#endif
    if (module && destroy) destroy(device, module);
    if (blob) HeapFree(GetProcessHeap(), 0, blob);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    if (queue) queue->Release();
    if (device) device->Release();
    if (adapter) adapter->Release();
    if (factory) factory->Release();
    // Retain the NVAPI module until process exit so driver-owned teardown is safe.
    return result->stage == 12 && result->status == NVAPI_OK ? 0 : 2;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
