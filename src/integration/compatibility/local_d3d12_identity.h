#pragma once
#include "inprocess_api.h"
#include <wrl/client.h>
#include <filesystem>

namespace phoenix_compat {
// ReShade's COM proxy escape hatch (source/com_utils.hpp). QueryInterface adds
// a reference; this is an interface contract, not a proxy-layout assumption.
inline constexpr GUID reshade_unwrapped_object =
    {0x7f2c9a11,0x3b4e,0x4d6a,{0x81,0x2f,0x5e,0x9c,0xd3,0x7a,0x1b,0x42}};

inline bool system_d3d12_implementation(const void* method) {
    HMODULE module = nullptr;
    if (!method || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(method), &module)) return false;
    wchar_t path[MAX_PATH]{}, system[MAX_PATH]{};
    const auto length = GetModuleFileNameW(module,path,MAX_PATH);
    const auto system_length = GetSystemDirectoryW(system,MAX_PATH);
    if (!length || length >= MAX_PATH || !system_length || system_length >= MAX_PATH) return false;
    const std::filesystem::path file(path);
    return _wcsicmp(file.parent_path().c_str(),system)==0 &&
        (_wcsicmp(file.filename().c_str(),L"D3D12Core.dll")==0 ||
         _wcsicmp(file.filename().c_str(),L"d3d12.dll")==0);
}

inline bool is_unwrapped_system_d3d12_queue(ID3D12CommandQueue* queue) {
    if (!queue) return false;
    // COM's published ID3D12CommandQueue ABI: QueryInterface=0, GetDevice=7,
    // ExecuteCommandLists=10, GetDesc=18. Check implementation ownership only
    // at session creation. A ReShade/other proxy must never pass as native.
    auto** methods = *reinterpret_cast<void***>(queue);
    return system_d3d12_implementation(methods[0]) &&
        system_d3d12_implementation(methods[7]) &&
        system_d3d12_implementation(methods[10]) &&
        system_d3d12_implementation(methods[18]);
}

// D3D12CreateDevice may be hooked when the game uses DX11/OpenGL/etc.
// Keep ReShade's wrappers alive, but create the private evaluator's objects on
// the native device. Mixing wrapper and native vtables invalidates observer
// identity, and passing wrappers to driver APIs is not a supported contract.
inline void resolve_local_d3d12_identity(
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>& queue,
    Microsoft::WRL::ComPtr<ID3D12Device>& wrapper_device,
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>& wrapper_queue) {
    if (!in_process_api.native_queue) return; // Unwrapped software-test context.
    auto* native = in_process_api.native_queue(queue.Get());
    if (!native) throw std::runtime_error("Private DX12 queue has no observed native identity");
    Microsoft::WRL::ComPtr<ID3D12Device> owner;
    if (FAILED(native->GetDevice(IID_PPV_ARGS(&owner))))
        throw std::runtime_error("Private DX12 queue has no native device");
    wrapper_device = device;
    wrapper_queue = queue;
    queue = native;
    device = std::move(owner);
}
} // namespace phoenix_compat
