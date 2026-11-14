#pragma once
#include <d3d12.h>

// Private, versioned ABI. Each DLL owns and frees its own context.
namespace phoenix_motion_plugin {
struct FidelityFX {
    static constexpr wchar_t filename[] = L"phoenix_motion_fidelityfx.dll";
    using Create = void* (*)(ID3D12Device*, unsigned, unsigned);
    using Destroy = void (*)(void*);
    using Run = bool (*)(void*, ID3D12GraphicsCommandList*, ID3D12Resource*,
                        ID3D12Resource*, ID3D12Resource*, bool);
};
struct DIS {
    static constexpr wchar_t filename[] = L"phoenix_motion_dis.dll";
    using Create = void* (*)();
    using Destroy = void (*)(void*);
    using Run = bool (*)(void*, unsigned, unsigned, const unsigned char*, unsigned,
                        const unsigned char*, float*, unsigned);
};
}
