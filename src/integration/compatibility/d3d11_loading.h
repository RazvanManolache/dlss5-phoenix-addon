#pragma once
#include <d3d11.h>
#include <wrl/client.h>

namespace phoenix_compat {
// Only used when processing is requested but no completed image is available.
// ClearRenderTargetView leaves shader bindings intact. Temporarily disable the
// game's predication so an occlusion result cannot suppress this loading gate.
inline bool d3d11_record_loading(ID3D11Device* device, ID3D11Texture2D* back) {
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> target;
    if (FAILED(device->CreateRenderTargetView(back, nullptr, &target)))
        return false;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(&context);
    Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
    BOOL expected = FALSE;
    context->GetPredication(&predicate, &expected);
    context->SetPredication(nullptr, FALSE);
    const float black[4]{0, 0, 0, 1};
    context->ClearRenderTargetView(target.Get(), black);
    context->SetPredication(predicate.Get(), expected);
    return true;
}
} // namespace phoenix_compat
