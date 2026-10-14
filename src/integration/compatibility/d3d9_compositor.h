// Compose a completed input/output pair. The optional live-frame edit is less
// latent, but can move a newer game image underneath an older model mask.
#pragma once
#include "host_client.h"
#include <d3d9.h>
#include <d3dcompiler.h>
namespace phoenix_compat {
class D3D9Compositor {
    Ptr<IDirect3DDevice9> device_;
    Ptr<IDirect3DTexture9> current_, captured_, matched_input_;
    Ptr<IDirect3DPixelShader9> shader_;
    Ptr<IDirect3DStateBlock9> state_;
    unsigned width_, height_, target_count_;
    D3DFORMAT format_;
    Ptr<IDirect3DSurface9> surface(IDirect3DTexture9* texture) {
        Ptr<IDirect3DSurface9> result;
        hr(texture->GetSurfaceLevel(0, &result));
        return result;
    }

  public:
    D3D9Compositor(IDirect3DDevice9* device, unsigned width, unsigned height, D3DFORMAT format)
        : device_(device), width_(width), height_(height), format_(format) {
        D3DCAPS9 caps{};
        hr(device_->GetDeviceCaps(&caps));
        target_count_ = caps.NumSimultaneousRTs;
        for (auto* texture : {std::addressof(captured_), std::addressof(matched_input_)})
            hr(device_->CreateTexture(width, height, 1, D3DUSAGE_RENDERTARGET, format,
                                      D3DPOOL_DEFAULT, texture->GetAddressOf(), nullptr));
        const char* source = R"(
sampler2D current_image : register(s0);
sampler2D model_input : register(s1);
sampler2D model_output : register(s2);
float4 display : register(c0);
float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float4 input = tex2D(model_input, uv);
    float4 output = tex2D(model_output, uv);
    float4 base = display.y != 0 ? input : tex2D(current_image, uv);
    float4 result = display.y != 0 ? output : float4(saturate(base.rgb + output.rgb - input.rgb), base.a);
    // Use an integer column boundary: interpolated UV at the center of an odd
    // width can fall on either side of 0.5 after rasterizer rounding.
    if (display.x == 2 && floor(uv.x * display.z) < floor(display.z * 0.5)) result = base;
    return result;
})";
        Ptr<ID3DBlob> code, errors;
        hr(D3DCompile(source, std::strlen(source), "phoenix_d3d9_current_frame", nullptr, nullptr,
                      "main", "ps_3_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors));
        hr(device_->CreatePixelShader(static_cast<const DWORD*>(code->GetBufferPointer()),
                                      &shader_));
        hr(device_->CreateStateBlock(D3DSBT_ALL, &state_));
    }
    void capture(IDirect3DSurface9* back) {
        auto destination = surface(captured_.Get());
        hr(device_->StretchRect(back, nullptr, destination.Get(), nullptr, D3DTEXF_NONE));
    }
    void readback_completed_capture(IDirect3DSurface9* staging) {
        // The caller first retires the event query issued after capture().
        // D3D9 readback is synchronous, but now reads an already completed
        // snapshot rather than draining the freshly rendered game backbuffer.
        auto source = surface(captured_.Get());
        hr(device_->GetRenderTargetData(source.Get(), staging));
    }
    void match_completed_input() {
        auto source = surface(captured_.Get()), destination = surface(matched_input_.Get());
        hr(device_->StretchRect(source.Get(), nullptr, destination.Get(), nullptr, D3DTEXF_NONE));
    }
    void draw(IDirect3DSurface9* back, IDirect3DTexture9* processed, int mode,
              bool coherent_frames = false) {
        if (!coherent_frames) {
            if (!current_)
                hr(device_->CreateTexture(width_, height_, 1, D3DUSAGE_RENDERTARGET, format_,
                                          D3DPOOL_DEFAULT, &current_, nullptr));
            auto destination = surface(current_.Get());
            hr(device_->StretchRect(back, nullptr, destination.Get(), nullptr, D3DTEXF_NONE));
        }
        hr(state_->Capture());
        std::vector<Ptr<IDirect3DSurface9>> old_targets(target_count_);
        for (unsigned index = 0; index < target_count_; ++index)
            device_->GetRenderTarget(index, &old_targets[index]);
        Ptr<IDirect3DSurface9> old_depth;
        device_->GetDepthStencilSurface(&old_depth);
        // State blocks omit render-target/depth bindings. Restore those too,
        // including on a device-loss exception during drawing.
        struct Restore {
            IDirect3DDevice9* device;
            IDirect3DStateBlock9* state;
            std::vector<Ptr<IDirect3DSurface9>>& targets;
            IDirect3DSurface9* depth;
            bool scene = false;
            ~Restore() {
                if (scene)
                    device->EndScene();
                for (unsigned index = 0; index < targets.size(); ++index)
                    device->SetRenderTarget(index, targets[index].Get());
                device->SetDepthStencilSurface(depth);
                state->Apply();
            }
        } restore{device_.Get(), state_.Get(), old_targets, old_depth.Get()};
        hr(device_->SetRenderTarget(0, back));
        for (unsigned index = 1; index < target_count_; ++index)
            device_->SetRenderTarget(index, nullptr);
        hr(device_->SetDepthStencilSurface(nullptr));
        D3DVIEWPORT9 viewport{0, 0, width_, height_, 0, 1};
        hr(device_->SetViewport(&viewport));
        hr(device_->SetVertexShader(nullptr));
        hr(device_->SetPixelShader(shader_.Get()));
        hr(device_->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1));
        device_->SetRenderState(D3DRS_ZENABLE, FALSE);
        device_->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device_->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
        device_->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device_->SetRenderState(D3DRS_STENCILENABLE, FALSE);
        device_->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
        device_->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        device_->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
        device_->SetRenderState(D3DRS_SRGBWRITEENABLE, FALSE);
        device_->SetRenderState(D3DRS_COLORWRITEENABLE, 15);
        device_->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
        IDirect3DTexture9* images[] = {current_.Get(), matched_input_.Get(), processed};
        for (unsigned slot = 0; slot < 3; ++slot) {
            hr(device_->SetTexture(slot, images[slot]));
            device_->SetSamplerState(slot, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            device_->SetSamplerState(slot, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            device_->SetSamplerState(slot, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
            device_->SetSamplerState(slot, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
            device_->SetSamplerState(slot, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
            device_->SetSamplerState(slot, D3DSAMP_SRGBTEXTURE, FALSE);
        }
        const float display[] = {float(mode), coherent_frames ? 1.f : 0.f, float(width_), 0};
        hr(device_->SetPixelShaderConstantF(0, display, 1));
        struct Vertex {
            float x, y, z, rhw, u, v;
        };
        const Vertex vertices[] = {{-.5f, -.5f, 0, 1, 0, 0},
                                   {float(width_) - .5f, -.5f, 0, 1, 1, 0},
                                   {-.5f, float(height_) - .5f, 0, 1, 0, 1},
                                   {float(width_) - .5f, float(height_) - .5f, 0, 1, 1, 1}};
        const auto begun = device_->BeginScene();
        if (SUCCEEDED(begun))
            restore.scene = true;
        else
            hr(begun == D3DERR_INVALIDCALL ? S_OK : begun);
        hr(device_->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, vertices, sizeof(Vertex)));
    }
};
} // namespace phoenix_compat
