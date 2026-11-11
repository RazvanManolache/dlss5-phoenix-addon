#pragma once
#include <d3d10_1.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace phoenix_compat {
inline unsigned depth_pixel_bytes(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_D16_UNORM: return 2;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_D32_FLOAT: return 4;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return 8;
    default: return 0;
    }
}

inline DXGI_FORMAT depth_staging_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24G8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

inline float unpack_device_depth(const uint8_t* bytes, DXGI_FORMAT format) {
    if (format == DXGI_FORMAT_D16_UNORM) {
        uint16_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return float(value) / 65535.f;
    }
    if (format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
        uint32_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return float(value & 0xffffff) / 16777215.f;
    }
    float value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// For a CPU capture route, copy only a depth attachment paired with the
// exact color target. Both copies retire behind the same capture fence.
class DxDepthReadback {
    template<class T> using Ptr = Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D10Texture2D> depth10_;
    Ptr<ID3D11Texture2D> depth11_;
    Ptr<ID3D11DeviceContext> context11_;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
    unsigned width_, height_;
    bool captured_ = false, reversed_ = false;

    template<class Resource, class ColorView, class DepthView>
    bool match(ColorView* color, DepthView* depth, uint64_t back, Ptr<Resource>& source) {
        if (!color || !depth) return false;
        Ptr<Resource> target;
        color->GetResource(&target);
        if (uint64_t(uintptr_t(target.Get())) != back) return false;
        depth->GetResource(&source);
        return bool(source);
    }

    template<class Device, class Context, class Texture, class Desc, class Resource, class Usage>
    bool stage(Device* device, Context* commands, Resource* source, DXGI_FORMAT format,
               Ptr<Texture>& staging, Usage usage, unsigned cpu_access) {
        Ptr<Texture> texture;
        if (FAILED(source->QueryInterface(IID_PPV_ARGS(&texture)))) return false;
        Desc desc{};
        texture->GetDesc(&desc);
        const auto staging_format = depth_staging_format(format);
        if (desc.Width != width_ || desc.Height != height_ || desc.MipLevels != 1 ||
            desc.ArraySize != 1 || desc.SampleDesc.Count != 1 || staging_format == DXGI_FORMAT_UNKNOWN ||
            (desc.Format != format && desc.Format != staging_format)) return false;
        // Read back through the matching typeless family with no depth bind.
        desc.Format = staging_format;
        if (staging) {
            Desc previous{};
            staging->GetDesc(&previous);
            if (previous.Format != desc.Format) staging.Reset();
        }
        if (!staging) {
            desc.Usage = usage;
            desc.BindFlags = 0;
            desc.MiscFlags = 0;
            desc.CPUAccessFlags = cpu_access;
            if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) return false;
        }
        format_ = format;
        commands->CopyResource(staging.Get(), texture.Get());
        return true;
    }

    bool convention(bool enabled, unsigned compare) {
        reversed_ = compare == D3D11_COMPARISON_GREATER || compare == D3D11_COMPARISON_GREATER_EQUAL;
        return enabled && (reversed_ || compare == D3D11_COMPARISON_LESS || compare == D3D11_COMPARISON_LESS_EQUAL);
    }

    void decode(const void* data, unsigned pitch, std::vector<float>& output) const {
        const auto bytes = depth_pixel_bytes(format_);
        if (!data || !bytes || pitch < width_ * bytes) { output.clear(); return; }
        float minimum = 1.f, maximum = 0.f;
        for (unsigned y = 0; y < height_; ++y) {
            const auto* row = static_cast<const uint8_t*>(data) + size_t(y) * pitch;
            for (unsigned x = 0; x < width_; ++x) {
                const float value = unpack_device_depth(row + size_t(x) * bytes, format_);
                if (!std::isfinite(value) || value < 0.f || value > 1.f) {
                    output.clear();
                    return;
                }
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                output[size_t(y) * width_ + x] = reversed_ ? 1.f - value : value;
            }
        }
        if (maximum - minimum <= 1e-6f) output.clear();
    }

  public:
    DxDepthReadback(unsigned width, unsigned height) : width_(width), height_(height) {}

    void capture(ID3D10Device* device, uint64_t back) {
        captured_ = false;
        Ptr<ID3D10Device1> device1;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device1))) || device1->GetFeatureLevel() < D3D10_FEATURE_LEVEL_10_1)
            return;
        Ptr<ID3D10RenderTargetView> color;
        Ptr<ID3D10DepthStencilView> depth;
        device->OMGetRenderTargets(1, &color, &depth);
        Ptr<ID3D10Resource> source;
        if (!match(color.Get(), depth.Get(), back, source)) return;
        D3D10_DEPTH_STENCIL_VIEW_DESC view{};
        depth->GetDesc(&view);
        if (view.ViewDimension != D3D10_DSV_DIMENSION_TEXTURE2D || view.Texture2D.MipSlice != 0) return;
        Ptr<ID3D10DepthStencilState> state;
        UINT stencil = 0;
        device->OMGetDepthStencilState(&state, &stencil);
        D3D10_DEPTH_STENCIL_DESC settings{};
        settings.DepthEnable = TRUE;
        settings.DepthFunc = D3D10_COMPARISON_LESS;
        if (state) state->GetDesc(&settings);
        if (!convention(settings.DepthEnable != FALSE, settings.DepthFunc)) return;
        captured_ = stage<ID3D10Device, ID3D10Device, ID3D10Texture2D, D3D10_TEXTURE2D_DESC>(
            device, device, source.Get(), view.Format, depth10_, D3D10_USAGE_STAGING, D3D10_CPU_ACCESS_READ);
    }

    void capture(ID3D11Device* device, uint64_t back) {
        captured_ = false;
        if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_10_1) return;
        if (!context11_) device->GetImmediateContext(&context11_);
        if (!context11_) return;
        Ptr<ID3D11RenderTargetView> color;
        Ptr<ID3D11DepthStencilView> depth;
        context11_->OMGetRenderTargets(1, &color, &depth);
        Ptr<ID3D11Resource> source;
        if (!match(color.Get(), depth.Get(), back, source)) return;
        D3D11_DEPTH_STENCIL_VIEW_DESC view{};
        depth->GetDesc(&view);
        if (view.ViewDimension != D3D11_DSV_DIMENSION_TEXTURE2D || view.Texture2D.MipSlice != 0) return;
        Ptr<ID3D11DepthStencilState> state;
        UINT stencil = 0;
        context11_->OMGetDepthStencilState(&state, &stencil);
        D3D11_DEPTH_STENCIL_DESC settings{};
        settings.DepthEnable = TRUE;
        settings.DepthFunc = D3D11_COMPARISON_LESS;
        if (state) state->GetDesc(&settings);
        if (!convention(settings.DepthEnable != FALSE, settings.DepthFunc)) return;
        captured_ = stage<ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_TEXTURE2D_DESC>(
            device, context11_.Get(), source.Get(), view.Format, depth11_, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);
    }

    void read(std::vector<float>& output) {
        output.clear();
        if (!captured_) return;
        captured_ = false;
        output.resize(size_t(width_) * height_);
        if (depth11_) {
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if (SUCCEEDED(context11_->Map(depth11_.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped))) {
                decode(mapped.pData, mapped.RowPitch, output);
                context11_->Unmap(depth11_.Get(), 0);
                return;
            }
        } else if (depth10_) {
            D3D10_MAPPED_TEXTURE2D mapped{};
            if (SUCCEEDED(depth10_->Map(0, D3D10_MAP_READ, D3D10_MAP_FLAG_DO_NOT_WAIT, &mapped))) {
                decode(mapped.pData, mapped.RowPitch, output);
                depth10_->Unmap(0);
                return;
            }
        }
        output.clear();
    }
};
}
