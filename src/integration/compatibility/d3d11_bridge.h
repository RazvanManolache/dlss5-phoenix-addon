// D3D11 presents; D3D12 computes. Shared NT textures avoid CPU pixel transfers.
#pragma once
#include "host_client.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>

namespace phoenix_compat {
class D3D11Bridge {
    Ptr<ID3D11Device5> device_;
    Ptr<ID3D11DeviceContext4> context_;
    Ptr<ID3DDeviceContextState> clean_state_;
    Ptr<ID3D12Device> sharing_device_;
    Ptr<ID3D12Resource> shared12_[FEED_SLOTS];
    Ptr<ID3D11Texture2D> shared11_[FEED_SLOTS];
    Handle handles_[FEED_SLOTS];
    Ptr<ID3D11Fence> input_fence_, output_fence_;
    Ptr<ID3D11Texture2D> current_, previous_input_, previous_output_;
    Ptr<ID3D11ShaderResourceView> current_view_, input_view_, output_view_;
    Ptr<ID3D11RenderTargetView> capture_view_;
    Ptr<ID3D11VertexShader> vertex_;
    Ptr<ID3D11PixelShader> copy_, compose_;
    Ptr<ID3D11Buffer> constants_;
    Ptr<ID3D11RasterizerState> rasterizer_;
    std::unique_ptr<HostClient> host_;
    unsigned width_ = 0, height_ = 0;
    uint64_t presented_ = 0, serial_ = 0, delivered_ = 0, submitted_frame_ = 0,
             delivered_frame_ = 0, submitted_tick_ = 0, delivered_tick_ = 0;
    bool snapshot_ = false;
    FeedBuild build_{};
    struct PresentationTarget {
        ID3D11Texture2D* identity;
        Ptr<ID3D11RenderTargetView> view;
    };
    std::vector<PresentationTarget> targets_;
    uint64_t current_copies_ = 0, input_submissions_ = 0, target_view_creations_ = 0;
    bool sharing_device_reused_ = false;

    ID3D11RenderTargetView* presentation_target(ID3D11Texture2D* back) {
        for (const auto& target : targets_)
            if (target.identity == back)
                return target.view.Get();
        D3D11_TEXTURE2D_DESC description{};
        back->GetDesc(&description);
        D3D11_RENDER_TARGET_VIEW_DESC view_description{};
        view_description.Format = description.Format;
        view_description.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        // Transport pixels are already gamma encoded. Avoid encoding them a
        // second time when an sRGB presentation surface is exposed by ReShade.
        if (view_description.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
            view_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        if (view_description.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
            view_description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        Ptr<ID3D11RenderTargetView> view;
        hr(device_->CreateRenderTargetView(back, &view_description, &view));
        // DXGI has at most sixteen swapchain buffers. Bound unexpected callers
        // too; ordinary runtime destruction releases these views before resize.
        if (targets_.size() == 16)
            targets_.clear();
        targets_.push_back({back, std::move(view)});
        ++target_view_creations_;
        return targets_.back().view.Get();
    }

    // Swap the entire immediate-context state, including slots unused by this
    // pass. Saving only our PS/RTV slots would corrupt unrelated game state.
    struct ContextScope {
        ID3D11DeviceContext4* context;
        Ptr<ID3DDeviceContextState> previous;
        ContextScope(ID3D11DeviceContext4* value, ID3DDeviceContextState* clean) : context(value) {
            context->SwapDeviceContextState(clean, &previous);
        }
        ~ContextScope() {
            context->ClearState();
            context->SwapDeviceContextState(previous.Get(), nullptr);
        }
    };
    void draw(ID3D11RenderTargetView* target, ID3D11PixelShader* shader,
              ID3D11ShaderResourceView* const* textures, unsigned count, int mode, float intensity,
              bool coherent_frames = false) {
        struct Constants {
            int mode;
            float intensity;
            float padding[2];
        } values{mode, intensity, {float(width_), coherent_frames ? 1.f : 0.f}};
        context_->UpdateSubresource(constants_.Get(), 0, nullptr, &values, 0, 0);
        ID3D11Buffer* buffer = constants_.Get();
        D3D11_VIEWPORT viewport{0, 0, float(width_), float(height_), 0, 1};
        context_->RSSetViewports(1, &viewport);
        context_->RSSetState(rasterizer_.Get());
        context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context_->VSSetShader(vertex_.Get(), nullptr, 0);
        context_->PSSetShader(shader, nullptr, 0);
        context_->PSSetConstantBuffers(0, 1, &buffer);
        context_->PSSetShaderResources(0, count, textures);
        context_->OMSetRenderTargets(1, &target, nullptr);
        context_->Draw(3, 0);
        ID3D11ShaderResourceView* empty[3]{};
        context_->PSSetShaderResources(0, 3, empty);
        context_->OMSetRenderTargets(0, nullptr, nullptr);
    }
    void make_snapshot(Ptr<ID3D11Texture2D>& texture, Ptr<ID3D11ShaderResourceView>& view,
                       DXGI_FORMAT format) {
        D3D11_TEXTURE2D_DESC description{};
        description.Width = width_;
        description.Height = height_;
        description.MipLevels = description.ArraySize = 1;
        description.Format = format;
        description.SampleDesc.Count = 1;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        hr(device_->CreateTexture2D(&description, nullptr, &texture));
        hr(device_->CreateShaderResourceView(texture.Get(), nullptr, &view));
    }

  public:
    D3D11Bridge(ID3D11Device* game, unsigned width, unsigned height, DXGI_FORMAT format,
                ID3D12Device* sharing_device = nullptr)
        : width_(width), height_(height) {
        require(width && height && width <= 8192 && height <= 8192, "Unsupported bridge extent");
        require(format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                    format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
                    format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                "D3D11 bridge currently requires an SDR RGBA8/BGRA8 back buffer");
        hr(game->QueryInterface(IID_PPV_ARGS(&device_)));
        Ptr<ID3D11DeviceContext> immediate;
        game->GetImmediateContext(&immediate);
        hr(immediate.As(&context_));
        D3D_FEATURE_LEVEL levels[] = {game->GetFeatureLevel()}, chosen{};
        // Context states must match a SINGLETHREADED game device. Passing zero
        // for such a device is invalid and unnecessarily forces CPU transport.
        const UINT state_flags = game->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED
            ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED : 0;
        hr(device_->CreateDeviceContextState(state_flags, levels, 1, D3D11_SDK_VERSION,
                                             __uuidof(ID3D11Device), &chosen, &clean_state_));
        Ptr<IDXGIDevice> dxgi;
        hr(game->QueryInterface(IID_PPV_ARGS(&dxgi)));
        Ptr<IDXGIAdapter> adapter;
        hr(dxgi->GetAdapter(&adapter));
        DXGI_ADAPTER_DESC adapter_description{};
        hr(adapter->GetDesc(&adapter_description));
        if (sharing_device) {
            const auto identity = sharing_device->GetAdapterLuid();
            require(identity.LowPart == adapter_description.AdapterLuid.LowPart &&
                        identity.HighPart == adapter_description.AdapterLuid.HighPart,
                    "Cached sharing device belongs to another game adapter");
            sharing_device_ = sharing_device;
            sharing_device_reused_ = true;
        } else
            hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&sharing_device_)));
        build_.width = build_.output_width = width;
        build_.height = build_.output_height = height;
        build_.color_fmt = build_.output_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        build_.mv_scale_x = build_.mv_scale_y = 1;
        build_.adapter_luid = (uint64_t(uint32_t(adapter_description.AdapterLuid.HighPart)) << 32) |
                              adapter_description.AdapterLuid.LowPart;
        for (unsigned slot = 0; slot < FEED_SLOTS; ++slot) {
            if (slot == FEED_MASK)
                continue;
            const auto pixel_format = slot == FEED_MV      ? DXGI_FORMAT_R16G16_FLOAT
                                      : slot == FEED_DEPTH ? DXGI_FORMAT_R32_FLOAT
                                                           : DXGI_FORMAT_R8G8B8A8_UNORM;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC description{};
            description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            description.Width = width;
            description.Height = height;
            description.DepthOrArraySize = description.MipLevels = 1;
            description.Format = pixel_format;
            description.SampleDesc.Count = 1;
            description.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            hr(sharing_device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &description,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&shared12_[slot])));
            hr(sharing_device_->CreateSharedHandle(shared12_[slot].Get(), nullptr, GENERIC_ALL,
                                                   nullptr, &handles_[slot].value));
            hr(device_->OpenSharedResource1(handles_[slot].value, IID_PPV_ARGS(&shared11_[slot])));
            build_.tex[slot] = uint64_t(uintptr_t(handles_[slot].value));
            if (slot == FEED_DEPTH || slot == FEED_MV) {
                Ptr<ID3D11RenderTargetView> view;
                hr(device_->CreateRenderTargetView(shared11_[slot].Get(), nullptr, &view));
                const float blank[4] = {slot == FEED_DEPTH ? 1.f : 0.f, 0, 0, 0};
                context_->ClearRenderTargetView(view.Get(), blank);
            }
        }
        hr(device_->CreateRenderTargetView(shared11_[FEED_COLOR].Get(), nullptr, &capture_view_));
        const auto unorm = format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB   ? DXGI_FORMAT_B8G8R8A8_UNORM
                           : format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? DXGI_FORMAT_R8G8B8A8_UNORM
                                                                       : format;
        make_snapshot(current_, current_view_, unorm);
        make_snapshot(previous_input_, input_view_, DXGI_FORMAT_R8G8B8A8_UNORM);
        make_snapshot(previous_output_, output_view_, DXGI_FORMAT_R8G8B8A8_UNORM);
        const char* shader = R"(
            Texture2D<float4> currentFrame:register(t0),oldInput:register(t1),oldOutput:register(t2);
            cbuffer Controls:register(b0){int mode;float intensity;float frameWidth;float coherent;};
            float4 vertex(uint index:SV_VertexID):SV_Position {
                return float4(index==2?3:-1,index==1?3:-1,0,1);
            }
            float4 copy(float4 position:SV_Position):SV_Target {return currentFrame.Load(int3(position.xy,0));}
            float4 compose(float4 position:SV_Position):SV_Target {
                int3 pixel=int3(position.xy,0);
                float4 input=oldInput.Load(pixel), output=oldOutput.Load(pixel);
                // Masks and pixels must belong to the same completed frame.
                // Live-frame editing is optional and can expose newer motion.
                float4 base=coherent!=0?input:currentFrame.Load(pixel);
                float4 edited=coherent!=0?output:float4(saturate(base.rgb+intensity*(output.rgb-input.rgb)),base.a);
                if(mode==2 && position.x<frameWidth*.5)return base;
                return edited;
            })";
        std::string source = shader;
        auto compile = [&](const char* entry, const char* profile) {
            Ptr<ID3DBlob> blob, errors;
            const auto result =
                D3DCompile(source.data(), source.size(), "phoenix_d3d11_bridge", nullptr, nullptr,
                           entry, profile, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &errors);
            if (FAILED(result))
                throw std::runtime_error(
                    errors ? std::string((char*)errors->GetBufferPointer(), errors->GetBufferSize())
                           : "Bridge shader failed");
            return blob;
        };
        auto bytecode = compile("vertex", "vs_5_0");
        hr(device_->CreateVertexShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                                       nullptr, &vertex_));
        bytecode = compile("copy", "ps_5_0");
        hr(device_->CreatePixelShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                                      nullptr, &copy_));
        bytecode = compile("compose", "ps_5_0");
        hr(device_->CreatePixelShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                                      nullptr, &compose_));
        D3D11_BUFFER_DESC buffer{};
        buffer.ByteWidth = 16;
        buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        hr(device_->CreateBuffer(&buffer, nullptr, &constants_));
        D3D11_RASTERIZER_DESC raster{};
        raster.FillMode = D3D11_FILL_SOLID;
        raster.CullMode = D3D11_CULL_NONE;
        raster.DepthClipEnable = TRUE;
        hr(device_->CreateRasterizerState(&raster, &rasterizer_));
    }
    ~D3D11Bridge() {
        host_.reset();
    }
    void start(HMODULE addon, const phoenix_model::WeightSource& source,
               const phoenix_addon::phoenix::Settings& settings, bool transport_test = false) {
        build_.transport = transport_test;
        host_ = std::make_unique<HostClient>(addon, source, settings, build_);
    }
    HostClient* host() const {
        return host_.get();
    }
    // Borrowed until this bridge is destroyed. The controller retains only the
    // device, never this bridge's textures, views, fences or model state.
    ID3D12Device* sharing_device() const {
        return sharing_device_.Get();
    }
    uint64_t delivered() const {
        return delivered_;
    }
    uint64_t presented() const {
        return presented_;
    }
    Json transfer_timings() const {
        return {{"current_snapshot_copies", current_copies_},
                {"input_submissions", input_submissions_},
                {"target_view_creations", target_view_creations_},
                {"sharing_device_reused", sharing_device_reused_}};
    }
    void frame(ID3D11Texture2D* back, uint64_t frame_number,
               const phoenix_addon::phoenix::Settings& settings, bool coherent_frames = false) {
        frame(back, presentation_target(back), frame_number, settings, coherent_frames);
    }
    void frame(ID3D11Texture2D* back, ID3D11RenderTargetView* target, uint64_t frame_number,
               const phoenix_addon::phoenix::Settings& settings, bool coherent_frames = false) {
        if (!host_)
            return;
        if (host_->phase() == Phase::OpenFences) {
            auto shared = host_->fences();
            Handle input, output;
            input.value = HANDLE(uintptr_t(shared.fence_in));
            output.value = HANDLE(uintptr_t(shared.fence_out));
            hr(device_->OpenSharedFence(input.value, IID_PPV_ARGS(&input_fence_)));
            hr(device_->OpenSharedFence(output.value, IID_PPV_ARGS(&output_fence_)));
            host_->opened_fences(output_fence_.Get());
        }
        ContextScope state(context_.Get(), clean_state_.Get());
        if (host_->phase() == Phase::Finished) {
            context_->CopyResource(previous_input_.Get(), shared11_[FEED_COLOR].Get());
            context_->CopyResource(previous_output_.Get(), shared11_[FEED_OUTPUT].Get());
            host_->take();
            snapshot_ = true;
            ++delivered_;
            delivered_frame_ = submitted_frame_;
            delivered_tick_ = submitted_tick_;
        }
        const bool submit_input =
            host_->phase() == Phase::Ready && settings.mode != phoenix_addon::phoenix::input;
        const bool show_live_edit =
            snapshot_ && !coherent_frames && settings.mode != phoenix_addon::phoenix::input &&
            frame_number - delivered_frame_ <= uint64_t(settings.offload_age_frames) &&
            GetTickCount64() - delivered_tick_ <= uint64_t(settings.offload_age_ms);
        // A held matched frame never reads current game pixels. Only capture a
        // snapshot when submitting a new input or applying the optional live edit.
        if (submit_input || show_live_edit) {
            context_->CopyResource(current_.Get(), back);
            ++current_copies_;
        }
        if (submit_input) {
            ID3D11ShaderResourceView* views[] = {current_view_.Get()};
            draw(capture_view_.Get(), copy_.Get(), views, 1, 0, 1);
            hr(context_->Signal(input_fence_.Get(), ++serial_));
            context_->Flush();
            require(host_->submit(serial_, true), "Bridge submission lost ownership");
            ++input_submissions_;
            submitted_frame_ = frame_number;
            submitted_tick_ = GetTickCount64();
        }
        if (snapshot_ && settings.mode != phoenix_addon::phoenix::input &&
            (coherent_frames ||
             (frame_number - delivered_frame_ <= uint64_t(settings.offload_age_frames) &&
              GetTickCount64() - delivered_tick_ <= uint64_t(settings.offload_age_ms)))) {
            ID3D11ShaderResourceView* views[] = {current_view_.Get(), input_view_.Get(),
                                                 output_view_.Get()};
            draw(target, compose_.Get(), views, 3, helper_presentation_mode(settings), settings.intensity,
                 coherent_frames);
            ++presented_;
        }
    }
};
} // namespace phoenix_compat
