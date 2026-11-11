// Compose a completed input/output pair. The optional live-frame edit is less
// latent, but can move a newer game image underneath an older model mask.
#pragma once
#include "../../native/support/shader_compiler.h"
#include "host_client.h"
#include <d3d9.h>
#include <d3dcompiler.h>
#include <nvapi.h>
namespace phoenix_compat {
// D3D9 cannot sample a multisampled depth-stencil surface directly. NVIDIA's
// extended copy can resolve it into an R32F texture without changing game AA.
class D3D9NvapiDepthResolve {
    using Lifecycle = NvAPI_Status(__cdecl*)();
    using Resource = NvAPI_Status(__cdecl*)(IDirect3DResource9*);
    using Copy = NvAPI_Status(__cdecl*)(IDirect3DDevice9*, IDirect3DResource9*, const RECT*,
                                         IDirect3DResource9*, const RECT*, D3DTEXTUREFILTERTYPE);
    HMODULE module_ = nullptr;
    Lifecycle unload_ = nullptr;
    Resource register_ = nullptr, unregister_ = nullptr;
    Copy copy_ = nullptr;

  public:
    explicit D3D9NvapiDepthResolve(IDirect3DDevice9* device) {
        D3DDEVICE_CREATION_PARAMETERS parameters{};
        D3DADAPTER_IDENTIFIER9 adapter{};
        Ptr<IDirect3D9> d3d;
        if (FAILED(device->GetCreationParameters(&parameters)) || FAILED(device->GetDirect3D(&d3d)) ||
            FAILED(d3d->GetAdapterIdentifier(parameters.AdapterOrdinal, 0, &adapter)) ||
            adapter.VendorId != 0x10de) return;
        const auto library = sizeof(void*) == 8 ? L"nvapi64.dll" : L"nvapi.dll";
        module_ = LoadLibraryExW(library, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module_) return;
        const auto query = reinterpret_cast<void*(__cdecl*)(unsigned)>(
            GetProcAddress(module_, "nvapi_QueryInterface"));
        if (!query) return;
        const auto initialize = reinterpret_cast<Lifecycle>(query(0x0150e828));
        const auto unload = reinterpret_cast<Lifecycle>(query(0xd22bdd7e));
        register_ = reinterpret_cast<Resource>(query(0xa064bdfc)); // NvAPI_D3D9_RegisterResource
        unregister_ = reinterpret_cast<Resource>(query(0xbb2b17aa)); // NvAPI_D3D9_UnregisterResource
        const auto copy = reinterpret_cast<Copy>(query(0x22de03aa)); // NvAPI_D3D9_StretchRectEx
        if (initialize && unload && register_ && unregister_ && copy && initialize() == NVAPI_OK) {
            unload_ = unload;
            copy_ = copy;
        }
    }
    D3D9NvapiDepthResolve(const D3D9NvapiDepthResolve&) = delete;
    ~D3D9NvapiDepthResolve() {
        if (unload_) unload_();
        if (module_) FreeLibrary(module_);
    }
    explicit operator bool() const { return copy_ != nullptr; }
    NvAPI_Status resolve(IDirect3DDevice9* device, IDirect3DSurface9* source, IDirect3DTexture9* destination) const {
        const auto registered = register_(source);
        if (registered != NVAPI_OK) return registered;
        auto result = register_(destination);
        if (result == NVAPI_OK) {
            result = copy_(device, source, nullptr, destination, nullptr, D3DTEXF_NONE);
            unregister_(destination);
        }
        unregister_(source);
        return result;
    }
};

class D3D9Compositor {
    struct DepthImage {
        Ptr<IDirect3DTexture9> texture;
        bool reversed = false;
    };
    Ptr<IDirect3DDevice9> device_;
    std::optional<D3D9NvapiDepthResolve> nvapi_depth_;
    Ptr<IDirect3DTexture9> current_, captured_, matched_input_;
    std::array<Ptr<IDirect3DTexture9>,2> capture_slots_;
    std::array<DepthImage, 2> depth_targets_;
    std::array<bool, 2> depth_captured_{};
    Ptr<IDirect3DSurface9> depth_readback_;
    Ptr<IDirect3DSurface9> depth_source_;
    DepthImage depth_snapshot_;
    IDirect3DSurface9* depth_back_ = nullptr;
    IDirect3DSurface9* depth_color_ = nullptr;
    uint64_t depth_vertices_ = 0, best_depth_vertices_ = 0;
    bool depth_backbuffer_pass_ = false, best_depth_backbuffer_ = false;
    bool depth_state_known_ = false, depth_writing_ = false, depth_reversed_ = false;
    unsigned depth_width_ = 0, depth_height_ = 0;
    std::array<bool, 2> depth_tracked_{};
    bool retry_present_depth_ = false;
    uint64_t depth_draws_ = 0, tracked_depth_copies_ = 0, present_depth_copies_ = 0;
    uint64_t depth_copy_attempts_ = 0, depth_copy_failures_ = 0, resolved_depth_copies_ = 0;
    uint64_t depth_bindings_ = 0, unreadable_depth_bindings_ = 0, depth_viewport_mismatches_ = 0;
    D3DSURFACE_DESC last_unreadable_depth_{};
    D3DVIEWPORT9 last_mismatched_viewport_{};
    std::string last_depth_copy_error_;
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
        // Optional extra snapshot overlaps capture with inference. Allocation
        // failure retains the original single-snapshot path.
        device_->CreateTexture(width,height,1,D3DUSAGE_RENDERTARGET,format,
                               D3DPOOL_DEFAULT,&capture_slots_[0],nullptr);
        if(capture_slots_[0])
            device_->CreateTexture(width,height,1,D3DUSAGE_RENDERTARGET,format,
                                   D3DPOOL_DEFAULT,&capture_slots_[1],nullptr);
        const char* source = R"(
sampler2D current_image : register(s0);
sampler2D model_input : register(s1);
sampler2D model_output : register(s2);
float4 display : register(c0);
float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    if (display.w == 2) return float4(0, 0, 0, 1);
    if (display.w != 0) {
        float depth = tex2D(current_image, uv).r;
        return display.w < 0 ? 1 - depth : depth;
    }
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
        hr(phoenix_shader::compile(source, std::strlen(source), "phoenix_d3d9_current_frame", nullptr, nullptr,
                      "main", "ps_3_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors));
        hr(device_->CreatePixelShader(static_cast<const DWORD*>(code->GetBufferPointer()),
                                      &shader_));
        hr(device_->CreateStateBlock(D3DSBT_ALL, &state_));
    }
    void capture(IDirect3DSurface9* back,unsigned slot=0) {
        auto destination = surface(capture_slots_.at(slot)?capture_slots_[slot].Get():captured_.Get());
        hr(device_->StretchRect(back, nullptr, destination.Get(), nullptr, D3DTEXF_NONE));
    }
    void readback_completed_capture(IDirect3DSurface9* staging,unsigned slot=0) {
        // The caller first retires the event query issued after capture().
        // D3D9 readback is synchronous, but now reads an already completed
        // snapshot rather than draining the freshly rendered game backbuffer.
        require(staging && staging->GetType() == D3DRTYPE_SURFACE,
                "D3D9 readback requires a surface, not a texture interface");
        D3DSURFACE_DESC description{};
        hr(staging->GetDesc(&description));
        require(description.Pool == D3DPOOL_SYSTEMMEM && description.Width == width_ &&
                    description.Height == height_ && description.MultiSampleType == D3DMULTISAMPLE_NONE,
                "D3D9 readback requires matching single-sample system-memory storage");
        auto source = surface(capture_slots_.at(slot)?capture_slots_[slot].Get():captured_.Get());
        hr(device_->GetRenderTargetData(source.Get(), staging));
    }
    bool prefetch_available() const {return bool(capture_slots_[0]);}
    unsigned capture_capacity()const{return capture_slots_[1]?2:1;}
    void promote_submitted_capture(unsigned slot=0) {
        if(capture_slots_.at(slot))captured_.Swap(capture_slots_[slot]);
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
        IDirect3DTexture9* images[] = {current_.Get(), matched_input_.Get(), processed};
        const float display[] = {float(mode), coherent_frames ? 1.f : 0.f, float(width_), 0};
        draw_quad(back, images, display);
    }
    void draw_depth_unavailable(IDirect3DSurface9* back) {
        IDirect3DTexture9* images[] = {nullptr, nullptr, nullptr};
        const float display[] = {0, 0, 0, 2};
        draw_quad(back, images, display);
    }

    // An attachment still bound at Present can contain a previous camera
    // view. Only consider readable or resolvable surfaces written by this frame's
    // geometry. Keep a copy before a clear or attachment change.
    void depth_bind(IDirect3DSurface9* color, IDirect3DSurface9* depth) {
        if (!depth_back_ || (depth_source_.Get() == depth && depth_color_ == color)) return;
        if (depth_source_.Get() != depth) {
            snapshot_depth_pass();
            depth_source_.Reset();
            depth_vertices_ = 0;
            depth_backbuffer_pass_ = false;
        }
        depth_color_ = color;
        depth_state_known_ = false;
        if (!depth || !color || depth_source_) return;
        ++depth_bindings_;
        D3DSURFACE_DESC desc{};
        if ((!readable_depth(depth, desc) && !resolvable_depth(desc)) ||
            uint64_t(desc.Width) * height_ != uint64_t(desc.Height) * width_) {
            ++unreadable_depth_bindings_;
            last_unreadable_depth_ = desc;
            return;
        }
        depth_width_ = desc.Width;
        depth_height_ = desc.Height;
        depth_source_ = depth;
    }
    void depth_state_changed() { depth_state_known_ = false; }
    void depth_draw(uint32_t vertices) {
        if (!depth_source_ || !depth_color_ || vertices < 3) return;
        if (!depth_state_known_) {
            DWORD compare = 0, enabled = 0, writes = 0;
            D3DVIEWPORT9 viewport{};
            depth_state_known_ = true;
            depth_writing_ = SUCCEEDED(device_->GetRenderState(D3DRS_ZFUNC, &compare)) &&
                SUCCEEDED(device_->GetRenderState(D3DRS_ZENABLE, &enabled)) && enabled &&
                SUCCEEDED(device_->GetRenderState(D3DRS_ZWRITEENABLE, &writes)) && writes;
            if (depth_writing_ && (!SUCCEEDED(device_->GetViewport(&viewport)) || viewport.X != 0 ||
                viewport.Y != 0 || viewport.Width != depth_width_ || viewport.Height != depth_height_)) {
                ++depth_viewport_mismatches_;
                last_mismatched_viewport_ = viewport;
                depth_writing_ = false;
            }
            const bool reversed = compare == D3DCMP_GREATER || compare == D3DCMP_GREATEREQUAL;
            depth_writing_ &= reversed || compare == D3DCMP_LESS || compare == D3DCMP_LESSEQUAL;
            if (depth_writing_ && depth_vertices_ && depth_reversed_ != reversed) {
                snapshot_depth_pass();
                depth_vertices_ = 0;
                depth_backbuffer_pass_ = false;
            }
            if (depth_writing_) depth_reversed_ = reversed;
        }
        if (depth_writing_) {
            ++depth_draws_;
            depth_vertices_ += vertices;
            depth_backbuffer_pass_ |= depth_color_ == depth_back_;
        }
    }
    void depth_clear(IDirect3DSurface9* depth) {
        if (!depth || depth_source_.Get() != depth) return;
        snapshot_depth_pass();
        depth_vertices_ = 0;
        depth_backbuffer_pass_ = false;
    }
    void next_depth_frame(IDirect3DSurface9* back) {
        depth_source_.Reset();
        depth_color_ = nullptr;
        depth_vertices_ = best_depth_vertices_ = 0;
        depth_backbuffer_pass_ = best_depth_backbuffer_ = depth_state_known_ = false;
        depth_back_ = back;
        if (!back) return;
        Ptr<IDirect3DSurface9> color, depth;
        if (SUCCEEDED(device_->GetRenderTarget(0, &color)) && SUCCEEDED(device_->GetDepthStencilSurface(&depth)))
            depth_bind(color.Get(), depth.Get());
    }
    bool tracking_depth() const { return depth_back_ != nullptr; }
    void capture_depth(IDirect3DSurface9* back, unsigned slot) {
        depth_captured_.at(slot) = false;
        depth_tracked_.at(slot) = false;
        if (!depth_back_) return;
        snapshot_depth_pass();
        const bool retry_present = std::exchange(retry_present_depth_, false);
        if ((!best_depth_vertices_ || retry_present) && capture_present_depth(back, depth_targets_.at(slot))) {
            ++present_depth_copies_;
            depth_captured_.at(slot) = true;
            return;
        }
        if (best_depth_vertices_) {
            std::swap(depth_snapshot_, depth_targets_.at(slot));
            depth_captured_.at(slot) = true;
            depth_tracked_.at(slot) = true;
            ++tracked_depth_copies_;
        }
    }
    Json depth_capture_status() const {
        return {{"eligible_depth_draws", depth_draws_}, {"tracked_copies", tracked_depth_copies_},
                {"present_copies", present_depth_copies_}, {"tracked_copy_attempts", depth_copy_attempts_},
                {"tracked_copy_failures", depth_copy_failures_}, {"last_copy_error", last_depth_copy_error_},
                {"msaa_resolves", resolved_depth_copies_},
                {"bindings", depth_bindings_}, {"ineligible_bindings", unreadable_depth_bindings_},
                {"last_ineligible_attachment", {last_unreadable_depth_.Width, last_unreadable_depth_.Height,
                    unsigned(last_unreadable_depth_.Format), unsigned(last_unreadable_depth_.MultiSampleType)}},
                {"viewport_mismatches", depth_viewport_mismatches_},
                {"last_mismatched_viewport", {last_mismatched_viewport_.X, last_mismatched_viewport_.Y,
                    last_mismatched_viewport_.Width, last_mismatched_viewport_.Height}}};
    }

  private:
    static bool readable_depth(IDirect3DSurface9* depth, D3DSURFACE_DESC& desc) {
        constexpr auto intz = D3DFORMAT(MAKEFOURCC('I', 'N', 'T', 'Z'));
        return depth && SUCCEEDED(depth->GetDesc(&desc)) && desc.Format == intz &&
            desc.MultiSampleType == D3DMULTISAMPLE_NONE && desc.Width && desc.Height;
    }
    bool resolvable_depth(const D3DSURFACE_DESC& desc) {
        if ((desc.Usage & D3DUSAGE_DEPTHSTENCIL) == 0 || desc.MultiSampleType == D3DMULTISAMPLE_NONE ||
            desc.Width != width_ || desc.Height != height_ ||
            (desc.Format != D3DFMT_D24S8 && desc.Format != D3DFMT_D24X8 && desc.Format != D3DFMT_D16)) return false;
        if (!nvapi_depth_) nvapi_depth_.emplace(device_.Get());
        return bool(*nvapi_depth_);
    }
    bool copy_depth(IDirect3DSurface9* source, bool reversed, DepthImage& target) {
        try {
            if (!target.texture) hr(device_->CreateTexture(width_, height_, 1, D3DUSAGE_RENDERTARGET,
                D3DFMT_R32F, D3DPOOL_DEFAULT, &target.texture, nullptr));
            D3DSURFACE_DESC desc{};
            hr(source->GetDesc(&desc));
            if (desc.MultiSampleType != D3DMULTISAMPLE_NONE) {
                const auto result = nvapi_depth_->resolve(device_.Get(), source, target.texture.Get());
                if (result != NVAPI_OK) {
                    last_depth_copy_error_ = "D3D9 depth resolve failed with NVAPI status " + std::to_string(int(result));
                    return false;
                }
                target.reversed = reversed;
                ++resolved_depth_copies_;
                return true;
            }
            Ptr<IDirect3DTexture9> texture;
            hr(source->GetContainer(IID_PPV_ARGS(&texture)));
            require(bool(texture), "D3D9 depth surface has no texture container");
            IDirect3DTexture9* images[]{texture.Get(), nullptr, nullptr};
            const float display[]{0, 0, float(width_), reversed ? -1.f : 1.f};
            draw_quad(surface(target.texture.Get()).Get(), images, display);
            target.reversed = false;
            return true;
        } catch (const std::exception& error) {
            last_depth_copy_error_ = error.what();
            return false;
        }
    }
    void snapshot_depth_pass() {
        if (!depth_source_ || depth_vertices_ < 3 || depth_vertices_ < best_depth_vertices_ ||
            (depth_vertices_ == best_depth_vertices_ && (best_depth_backbuffer_ || !depth_backbuffer_pass_)))
            return;
        ++depth_copy_attempts_;
        if (!copy_depth(depth_source_.Get(), depth_reversed_, depth_snapshot_)) {
            ++depth_copy_failures_;
            return;
        }
        best_depth_vertices_ = depth_vertices_;
        best_depth_backbuffer_ = depth_backbuffer_pass_;
    }
    bool capture_present_depth(IDirect3DSurface9* back, DepthImage& target) {
        Ptr<IDirect3DSurface9> color, depth;
        D3DSURFACE_DESC desc{};
        if (FAILED(device_->GetRenderTarget(0, &color)) || color.Get() != back ||
            FAILED(device_->GetDepthStencilSurface(&depth)) || !readable_depth(depth.Get(), desc) ||
            desc.Width != width_ || desc.Height != height_) return false;
        DWORD compare = 0;
        if (FAILED(device_->GetRenderState(D3DRS_ZFUNC, &compare))) return false;
        const bool reversed = compare == D3DCMP_GREATER || compare == D3DCMP_GREATEREQUAL;
        if (!reversed && compare != D3DCMP_LESS && compare != D3DCMP_LESSEQUAL) return false;
        return copy_depth(depth.Get(), reversed, target);
    }

  public:
    void readback_depth(unsigned slot, std::vector<float>& pixels) {
        pixels.clear();
        if (!depth_captured_.at(slot)) return;
        depth_captured_[slot] = false;
        // A copied pass may still be flat or unreadable. Try the exact
        // presentation attachment on the next eligible capture.
        retry_present_depth_ |= depth_tracked_.at(slot);
        if (!depth_readback_ && FAILED(device_->CreateOffscreenPlainSurface(width_, height_,
                D3DFMT_R32F, D3DPOOL_SYSTEMMEM, &depth_readback_, nullptr))) return;
        const auto& source = depth_targets_.at(slot);
        Ptr<IDirect3DSurface9> resolved;
        if (FAILED(source.texture->GetSurfaceLevel(0, &resolved)) ||
            FAILED(device_->GetRenderTargetData(resolved.Get(), depth_readback_.Get()))) return;
        pixels.resize(size_t(width_) * height_);
        D3DLOCKED_RECT mapped{};
        if (FAILED(depth_readback_->LockRect(&mapped, nullptr, D3DLOCK_READONLY))) {
            pixels.clear();
            return;
        }
        if (!mapped.pBits || mapped.Pitch < int(width_ * sizeof(float))) {
            depth_readback_->UnlockRect();
            pixels.clear();
            return;
        }
        for (unsigned y = 0; y < height_; ++y)
            std::memcpy(pixels.data() + size_t(y) * width_,
                        static_cast<const uint8_t*>(mapped.pBits) + size_t(y) * mapped.Pitch,
                        size_t(width_) * sizeof(float));
        depth_readback_->UnlockRect();
        if (!std::all_of(pixels.begin(), pixels.end(), [](float z) { return std::isfinite(z) && z >= 0 && z <= 1; })) {
            pixels.clear();
            return;
        }
        if (source.reversed)
            for (auto& pixel : pixels) pixel = 1.f - pixel;
        const auto [minimum, maximum] = std::minmax_element(pixels.begin(), pixels.end());
        if (*maximum - *minimum <= 1e-6f) pixels.clear();
        else if (depth_tracked_.at(slot)) retry_present_depth_ = false;
    }

  private:
    void draw_quad(IDirect3DSurface9* back, IDirect3DTexture9* const (&images)[3], const float (&display)[4]) {
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
        hr(device_->SetDepthStencilSurface(nullptr));
        for (unsigned index = 1; index < target_count_; ++index)
            device_->SetRenderTarget(index, nullptr);
        hr(device_->SetRenderTarget(0, back));
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
        for (unsigned slot = 0; slot < 3; ++slot) {
            hr(device_->SetTexture(slot, images[slot]));
            device_->SetSamplerState(slot, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            device_->SetSamplerState(slot, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            device_->SetSamplerState(slot, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
            device_->SetSamplerState(slot, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
            device_->SetSamplerState(slot, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
            device_->SetSamplerState(slot, D3DSAMP_SRGBTEXTURE, FALSE);
        }
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
