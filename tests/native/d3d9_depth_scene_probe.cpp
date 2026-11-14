// Long-scene D3D9 depth capture fixture: distant geometry across many
// depth-writing passes with an MSAA depth buffer, driving the real
// D3D9Compositor capture/readback path at 2560x1080. Reports frame and
// depth-stage timings so depth readback costs can be iterated locally
// instead of in a game. x86, legacy Direct3DCreate9 device (like TTW).
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "../../src/integration/compatibility/d3d9_compositor.h"

template<class T> using Ptr = Microsoft::WRL::ComPtr<T>;
using Clock = std::chrono::steady_clock;
constexpr UINT width = 2560, height = 1080;

void hr(HRESULT result, const char* stage) {
    if (FAILED(result)) throw std::runtime_error(std::string(stage) + " failed");
}
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Vertex { float x, y, z; DWORD color; };
// A distant "city": many boxes at far depth, drawn in several passes.
void draw_city(IDirect3DDevice9* device, unsigned passes, unsigned boxes) {
    std::vector<Vertex> vertices;
    for (unsigned i = 0; i < boxes; ++i) {
        const float fx = float(i % 64) / 64.f, fy = float(i / 64) / 40.f;
        const float x0 = -1.f + 2.f * fx, y0 = -0.6f + 1.1f * fy;
        const float s = 0.012f;
        const DWORD c = 0xff000000u | (i * 37 & 0xffffff);
        vertices.push_back({x0, y0, 0.995f, c});
        vertices.push_back({x0 + s, y0, 0.995f, c});
        vertices.push_back({x0, y0 + s * 2, 0.998f, c});
        vertices.push_back({x0 + s, y0 + s * 2, 0.998f, c});
    }
    device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE);
    device->SetTexture(0, nullptr);
    for (unsigned pass = 0; pass < passes; ++pass)
        hr(device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, UINT(vertices.size() - 2),
                                   vertices.data(), sizeof(Vertex)), "Draw city");
}

int main(int argc, char** argv) {
    try {
        const unsigned frames = argc > 1 ? unsigned(std::stoul(argv[1])) : 90;
        const unsigned passes = argc > 2 ? unsigned(std::stoul(argv[2])) : 6;
        const bool capture_depth = argc <= 3 || std::string(argv[3]) != "nodepth";
        const auto window = CreateWindowExW(0, L"STATIC", L"Phoenix D3D9 depth scene", WS_POPUP,
                                            0, 0, 320, 180, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        if (!window) throw std::runtime_error("CreateWindow");
        Ptr<IDirect3D9> d3d;
        d3d.Attach(Direct3DCreate9(D3D_SDK_VERSION));
        if (!d3d) throw std::runtime_error("Direct3DCreate9");
        D3DPRESENT_PARAMETERS pp{};
        pp.Windowed = TRUE;
        pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
        pp.hDeviceWindow = window;
        pp.BackBufferWidth = width;
        pp.BackBufferHeight = height;
        pp.BackBufferFormat = D3DFMT_X8R8G8B8;
        pp.EnableAutoDepthStencil = TRUE;
        pp.AutoDepthStencilFormat = D3DFMT_D24S8;
        pp.MultiSampleType = D3DMULTISAMPLE_4_SAMPLES;
        pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
        Ptr<IDirect3DDevice9> device;
        hr(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window,
                             D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED, &pp, &device),
           "CreateDevice");
        // A shadow-cascade-like pass target (ineligible aspect for capture).
        Ptr<IDirect3DTexture9> shadow_tex;
        hr(device->CreateTexture(1024, 1024, 1, D3DUSAGE_RENDERTARGET, D3DFMT_X8R8G8B8,
                                 D3DPOOL_DEFAULT, &shadow_tex, nullptr), "Create shadow texture");
        Ptr<IDirect3DSurface9> shadow_surface, shadow_depth;
        hr(shadow_tex->GetSurfaceLevel(0, &shadow_surface), "Shadow surface");
        hr(device->CreateDepthStencilSurface(1024, 1024, D3DFMT_D24S8, D3DMULTISAMPLE_4_SAMPLES,
                                             0, FALSE, &shadow_depth, nullptr), "Shadow depth");

        phoenix_compat::D3D9Compositor compositor(device.Get(), width, height, D3DFMT_X8R8G8B8);
        Ptr<IDirect3DSurface9> back;
        hr(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back), "GetBackBuffer");
        Ptr<IDirect3DSurface9> main_depth;
        hr(device->GetDepthStencilSurface(&main_depth), "Main depth");
        Ptr<IDirect3DTexture9> staging_tex;
        hr(device->CreateTexture(width, height, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM,
                                 &staging_tex, nullptr), "Staging texture");
        Ptr<IDirect3DSurface9> staging;
        hr(staging_tex->GetSurfaceLevel(0, &staging), "Staging surface");
        std::vector<float> depth(size_t(width) * height);
        Ptr<IDirect3DQuery9> fence;
        hr(device->CreateQuery(D3DQUERYTYPE_EVENT, &fence), "Create event query");
        auto retire = [&] {
            hr(fence->Issue(D3DISSUE_END), "Signal event query");
            const auto deadline = GetTickCount64() + 3000;
            HRESULT status;
            do {
                status = fence->GetData(nullptr, 0, D3DGETDATA_FLUSH);
                if (status == S_FALSE) Sleep(0);
            } while (status == S_FALSE && GetTickCount64() < deadline);
            if (status != S_OK) throw std::runtime_error("Depth fixture event query did not retire");
        };
        std::vector<double> frame_ms, readback_ms;
        for (unsigned frame = 0; frame < frames; ++frame) {
            const auto begin = Clock::now();
            compositor.next_depth_frame(back.Get());
            hr(device->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xff203040, 1.f, 0), "Clear");
            hr(device->BeginScene(), "BeginScene");
            // Shadow pass (different attachment, tracked then abandoned).
            Ptr<IDirect3DSurface9> old_target;
            hr(device->GetRenderTarget(0, &old_target), "GetRT");
            hr(device->SetRenderTarget(0, shadow_surface.Get()), "Set shadow RT");
            hr(device->SetDepthStencilSurface(shadow_depth.Get()), "Set shadow depth");
            compositor.depth_bind(shadow_surface.Get(), shadow_depth.Get());
            draw_city(device.Get(), 1, 400);
            compositor.depth_draw(400 * 4);
            // Main distant pass.
            hr(device->SetRenderTarget(0, back.Get()), "Set back RT");
            hr(device->SetDepthStencilSurface(main_depth.Get()), "Set main depth");
            compositor.depth_bind(back.Get(), main_depth.Get());
            draw_city(device.Get(), passes, 400);
            compositor.depth_draw(passes * 400 * 4);
            hr(device->EndScene(), "EndScene");
            compositor.capture(back.Get(), 0);
            if (capture_depth) compositor.capture_depth(back.Get(), 0);
            const auto read_begin = Clock::now();
            if (capture_depth) {
                retire();
                compositor.stage_depth_readback(0);
                retire();
                const bool ok = compositor.readback_depth_into(0, depth);
                if (frame > 2 && !ok) throw std::runtime_error("Depth readback rejected");
            }
            readback_ms.push_back(ms(read_begin, Clock::now()));
            hr(device->Present(nullptr, nullptr, nullptr, nullptr), "Present");
            frame_ms.push_back(ms(begin, Clock::now()));
        }
        // Standalone breakdown of the same work the compositor copy block
        // does, to separate memory movement from the validation/expand loop.
        double bench_memcpy = 0, bench_loop = 0;
        if (capture_depth) {
            std::vector<float> scaled(depth.begin(), depth.begin() + (width / 2) * (height / 2));
            std::vector<float> out(depth.size());
            const auto b0 = Clock::now();
            for (int rep = 0; rep < 5; ++rep) std::memcpy(out.data(), depth.data(), depth.size() * 4);
            bench_memcpy = ms(b0, Clock::now()) / 5;
            const auto b1 = Clock::now();
            for (int rep = 0; rep < 5; ++rep) {
                bool valid = true;
                float mn = 1e30f, mx = -1e30f;
                const uint32_t step = uint32_t((uint64_t(width / 2) << 16) / width);
                for (unsigned y = 0; y < height; ++y) {
                    const auto* row = scaled.data() + size_t(y * (height / 2) / height) * (width / 2);
                    auto* dest = out.data() + size_t(y) * width;
                    uint32_t sx = 0;
                    for (unsigned x = 0; x < width; ++x, sx += step) {
                        const float v = row[sx >> 16];
                        valid &= std::isfinite(v) && v >= 0 && v <= 1;
                        mn = (std::min)(mn, v); mx = (std::max)(mx, v);
                        dest[x] = v;
                    }
                }
                if (!valid) bench_loop = -1;
            }
            bench_loop = ms(b1, Clock::now()) / 5;
        }
        const auto status = compositor.depth_capture_status();
        std::cout << "{\"bench_memcpy11mb_ms\":" << bench_memcpy
                  << ",\"bench_expand_loop_ms\":" << bench_loop << "}\n";
        std::cout << "{\"passed\":true,\"frames\":" << frames << ",\"passes\":" << passes
                  << ",\"depth\":" << (capture_depth ? "true" : "false")
                  << ",\"frame_ms\":" << median(frame_ms)
                  << ",\"readback_ms\":" << median(readback_ms)
                  << ",\"getdata_ms\":" << compositor.depth_getdata_ms()
                  << ",\"lock_ms\":" << status.value("depth_lock_ms", 0.0)
                  << ",\"memcpy_ms\":" << compositor.depth_memcpy_ms()
                  << ",\"copy_ms\":" << compositor.depth_copy_ms()
                  << ",\"attempts\":" << status.value("tracked_copy_attempts", uint64_t(0))
                  << ",\"copies\":" << status.value("tracked_copies", uint64_t(0))
                  << ",\"resolves\":" << status.value("msaa_resolves", uint64_t(0))
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"passed\":false,\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
