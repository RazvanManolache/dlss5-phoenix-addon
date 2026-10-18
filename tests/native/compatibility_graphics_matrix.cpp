// Real ReShade/API integration: a bounded hidden window, known image and
// readback of the actual presented result, before ReShade draws its overlay.
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <d3d10_1.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <GL/gl.h>
#include <wrl/client.h>
#include <wincodec.h>
#include <reshade.hpp>
#include "../../src/integration/compatibility/pixel_transport.h"
#include "../../src/native/model/weight_source.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <map>
#ifdef PHOENIX_TEST_VULKAN
#include "compatibility_vulkan_fixture.h"
#endif
using Json = nlohmann::json;
using namespace reshade::api;
template <class T> using Ptr = Microsoft::WRL::ComPtr<T>;
unsigned width = 257, height = 193;
effect_runtime* runtime = nullptr;
std::map<void*, effect_runtime*> window_runtimes;
HWND first_window = nullptr;
bool multiwindow = false, verify_disabled_window = false;
bool secondary_queue_seen = false;
resource input{}, download{}, resolved_download{};
unsigned d3d9_samples = 1;
Json evidence;
bool capture = false, complete = false;
uint64_t capture_min_delivered = 3;
std::string failure;
std::vector<uint8_t> output;
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void hr(HRESULT result) {
    check(SUCCEEDED(result), "Native API creation failed");
}
void draw_d3d9_input(IDirect3DDevice9* device, resource texture, resource target) {
    // A texture copy cannot populate a multisampled render target. Draw the
    // known image exactly as a game would, with point samples at pixel centers.
    hr(device->SetRenderTarget(0, reinterpret_cast<IDirect3DSurface9*>(uintptr_t(target.handle))));
    hr(device->SetDepthStencilSurface(nullptr));
    D3DVIEWPORT9 viewport{0, 0, width, height, 0, 1};
    hr(device->SetViewport(&viewport));
    hr(device->SetVertexShader(nullptr));
    hr(device->SetPixelShader(nullptr));
    hr(device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1));
    hr(device->SetTexture(0, reinterpret_cast<IDirect3DTexture9*>(uintptr_t(texture.handle))));
    device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
    device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    device->SetSamplerState(0, D3DSAMP_SRGBTEXTURE, FALSE);
    for (auto state : {D3DRS_ZENABLE, D3DRS_ZWRITEENABLE, D3DRS_ALPHATESTENABLE,
                       D3DRS_ALPHABLENDENABLE, D3DRS_SCISSORTESTENABLE, D3DRS_SRGBWRITEENABLE})
        device->SetRenderState(state, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_COLORWRITEENABLE, 15);
    struct Vertex { float x, y, z, rhw, u, v; };
    const Vertex vertices[] = {{-.5f, -.5f, 0, 1, 0, 0}, {float(width)-.5f, -.5f, 0, 1, 1, 0},
        {-.5f, float(height)-.5f, 0, 1, 0, 1}, {float(width)-.5f, float(height)-.5f, 0, 1, 1, 1}};
    hr(device->BeginScene());
    hr(device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, vertices, sizeof(Vertex)));
    hr(device->EndScene());
}
template <class Function> Function function(HMODULE module, const char* name) {
    auto result = reinterpret_cast<Function>(GetProcAddress(module, name));
    check(result != nullptr, name);
    return result;
}
void initialized(effect_runtime* created) {
    runtime = created;
    window_runtimes[created->get_hwnd()] = created;
    // This callback is registered before the addon initializes its Session.
    // Give the two windows opposite settings despite their shared GL_BACK
    // resource handle. The controller must keep the settings per runtime.
    if (multiwindow)
        reshade::set_config_value(created, "Phoenix.Compatibility", "Enabled",
                                  created->get_hwnd() != first_window);
}
Json status() {
#ifdef _WIN64
    auto addon = GetModuleHandleW(L"Phoenix.addon64");
#else
    auto addon = GetModuleHandleW(L"Phoenix.addon32");
#endif
    check(addon != nullptr, "Packaged addon did not load");
    auto query = function<uint32_t (*)(char*, uint32_t)>(addon, "CnrPHOENIXCompatibilityStatus");
    // Status can grow between calls as the asynchronous worker updates timing
    // digits or errors. An undersized export intentionally writes nothing.
    // Check its returned size before parsing, with bounded growth and retries.
    constexpr uint32_t maximum_bytes = 1u << 20, headroom_bytes = 4096;
    uint32_t required = query(nullptr, 0);
    Json result;
    bool received = false;
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        check(required >= 3 && required <= maximum_bytes, "Invalid bridge status size");
        const uint32_t capacity = required > maximum_bytes - headroom_bytes
                                      ? maximum_bytes : required + headroom_bytes;
        std::string text(capacity, '\0');
        required = query(text.data(), capacity);
        if (required > capacity)
            continue;
        check(required >= 3 && text[required - 1] == '\0' && text[0] != '\0',
              "Bridge status export returned an incomplete payload");
        result = Json::parse(text.data(), text.data() + required - 1);
        received = true;
        break;
    }
    check(received, "Bridge status grew beyond the bounded retry budget");
    if (result.size() == 1 && !multiwindow)
        return result[0];
    for (const auto& item : result)
        if (item.value("window_id", uint64_t(0)) == uint64_t(uintptr_t(runtime->get_hwnd())))
            return item;
    throw std::runtime_error("No bridge status for the presenting window");
}
void observe(command_queue* queue, swapchain* chain, const rect*, const rect*, uint32_t,
             const rect*) {
    if (!runtime || complete || capture)
        return;
    if (runtime->get_hwnd() != chain->get_hwnd())
        return;
    if (multiwindow && !verify_disabled_window && chain->get_hwnd() == first_window)
        return;
    try {
        secondary_queue_seen = secondary_queue_seen || runtime->get_command_queue() != queue;
        evidence = status();
        if (verify_disabled_window)
            check(!evidence.value("enabled", true), "Disabled window inherited another session");
        else if (!evidence.value("enabled", false))
            throw std::runtime_error(evidence.value("message", std::string()));
        if (!evidence.value("error", std::string()).empty())
            throw std::runtime_error(evidence["error"]);
        if (!verify_disabled_window &&
            (evidence.value("delivered", uint64_t(0)) < capture_min_delivered ||
             evidence.value("presented", 0) < 1))
            return;
        auto* device = queue->get_device();
        auto back = chain->get_current_back_buffer();
        auto description = device->get_resource_desc(back);
        const auto sample_count = description.texture.samples;
        description.texture.samples = 1;
        description.heap = memory_heap::readback;
        description.usage = resource_usage::copy_dest;
        description.flags = resource_flags::none;
        const auto pitch = phoenix_compat::transfer_row_pitch(device->get_api(), width);
        if (phoenix_compat::uses_buffer_readback(device->get_api()))
            description = resource_desc(uint64_t(pitch) * height, memory_heap::readback,
                                        resource_usage::copy_dest);
        check(device->create_resource(description, nullptr, resource_usage::copy_dest, &download),
              "Cannot create test download");
        auto* command = runtime->get_command_queue()->get_immediate_command_list();
        command->barrier(back, resource_usage::present, resource_usage::copy_source);
        if (description.type == resource_type::buffer)
            command->copy_texture_to_buffer(back, 0, nullptr, download, 0, pitch / 4, height);
        else if (sample_count > 1) {
            auto resolved = description;
            resolved.heap = memory_heap::gpu_only;
            resolved.usage = resource_usage::render_target | resource_usage::resolve_dest | resource_usage::copy_source;
            check(device->create_resource(resolved, nullptr, resource_usage::resolve_dest, &resolved_download),
                  "Cannot create test resolve target");
            command->resolve_texture_region(back, 0, nullptr, resolved_download, 0, 0, 0, 0, resolved.texture.format);
            command->copy_resource(resolved_download, download);
        } else command->copy_resource(back, download);
        command->barrier(back, resource_usage::copy_source, resource_usage::present);
        capture = true;
    } catch (const std::exception& error) {
        failure = error.what();
        complete = true;
    }
}
void finish(command_queue* queue, swapchain* chain) {
    if (!capture || complete)
        return;
    try {
        // This is the validation harness, not the addon render path. Wait only
        // to inspect a screenshot after the API's ordered present submission.
        runtime->get_command_queue()->wait_idle();
        auto* device = queue->get_device();
        auto description = device->get_resource_desc(chain->get_current_back_buffer());
        bool buffers = phoenix_compat::uses_buffer_readback(device->get_api());
        subresource_data mapped{};
        if (buffers) {
            mapped.row_pitch = phoenix_compat::transfer_row_pitch(device->get_api(), width);
            check(device->map_buffer_region(download, 0, uint64_t(mapped.row_pitch) * height,
                                            map_access::read_only, &mapped.data),
                  "Cannot map test buffer");
        } else
            check(device->map_texture_region(download, 0, nullptr, map_access::read_only, &mapped),
                  "Cannot map test texture");
        output.resize(size_t(width) * height * 4);
        // Test normalization is intentionally independent of the bridge helper.
        const bool flip = device->get_api() == device_api::opengl;
        const bool bgra = description.texture.format == format::b8g8r8a8_unorm ||
                          description.texture.format == format::b8g8r8x8_unorm;
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x) {
                const auto* native = static_cast<uint8_t*>(mapped.data) +
                                     size_t(flip ? height - 1 - y : y) * mapped.row_pitch + x * 4;
                auto* pixel = output.data() + (size_t(y) * width + x) * 4;
                if (description.texture.format == format::r10g10b10a2_unorm ||
                    description.texture.format == format::r10g10b10a2_typeless) {
                    uint32_t packed = 0; std::memcpy(&packed, native, 4);
                    for (unsigned channel = 0; channel < 3; ++channel)
                        pixel[channel] = uint8_t((double((packed >> (channel * 10)) & 1023u) / 1023.0 * 255.0) + 0.5);
                } else {
                    pixel[0] = native[bgra ? 2 : 0];
                    pixel[1] = native[1];
                    pixel[2] = native[bgra ? 0 : 2];
                }
                pixel[3] = 255;
            }
        if (buffers)
            device->unmap_buffer_region(download);
        else
            device->unmap_texture_region(download, 0);
        device->destroy_resource(download);
        download = {};
        if (resolved_download.handle) device->destroy_resource(resolved_download);
        resolved_download = {};
        complete = true;
    } catch (const std::exception& error) {
        failure = error.what();
        complete = true;
    }
}
std::vector<uint8_t> pattern(bool native_flip = false, bool native_bgra = false) {
    std::vector<uint8_t> image(size_t(width) * height * 4);
    for (unsigned native_y = 0; native_y < height; ++native_y)
        for (unsigned x = 0; x < width; ++x) {
            const auto y = native_flip ? height - 1 - native_y : native_y;
            auto* pixel = image.data() + (size_t(native_y) * width + x) * 4;
            pixel[0] = uint8_t(x * 255 / width);
            pixel[1] = uint8_t(y * 255 / height);
            pixel[2] = uint8_t(((x / 8) ^ (y / 8)) & 1 ? 180 : 70);
            pixel[3] = 255;
            if (native_bgra)
                std::swap(pixel[0], pixel[2]);
        }
    return image;
}
std::vector<uint8_t> decode_png_rgba(const std::filesystem::path& path) {
    const auto apartment = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    check(SUCCEEDED(apartment) || apartment == RPC_E_CHANGED_MODE,
          "Cannot initialize screenshot decoder");
    struct Uninitialize {
        bool owned;
        ~Uninitialize() {
            if (owned)
                CoUninitialize();
        }
    } scope{SUCCEEDED(apartment)};
    Ptr<IWICImagingFactory> factory;
    hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                        IID_PPV_ARGS(&factory)));
    Ptr<IWICBitmapDecoder> decoder;
    hr(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                          WICDecodeMetadataCacheOnLoad, &decoder));
    Ptr<IWICBitmapFrameDecode> frame;
    hr(decoder->GetFrame(0, &frame));
    UINT decoded_width = 0, decoded_height = 0;
    hr(frame->GetSize(&decoded_width, &decoded_height));
    check(decoded_width == width && decoded_height == height,
          "Screenshot dimensions differ from the actual frame");
    Ptr<IWICFormatConverter> converter;
    hr(factory->CreateFormatConverter(&converter));
    hr(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                             nullptr, 0, WICBitmapPaletteTypeCustom));
    std::vector<uint8_t> pixels(size_t(width) * height * 4);
    hr(converter->CopyPixels(nullptr, width * 4, UINT(pixels.size()), pixels.data()));
    return pixels;
}
void bind_gl_window(HMODULE module, HDC dc, HGLRC context) {
    auto make_current = function<decltype(&wglMakeCurrent)>(module, "wglMakeCurrent");
    // ReShade 6.8 short-circuits its hook when the HGLRC is unchanged, even
    // if the drawable HDC changed. Detach first so both window swapchains are
    // actually registered on the same device instead of silently testing one.
    check(make_current(nullptr, nullptr) && make_current(dc, context),
          "Cannot bind OpenGL context to the requested window");
}
void upload_known_input(device* graphics, std::vector<uint8_t>& rgba) {
    if (graphics->get_resource_desc(input).texture.format == format::r10g10b10a2_unorm) {
        std::vector<uint8_t> packed(rgba.size());
        phoenix_compat::copy_rgb10a2_rows(packed.data(), width * 4, rgba.data(), width * 4, width, height, false);
        graphics->update_texture_region({packed.data(), width * 4, width * height * 4}, input, 0);
    } else
        graphics->update_texture_region({rgba.data(), width * 4, width * height * 4}, input, 0);
}
int main(int argc, char** argv) {
    HWND window = nullptr;
    HWND second_window = nullptr;
    try {
        check(argc == 2 || argc == 3,
              "Specify a graphics API and optional --dynamic, --multiwindow, --control or --resize");
        const std::string api = argv[1];
        const bool rgb10 = api == "d3d12" && GetEnvironmentVariableA("PHOENIX_TEST_RGB10", nullptr, 0) != 0;
        const bool dynamic = argc == 3 && std::string(argv[2]) == "--dynamic";
        multiwindow = argc == 3 && std::string(argv[2]) == "--multiwindow";
        const bool control_test = argc == 3 && (std::string(argv[2]) == "--control" || std::string(argv[2]) == "--msaa-control");
        const bool resize_test = argc == 3 && std::string(argv[2]) == "--resize";
        const bool secondary_queue = argc == 3 && std::string(argv[2]) == "--secondary-queue";
        const bool compute_present = argc == 3 && std::string(argv[2]) == "--compute-present";
        const bool multisample = argc == 3 && (std::string(argv[2]) == "--msaa" || std::string(argv[2]) == "--msaa-control");
        check(argc == 2 || dynamic || multiwindow || control_test || resize_test || secondary_queue || compute_present || multisample, "Unknown fixture option");
        check(!multisample || api == "d3d9", "The multisample fixture currently exercises D3D9");
        check(!(secondary_queue || compute_present) || api == "vulkan", "Separate queues require Vulkan");
        check(!multiwindow || api == "opengl", "Multiwindow scenario currently requires OpenGL");
        check(!resize_test || api == "d3d10" || api == "d3d10_1" || api == "d3d11" || api == "d3d12",
              "Resize scenario currently requires a DXGI swapchain");
        auto reshade = LoadLibraryW(api == "opengl" ? L".\\opengl32.dll" : L".\\dxgi.dll");
        check(reshade && reshade::register_addon(GetModuleHandleW(nullptr), reshade),
              "ReShade unavailable");
        reshade::register_event<reshade::addon_event::init_effect_runtime>(initialized);
        WNDCLASSW klass{};
        klass.lpfnWndProc = DefWindowProcW;
        klass.style = CS_OWNDC;
        klass.hInstance = GetModuleHandleW(nullptr);
        klass.lpszClassName = L"PhoenixGraphicsMatrix";
        RegisterClassW(&klass);
        window = CreateWindowW(klass.lpszClassName, L"Phoenix DLSS5 graphics integration check", WS_POPUP,
                               -30000, -30000, width, height, nullptr, nullptr, klass.hInstance,
                               nullptr);
        check(window != nullptr, "Cannot create test window");
        first_window = window;
        Ptr<IDirect3D9> d9;
        Ptr<IDirect3DDevice9> device9;
        Ptr<ID3D10Device> device10;
        Ptr<ID3D10Device1> device10_1;
        Ptr<ID3D11Device> device11;
        Ptr<ID3D11DeviceContext> context11;
        Ptr<ID3D12Device> device12;
        Ptr<ID3D12CommandQueue> queue12;
        Ptr<IDXGIFactory> factory;
        Ptr<IDXGISwapChain> dxgi_chain;
        HDC dc = nullptr;
        HDC second_dc = nullptr;
        HGLRC gl = nullptr;
        std::function<void()> present;
        std::function<void()> begin_frame = [] {};
        std::function<void()> release_api = [] {};
        DXGI_SWAP_CHAIN_DESC swap{};
        swap.BufferDesc.Width = width;
        swap.BufferDesc.Height = height;
        swap.BufferDesc.Format = rgb10 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM;
        swap.SampleDesc.Count = 1;
        swap.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap.BufferCount = 2;
        swap.OutputWindow = window;
        swap.Windowed = TRUE;
        swap.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        if (api == "d3d9") {
            auto module = LoadLibraryW(L"d3d9.dll");
            d9.Attach(
                function<decltype(&Direct3DCreate9)>(module, "Direct3DCreate9")(D3D_SDK_VERSION));
            check(bool(d9), "D3D9 unavailable");
            D3DPRESENT_PARAMETERS swap9{};
            swap9.Windowed = TRUE;
            swap9.SwapEffect = D3DSWAPEFFECT_DISCARD;
            swap9.hDeviceWindow = window;
            swap9.BackBufferWidth = width;
            swap9.BackBufferHeight = height;
            swap9.BackBufferFormat = D3DFMT_X8R8G8B8;
            swap9.BackBufferCount = 1;
            if (multisample) {
                for (auto samples : {D3DMULTISAMPLE_8_SAMPLES, D3DMULTISAMPLE_4_SAMPLES, D3DMULTISAMPLE_2_SAMPLES}) {
                    DWORD quality = 0;
                    if (SUCCEEDED(d9->CheckDeviceMultiSampleType(0, D3DDEVTYPE_HAL, swap9.BackBufferFormat, TRUE, samples, &quality)) && quality) {
                        swap9.MultiSampleType = samples;
                        d3d9_samples = unsigned(samples);
                        break;
                    }
                }
                check(d3d9_samples > 1, "Windowed D3D9 multisampling is unsupported by this device");
            }
            swap9.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
            hr(d9->CreateDevice(0, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING,
                                &swap9, &device9));
            present = [&] { hr(device9->Present(nullptr, nullptr, nullptr, nullptr)); };
        } else if (api == "d3d10" || api == "d3d10_1") {
            if (api == "d3d10") {
                auto module = LoadLibraryW(L"d3d10.dll");
                hr(function<decltype(&D3D10CreateDeviceAndSwapChain)>(
                    module, "D3D10CreateDeviceAndSwapChain")(nullptr, D3D10_DRIVER_TYPE_HARDWARE,
                                                             nullptr, 0, D3D10_SDK_VERSION, &swap,
                                                             &dxgi_chain, &device10));
            } else {
                auto module = LoadLibraryW(L"d3d10_1.dll");
                hr(function<decltype(&D3D10CreateDeviceAndSwapChain1)>(
                    module, "D3D10CreateDeviceAndSwapChain1")(
                    nullptr, D3D10_DRIVER_TYPE_HARDWARE, nullptr, 0, D3D10_FEATURE_LEVEL_10_1,
                    D3D10_1_SDK_VERSION, &swap, &dxgi_chain, &device10_1));
            }
            present = [&] { hr(dxgi_chain->Present(0, 0)); };
        } else if (api == "d3d11") {
            auto module = LoadLibraryW(L"d3d11.dll");
            D3D_FEATURE_LEVEL level{};
            hr(function<decltype(&D3D11CreateDeviceAndSwapChain)>(module,
                                                                  "D3D11CreateDeviceAndSwapChain")(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &swap,
                &dxgi_chain, &device11, &level, &context11));
            present = [&] { hr(dxgi_chain->Present(0, 0)); };
        } else if (api == "d3d12") {
            auto module = LoadLibraryW(L"d3d12.dll");
            hr(function<decltype(&D3D12CreateDevice)>(module, "D3D12CreateDevice")(
                nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device12)));
            D3D12_COMMAND_QUEUE_DESC description{};
            hr(device12->CreateCommandQueue(&description, IID_PPV_ARGS(&queue12)));
            hr(function<decltype(&CreateDXGIFactory)>(reshade,
                                                      "CreateDXGIFactory")(IID_PPV_ARGS(&factory)));
            swap.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            hr(factory->CreateSwapChain(queue12.Get(), &swap, &dxgi_chain));
            present = [&] { hr(dxgi_chain->Present(0, 0)); };
        } else if (api == "opengl") {
            auto module = LoadLibraryW(L"opengl32.dll");
            dc = GetDC(window);
            PIXELFORMATDESCRIPTOR pixel{};
            pixel.nSize = sizeof(pixel);
            pixel.nVersion = 1;
            pixel.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pixel.iPixelType = PFD_TYPE_RGBA;
            pixel.cColorBits = 32;
            pixel.cAlphaBits = 8;
            auto number = ChoosePixelFormat(dc, &pixel);
            check(number && SetPixelFormat(dc, number, &pixel), "Cannot configure GL pixel format");
            gl = function<decltype(&wglCreateContext)>(module, "wglCreateContext")(dc);
            check(gl && function<decltype(&wglMakeCurrent)>(module, "wglMakeCurrent")(dc, gl),
                  "Cannot create OpenGL context");
            present = [&] { check(SwapBuffers(dc), "OpenGL presentation failed"); };
        }
#ifdef PHOENIX_TEST_VULKAN
        else if (api == "vulkan") {
            auto fixture = std::make_shared<VulkanFixture>(window, width, height, secondary_queue, compute_present);
            begin_frame = [fixture] { fixture->acquire(); };
            present = [fixture] { fixture->present(); };
            release_api = [&, fixture] {
                present = {};
                begin_frame = {};
            };
        }
#endif
        else
            throw std::runtime_error("Unknown test API");
        present();
        check(runtime != nullptr, "ReShade runtime did not initialize");
        effect_runtime* first_runtime = runtime;
        if (multiwindow) {
            second_window = CreateWindowW(klass.lpszClassName, L"Phoenix DLSS5 second graphics window",
                                          WS_POPUP, -30000, -30000, width, height, nullptr, nullptr,
                                          klass.hInstance, nullptr);
            check(second_window != nullptr, "Cannot create second OpenGL window");
            second_dc = GetDC(second_window);
            PIXELFORMATDESCRIPTOR pixel{};
            const int pixel_format = GetPixelFormat(dc);
            check(DescribePixelFormat(dc, pixel_format, sizeof(pixel), &pixel) &&
                      SetPixelFormat(second_dc, pixel_format, &pixel),
                  "Cannot share pixel format with second window");
            auto module = GetModuleHandleW(L"opengl32.dll");
            bind_gl_window(module, second_dc, gl);
            present = [&] { check(SwapBuffers(second_dc), "Second OpenGL presentation failed"); };
            present();
            check(runtime != first_runtime && runtime->get_device() == first_runtime->get_device(),
                  "Two windows did not expose independent runtimes on a shared device");
            check(runtime->get_current_back_buffer().handle ==
                      first_runtime->get_current_back_buffer().handle,
                  "Scenario did not reproduce the aliased OpenGL backbuffer identity");
        }
        effect_runtime* model_runtime = runtime;
        reshade::register_event<reshade::addon_event::present>(observe);
        reshade::register_event<reshade::addon_event::finish_present>(finish);
        auto* device = runtime->get_device();
        auto* queue = runtime->get_command_queue();
        const auto expected_api = api == "d3d9"                        ? device_api::d3d9
                                  : api == "d3d10" || api == "d3d10_1" ? device_api::d3d10
                                  : api == "d3d11"                     ? device_api::d3d11
                                  : api == "d3d12"                     ? device_api::d3d12
                                  : api == "opengl"                    ? device_api::opengl
                                                                       : device_api::vulkan;
        check(device->get_api() == expected_api,
              "The test initialized a different API than requested");
        if (api == "vulkan")
            check((device->get_resource_desc(runtime->get_current_back_buffer()).usage & resource_usage::copy_dest)
                    == resource_usage::copy_dest,
                  "The addon did not request transfer-destination usage for its Vulkan output");
        resource_desc image(resource_type::texture_2d, width, height, 1, 1, format::r8g8b8a8_unorm,
                            1, memory_heap::gpu_only,
                            resource_usage::copy_source | resource_usage::copy_dest |
                                resource_usage::shader_resource | resource_usage::render_target);
        const auto back_format =
            device->get_resource_desc(runtime->get_current_back_buffer()).texture.format;
        int display_mode = 3;
        reshade::get_config_value(runtime, "Phoenix", "Backend", display_mode);
        const bool bgra_input = api == "vulkan" && back_format == format::b8g8r8a8_unorm;
        if (bgra_input || rgb10)
            image.texture.format = back_format;
        check(device->create_resource(image, nullptr, resource_usage::copy_source, &input),
              "Cannot create known input texture");
        // ReShade normalizes OpenGL texture-to-backbuffer copies internally.
        // Pre-flipping here would hide an upside-down neural input/output pair.
        auto pixels = pattern(false, bgra_input);
        upload_known_input(device, pixels);
        const auto deadline = GetTickCount64() + 90000;
        const auto pump_frame = [&] {
            if (multiwindow) {
                auto module = GetModuleHandleW(L"opengl32.dll");
                bind_gl_window(module, dc, gl);
                runtime = first_runtime;
                auto* first_queue = runtime->get_command_queue();
                auto back = runtime->get_current_back_buffer();
                auto* command = first_queue->get_immediate_command_list();
                command->barrier(back, resource_usage::present, resource_usage::copy_dest);
                command->copy_resource(input, back);
                command->barrier(back, resource_usage::copy_dest, resource_usage::present);
                first_queue->flush_immediate_command_list();
                check(SwapBuffers(dc), "First OpenGL presentation failed");
                check(!status().value("enabled", true),
                      "First window did not keep its disabled setting");
                bind_gl_window(module, second_dc, gl);
                runtime = model_runtime;
            }
            begin_frame();
            auto back = runtime->get_current_back_buffer();
            auto* command = queue->get_immediate_command_list();
            command->barrier(back, resource_usage::present, resource_usage::copy_dest);
            if (multisample) draw_d3d9_input(device9.Get(), input, back);
            else command->copy_resource(input, back);
            command->barrier(back, resource_usage::copy_dest, resource_usage::present);
            if (api != "vulkan")
                queue->flush_immediate_command_list();
            present();
            Sleep(5);
        };
        bool current_changed = false;
        while (!complete && GetTickCount64() < deadline) {
            if (dynamic && !current_changed) {
                const auto progress = status();
                if (progress.value("delivered", 0) >= 2 && progress.value("phase", 0) == 4) {
                    for (size_t pixel = 0; pixel < pixels.size(); pixel += 4) {
                        pixels[pixel] = uint8_t((std::min)(255, int(pixels[pixel]) + 17));
                        pixels[pixel + 1] = uint8_t((std::min)(255, int(pixels[pixel + 1]) + 23));
                        pixels[pixel + 2] = uint8_t((std::max)(0, int(pixels[pixel + 2]) - 10));
                    }
                    upload_known_input(device, pixels);
                    current_changed = true;
                }
            }
            pump_frame();
        }
        check(complete, "No completed and presented image in 90 seconds");
        check(failure.empty(), failure.c_str());
        check(!output.empty(), "No output pixel evidence");
        std::ofstream raw("output.rgba", std::ios::binary);
        raw.write(reinterpret_cast<const char*>(output.data()), output.size());
        raw.close();
        const auto original = pattern();
        size_t changed = 0;
        for (size_t index = 0; index < output.size(); ++index)
            changed += output[index] != original[index];
        check(changed > output.size() / 100,
              "Bridge completed but presented the unprocessed input");
        // ReShade may bind directly to a second graphics-capable queue. A
        // compute-only presenter must instead cross to its graphics queue.
        check(!compute_present || secondary_queue_seen,
              "Compute presentation did not exercise a separate runtime graphics queue");
        evidence["two_graphics_queues_requested"] = secondary_queue;
        evidence["secondary_present_queue"] = secondary_queue_seen;
        evidence["compute_presentation_family"] = compute_present;
        evidence["passed"] = true;
        evidence["d3d9_samples"] = d3d9_samples;
        evidence["fixture_api"] = api;
        evidence["rgb10_sdr"] = rgb10;
        evidence["bits"] = sizeof(void*) * 8;
        evidence["output_sha256"] = phoenix_model::weight_digest(output.data(), output.size());
        const bool in_process = evidence.value("execution_mode", std::string()) == "in_process";
        if (GetEnvironmentVariableA("PHOENIX_EXPECT_INPROCESS", nullptr, 0)) {
            check(in_process && evidence.value("helper_pid", 1) == 0 &&
                  evidence.value("composition", std::string()) == "same_frame_gpu",
                  "In-process test silently used a helper or unproven composition");
        }
        evidence["changed_bytes"] = changed;
        evidence["dynamic_current_changed"] = current_changed;
        evidence["display_mode"] = display_mode;
        check(!dynamic || current_changed || in_process, "Dynamic frame was never injected");
        if (resize_test) {
            const auto initial_evidence = evidence;
            const auto initial_hash = evidence.at("output_sha256").get<std::string>();
            Json resized = Json::array();
            // Return to the original extent after two odd-sized canvases. The
            // addon must relinquish all backbuffer views before ResizeBuffers,
            // then process fresh input with a correctly sized replacement host.
            for (const auto extent : {std::pair{321u, 241u}, std::pair{641u, 361u},
                                      std::pair{257u, 193u}}) {
                queue->wait_idle();
                device->destroy_resource(input);
                input = {};
                if (context11) { context11->ClearState(); context11->Flush(); }
                if (device10) { device10->ClearState(); device10->Flush(); }
                if (device10_1) { device10_1->ClearState(); device10_1->Flush(); }
                width = extent.first;
                height = extent.second;
                const auto started = GetTickCount64();
                const auto result = dxgi_chain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
                check(SUCCEEDED(result), "ResizeBuffers failed: a backbuffer reference may still be retained");
                const auto resize_ms = GetTickCount64() - started;
                present();
                check(runtime != nullptr, "Runtime did not survive ResizeBuffers");
                device = runtime->get_device();
                queue = runtime->get_command_queue();
                image.texture.width = width;
                image.texture.height = height;
                check(device->create_resource(image, nullptr, resource_usage::copy_source, &input),
                      "Cannot create resized input texture");
                pixels = pattern(false, bgra_input);
                upload_known_input(device, pixels);
                complete = capture = false;
                capture_min_delivered = 3;
                failure.clear();
                const auto until = GetTickCount64() + 60000;
                while (!complete && GetTickCount64() < until) pump_frame();
                check(complete && failure.empty(), "Resized canvas never produced a presented result");
                check(output.size() == size_t(width) * height * 4,
                      "Resize produced the wrong output extent");
                const auto resized_input = pattern();
                size_t processed_bytes = 0;
                for (size_t index = 0; index < output.size(); ++index)
                    processed_bytes += output[index] != resized_input[index];
                check(processed_bytes > output.size() / 100,
                      "Resized canvas displayed raw or stale input");
                if (api == "d3d11")
                    check(evidence.value("transfer_cpu_ms", Json::object()).value("sharing_device_reused", false),
                          "D3D11 resize discarded the game-device sharing context");
                const auto hash = phoenix_model::weight_digest(output.data(), output.size());
                if (width == 257 && height == 193)
                    check(hash == initial_hash, "Returning to original extent changed the deterministic output");
                resized.push_back({{"width", width}, {"height", height}, {"resize_ms", resize_ms},
                    {"output_sha256", hash}, {"processed_bytes", processed_bytes},
                    {"delivered", evidence.value("delivered", uint64_t(0))}});
            }
            evidence = initial_evidence;
            evidence["resizes"] = std::move(resized);
            evidence["resize_cycle_processed_and_exact_on_return"] = true;
        }
        if (dynamic) {
            if (in_process && !current_changed) {
                for (size_t pixel = 0; pixel < pixels.size(); pixel += 4) {
                    pixels[pixel] = uint8_t((std::min)(255, int(pixels[pixel]) + 17));
                    pixels[pixel + 1] = uint8_t((std::min)(255, int(pixels[pixel + 1]) + 23));
                    pixels[pixel + 2] = uint8_t((std::max)(0, int(pixels[pixel + 2]) - 10));
                }
                upload_known_input(device, pixels);
                current_changed = true;
                evidence["dynamic_current_changed"] = true;
                evidence["composition"] = "same_frame_gpu";
            }
            // Holding the old pair once is only half of the contract. Keep the
            // changed input on screen long enough to pass through capture,
            // inference and presentation; a permanently frozen bridge must fail.
            const auto held_output = output;
            const auto held_evidence = evidence;
            capture_min_delivered = evidence.value("delivered", uint64_t(0)) + 3;
            complete = capture = false;
            const auto changed_deadline = GetTickCount64() + 60000;
            while (!complete && GetTickCount64() < changed_deadline)
                pump_frame();
            check(complete && failure.empty(), "Changed input never produced a presented result");
            check(output != held_output, "Bridge kept displaying the old pair indefinitely");
            auto changed_input = pixels;
            if (bgra_input)
                for (size_t pixel = 0; pixel < changed_input.size(); pixel += 4)
                    std::swap(changed_input[pixel], changed_input[pixel + 2]);
            size_t processed_bytes = 0;
            const bool split = display_mode == 2;
            for (unsigned row = 0; row < height; ++row) {
                if (split)
                    check(std::memcmp(output.data() + size_t(row) * width * 4,
                                      changed_input.data() + size_t(row) * width * 4,
                                      size_t(width / 2) * 4) == 0,
                          "Changed split result has no matching changed input");
                for (unsigned column = split ? width / 2 : 0; column < width; ++column)
                    for (unsigned channel = 0; channel < 3; ++channel) {
                        const auto index = (size_t(row) * width + column) * 4 + channel;
                        processed_bytes += output[index] != changed_input[index];
                    }
            }
            check(processed_bytes > output.size() / 100,
                  "Changed result displayed the raw game rather than processed input");
            std::ofstream changed_file("output_changed.rgba", std::ios::binary);
            changed_file.write(reinterpret_cast<const char*>(output.data()), output.size());
            changed_file.close();
            const auto counters = evidence.value("transfer_cpu_ms", Json::object());
            if (counters.contains("current_snapshot_copies"))
                check(counters["current_snapshot_copies"] == counters["input_submissions"],
                      "Coherent display copied current pixels without submitting an input");
            const auto final_evidence = evidence;
            evidence = held_evidence;
            evidence["changed_output_sha256"] =
                phoenix_model::weight_digest(output.data(), output.size());
            evidence["changed_input_eventually_processed"] = true;
            evidence["changed_input_processed_bytes"] = processed_bytes;
            evidence["continued_delivered"] = final_evidence["delivered"];
            evidence["continued_transfer_counters"] = counters;
        }
        if (multiwindow) {
            const auto processed_evidence = evidence;
            auto module = GetModuleHandleW(L"opengl32.dll");
            bind_gl_window(module, dc, gl);
            runtime = first_runtime;
            complete = capture = false;
            verify_disabled_window = true;
            auto* first_queue = runtime->get_command_queue();
            auto* command = first_queue->get_immediate_command_list();
            auto back = runtime->get_current_back_buffer();
            command->barrier(back, resource_usage::present, resource_usage::copy_dest);
            command->copy_resource(input, back);
            command->barrier(back, resource_usage::copy_dest, resource_usage::present);
            first_queue->flush_immediate_command_list();
            check(SwapBuffers(dc), "Disabled window screenshot failed");
            check(complete && failure.empty() && output == original,
                  "Disabled window displayed another window's processed image");
            evidence = processed_evidence;
            evidence["aliased_backbuffer_window_isolation_verified"] = true;
            evidence["disabled_window_sha256"] =
                phoenix_model::weight_digest(output.data(), output.size());
        }
        if (control_test) {
            const auto completed_evidence = evidence;
            const auto folder =
                std::filesystem::u8path(status().at("control_folder").get<std::string>());
            const auto endpoint_path = folder / "endpoint.json";
            const auto endpoint_deadline = GetTickCount64() + 5000;
            while (!std::filesystem::exists(endpoint_path) && GetTickCount64() < endpoint_deadline)
                pump_frame();
            check(std::filesystem::is_regular_file(endpoint_path),
                  "Control endpoint was not published");
            Json endpoint;
            std::ifstream(endpoint_path) >> endpoint;
            check(endpoint.at("schema") == "phoenix-control/1" &&
                      endpoint.at("pid") == GetCurrentProcessId(),
                  "Control endpoint belongs to another process");
            unsigned sequence = 0;
            Json responses = Json::array();
            const auto request = [&](const char* operation, Json extra = Json::object()) {
                const auto id = "fixture-" + std::to_string(++sequence);
                Json message{{"schema", "phoenix-control/1"},
                             {"session", endpoint.at("session")},
                             {"pid", GetCurrentProcessId()},
                             {"expires_tick_ms", GetTickCount64() + 30000},
                             {"id", id},
                             {"op", operation}};
                message.update(extra);
                const auto temporary = folder / "request-fixture.tmp",
                           incoming = folder / "request.json";
                check(!std::filesystem::exists(incoming),
                      "Another control request is already pending");
                {
                    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
                    stream << message.dump();
                    stream.close();
                    check(bool(stream), "Cannot write fixture control request");
                }
                check(MoveFileExW(temporary.c_str(), incoming.c_str(), MOVEFILE_WRITE_THROUGH),
                      "Cannot publish fixture control request");
                const auto response_path = folder / "response.json";
                const auto response_deadline = GetTickCount64() + 10000;
                while (GetTickCount64() < response_deadline) {
                    pump_frame();
                    if (!std::filesystem::exists(response_path))
                        continue;
                    Json response;
                    std::ifstream(response_path) >> response;
                    if (response.value("id", std::string()) != id)
                        continue;
                    check(response.value("ok", false), "Addon rejected fixture control operation");
                    responses.push_back(response);
                    return response;
                }
                throw std::runtime_error("Addon did not acknowledge fixture control operation");
            };
            request("status");
            request("set", {{"enabled", false}});
            check(!status().value("enabled", true) && status().value("phase", 0) == 6,
                  "Disabling through the addon bridge did not retire inference");
            request("set", {{"enabled", true}, {"settings", {{"Backend", 0}}}});
            check(status().at("applied_settings").at("Backend") == 0 &&
                      status().value("phase", 0) == 6,
                  "Input mode did not apply as a real bypass");
            request("set", {{"enabled", true}, {"settings", {{"Backend", 3}}}});
            const auto restart_deadline = GetTickCount64() + 60000;
            while (status().value("delivered", 0) < 3 && GetTickCount64() < restart_deadline)
                pump_frame();
            check(status().value("delivered", 0) >= 3 &&
                      status().at("applied_settings").at("Backend") == 3,
                  "Phoenix DLSS5 did not resume producing output after control reapply");
            // Obtain an independent actual-present screenshot of the resumed
            // Phoenix mode. The initial mode may have been split comparison.
            complete = capture = false;
            const auto expected_deadline = GetTickCount64() + 10000;
            while (!complete && GetTickCount64() < expected_deadline)
                pump_frame();
            check(complete && failure.empty(), "Cannot capture screenshot reference after reapply");
            const auto expected_screenshot = output;
            const auto shot = request("screenshot");
            const auto postfix = shot.at("screenshot_postfix").get<std::string>();
            std::filesystem::path screenshot;
            const auto screenshot_deadline = GetTickCount64() + 10000;
            while (screenshot.empty() && GetTickCount64() < screenshot_deadline) {
                pump_frame();
                // ReShade caches the screenshot directory at initialization.
                // Accept its real runtime file even when it retained the game
                // directory rather than a subsequently edited INI SavePath.
                for (const auto& directory :
                     {folder.parent_path(), std::filesystem::current_path()}) {
                    for (const auto& file : std::filesystem::directory_iterator(directory)) {
                        const auto extension = file.path().extension().string();
                        if (file.is_regular_file() &&
                            file.path().filename().string().find(postfix) != std::string::npos &&
                            (extension == ".png" || extension == ".bmp" || extension == ".jpg") &&
                            file.file_size() > 64)
                            screenshot = file.path();
                    }
                }
            }
            check(!screenshot.empty(), "Screenshot operation produced no image file");
            const auto screenshot_pixels = decode_png_rgba(screenshot);
            check(screenshot_pixels == expected_screenshot,
                  "Addon screenshot pixels differ from the actual processed frame");
            evidence = completed_evidence;
            evidence["control_endpoint_verified"] = true;
            evidence["control_disable_input_phoenix_verified"] = true;
            evidence["control_screenshot_verified"] = true;
            evidence["control_screenshot_pixels_identical"] = true;
            evidence["control_screenshot_pixels_sha256"] =
                phoenix_model::weight_digest(screenshot_pixels.data(), screenshot_pixels.size());
            evidence["control_screenshot"] = screenshot.string();
            evidence["control_responses"] = std::move(responses);
            evidence["control_final_status"] = status();
        }
        std::ofstream("result.json") << evidence.dump(2);
        std::cout << evidence.dump() << '\n';
        queue->wait_idle();
        device->destroy_resource(input);
        reshade::unregister_event<reshade::addon_event::present>(observe);
        reshade::unregister_event<reshade::addon_event::finish_present>(finish);
        // Destroy the native surface before returning so helper teardown is checked.
        dxgi_chain.Reset();
        device9.Reset();
        d9.Reset();
        release_api();
        release_api = {};
        if (gl) {
            auto module = GetModuleHandleW(L"opengl32.dll");
            function<decltype(&wglMakeCurrent)>(module, "wglMakeCurrent")(nullptr, nullptr);
            function<decltype(&wglDeleteContext)>(module, "wglDeleteContext")(gl);
            ReleaseDC(window, dc);
            if (second_dc)
                ReleaseDC(second_window, second_dc);
        }
        if (second_window)
            DestroyWindow(second_window);
        DestroyWindow(window);
        return 0;
    } catch (const std::exception& error) {
        std::ofstream("result.json")
            << Json({{"passed", false}, {"error", error.what()}, {"last_status", evidence}})
                   .dump(2);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
