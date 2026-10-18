// Bounded addon-discovery/callback fixture. Unlike the direct bridge probe,
// this loads real ReShade and lets the packaged addon process presentation.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <reshade.hpp>
#include <wrl/client.h>
#include "../../third_party/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
template <class T> using Ptr = Microsoft::WRL::ComPtr<T>;
using Json = nlohmann::json;
using Status = uint32_t (*)(char*, uint32_t);
Json evidence;
bool completed = false;
std::string failure;
void observed(reshade::api::effect_runtime*) {
#ifdef _WIN64
    auto addon = GetModuleHandleW(L"Phoenix.addon64");
#else
    auto addon = GetModuleHandleW(L"Phoenix.addon32");
#endif
    auto status =
        addon ? reinterpret_cast<Status>(GetProcAddress(addon, "CnrPHOENIXCompatibilityStatus"))
              : nullptr;
    if (!status)
        return;
    const auto required = status(nullptr, 0);
    std::string text(required, '\0');
    status(text.data(), required);
    auto sessions = Json::parse(text.c_str());
    if (sessions.empty())
        return;
    evidence = sessions[0];
    if (!evidence.value("enabled", false)) {
        failure = evidence.value("message", std::string());
        completed = true;
    }
    if (!evidence.value("error", std::string()).empty()) {
        failure = evidence["error"];
        completed = true;
    }
    if (evidence.value("delivered", 0) >= 3)
        completed = true;
}
int main() {
    try {
        wchar_t own_path[32768]{};
        GetModuleFileNameW(nullptr, own_path, 32768);
        const auto folder = std::filesystem::path(own_path).parent_path();
        auto reshade = LoadLibraryW((folder / L"dxgi.dll").c_str());
        if (!reshade || !reshade::register_addon(GetModuleHandleW(nullptr), reshade))
            throw std::runtime_error("ReShade is unavailable/incompatible");
        auto d3d11 = LoadLibraryW(L"d3d11.dll");
        using Create = decltype(&D3D11CreateDeviceAndSwapChain);
        auto create =
            reinterpret_cast<Create>(GetProcAddress(d3d11, "D3D11CreateDeviceAndSwapChain"));
        if (!create)
            throw std::runtime_error("D3D11 is unavailable");
        WNDCLASSW window_class{};
        window_class.lpfnWndProc = DefWindowProcW;
        window_class.hInstance = GetModuleHandleW(nullptr);
        window_class.lpszClassName = L"PhoenixD3D11Fixture";
        RegisterClassW(&window_class);
        HWND window =
            CreateWindowW(window_class.lpszClassName, L"Phoenix DLSS5 D3D11 fixture", WS_POPUP, -30000,
                          -30000, 320, 192, nullptr, nullptr, window_class.hInstance, nullptr);
        if (!window)
            throw std::runtime_error("Cannot create fixture window");
        DXGI_SWAP_CHAIN_DESC description{};
        description.BufferDesc.Width = 320;
        description.BufferDesc.Height = 192;
        description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = 2;
        description.OutputWindow = window;
        description.Windowed = TRUE;
        description.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        Ptr<ID3D11Device> device;
        Ptr<ID3D11DeviceContext> context;
        Ptr<IDXGISwapChain> swapchain;
        D3D_FEATURE_LEVEL level{};
        if (FAILED(create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                          D3D11_SDK_VERSION, &description, &swapchain, &device, &level, &context)))
            throw std::runtime_error("D3D11 device/swapchain creation failed");
        Ptr<ID3D11Texture2D> back;
        swapchain->GetBuffer(0, IID_PPV_ARGS(&back));
        Ptr<ID3D11RenderTargetView> target;
        device->CreateRenderTargetView(back.Get(), nullptr, &target);
        swapchain->Present(0, 0);
        reshade::register_event<reshade::addon_event::reshade_present>(observed);
        const auto deadline = GetTickCount64() + 90000;
        unsigned frames = 0;
        while (!completed && GetTickCount64() < deadline) {
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            const float color[]{.25f, .45f, .65f, 1};
            context->ClearRenderTargetView(target.Get(), color);
            swapchain->Present(0, 0);
            ++frames;
            Sleep(5);
        }
        reshade::unregister_event<reshade::addon_event::reshade_present>(observed);
        if (!completed)
            throw std::runtime_error("Packaged D3D11 addon produced no completed output");
        if (!failure.empty())
            throw std::runtime_error(failure);
        evidence["passed"] = true;
        evidence["client_bits"] = sizeof(void*) * 8;
        evidence["fixture_frames"] = frames;
        std::ofstream("fixture_result.json") << evidence.dump(2) << '\n';
        std::cout << evidence.dump() << '\n';
        target.Reset();
        back.Reset();
        context.Reset();
        swapchain.Reset();
        device.Reset();
        DestroyWindow(window);
        reshade::unregister_addon(GetModuleHandleW(nullptr));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::ofstream("fixture_result.json")
            << Json({{"passed", false}, {"error", error.what()}, {"last_status", evidence}})
                   .dump(2);
        return 1;
    }
}
