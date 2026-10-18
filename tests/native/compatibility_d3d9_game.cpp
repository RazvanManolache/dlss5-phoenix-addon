// A bounded real ReShade/D3D9 fixture, compiled as x86. It exercises addon
// discovery and the generic readback path, rather than calling transport directly.
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <reshade.hpp>
#include <wrl/client.h>
#include "../../third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
template <class T> using Ptr = Microsoft::WRL::ComPtr<T>;
using Json = nlohmann::json;
using Status = uint32_t (*)(char*, uint32_t);
bool complete = false;
std::string failure;
Json evidence;
void observed(reshade::api::effect_runtime*) {
    auto addon = GetModuleHandleW(L"Phoenix.addon32");
    auto status =
        addon ? reinterpret_cast<Status>(GetProcAddress(addon, "CnrPHOENIXCompatibilityStatus"))
              : nullptr;
    if (!status)
        return;
    const auto bytes = status(nullptr, 0);
    std::string text(bytes, '\0');
    status(text.data(), bytes);
    const auto sessions = Json::parse(text.c_str());
    if (sessions.empty())
        return;
    evidence = sessions[0];
    if (!evidence.value("enabled", false)) {
        failure = evidence.value("message", std::string());
        complete = true;
    } else if (!evidence.value("error", std::string()).empty()) {
        failure = evidence["error"];
        complete = true;
    } else if (evidence.value("delivered", 0) >= 3)
        complete = true;
}
int main() {
    try {
        if (!reshade::register_addon(GetModuleHandleW(nullptr)))
            throw std::runtime_error("ReShade x86 is missing/incompatible");
        WNDCLASSW window_class{};
        window_class.lpfnWndProc = DefWindowProcW;
        window_class.hInstance = GetModuleHandleW(nullptr);
        window_class.lpszClassName = L"PhoenixD3D9Fixture";
        RegisterClassW(&window_class);
        HWND window =
            CreateWindowW(window_class.lpszClassName, L"Phoenix DLSS5 D3D9 fixture", WS_POPUP, -30000,
                          -30000, 320, 192, nullptr, nullptr, window_class.hInstance, nullptr);
        if (!window)
            throw std::runtime_error("Cannot create fixture window");
        Ptr<IDirect3D9> d3d;
        d3d.Attach(Direct3DCreate9(D3D_SDK_VERSION));
        if (!d3d)
            throw std::runtime_error("D3D9 creation failed");
        D3DPRESENT_PARAMETERS parameters{};
        parameters.Windowed = TRUE;
        parameters.SwapEffect = D3DSWAPEFFECT_DISCARD;
        parameters.hDeviceWindow = window;
        parameters.BackBufferWidth = 320;
        parameters.BackBufferHeight = 192;
        parameters.BackBufferFormat = D3DFMT_X8R8G8B8;
        parameters.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
        Ptr<IDirect3DDevice9> device;
        if (FAILED(d3d->CreateDevice(0, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING,
                                     &parameters, &device)))
            throw std::runtime_error("D3D9 device creation failed");
        // Runtime creation loads the addon. Register after its callbacks.
        device->Clear(0, nullptr, D3DCLEAR_TARGET, 0xff307080, 1, 0);
        device->Present(nullptr, nullptr, nullptr, nullptr);
        reshade::register_event<reshade::addon_event::reshade_present>(observed);
        const auto until = GetTickCount64() + 90000;
        unsigned frames = 0;
        while (!complete && GetTickCount64() < until) {
            MSG message{};
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            device->Clear(0, nullptr, D3DCLEAR_TARGET, 0xff307080, 1, 0);
            for (unsigned row = 0; row < 6; ++row)
                for (unsigned column = 0; column < 10; ++column)
                    if ((row ^ column) & 1) {
                        D3DRECT rectangle{LONG(column * 32), LONG(row * 32), LONG(column * 32 + 32),
                                          LONG(row * 32 + 32)};
                        device->Clear(1, &rectangle, D3DCLEAR_TARGET, 0xffc09050, 1, 0);
                    }
            device->Present(nullptr, nullptr, nullptr, nullptr);
            ++frames;
            Sleep(5);
        }
        reshade::unregister_event<reshade::addon_event::reshade_present>(observed);
        if (!complete)
            throw std::runtime_error("D3D9 addon produced no completed output within 90 seconds");
        if (!failure.empty())
            throw std::runtime_error(failure);
        evidence["passed"] = true;
        evidence["client_bits"] = sizeof(void*) * 8;
        evidence["fixture_frames"] = frames;
        std::ofstream("fixture_result.json") << evidence.dump(2) << '\n';
        std::cout << evidence.dump() << '\n';
        device.Reset();
        d3d.Reset();
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
