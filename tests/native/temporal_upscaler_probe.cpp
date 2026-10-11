// Exercise the public vendor API on the real rendering adapter. This synthetic
// stationary scene checks execution, state restoration and composition, not
// temporal image quality or game integration.
#include "../../src/native/support/direct.h"
#include "../../src/native/support/resolution_bridge.h"
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <iostream>
#include <fstream>
#include <cmath>
using namespace phoenix_addon;

static uint16_t half(float value) {
    // All fixture values are positive, finite and at least 0.125.
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t(((bits >> 13) & 0x3ffu) | (((bits >> 23) - 112u) << 10));
}
static float real(uint16_t value) {
    const uint32_t exponent = (value >> 10) & 31u, mantissa = value & 1023u;
    if (exponent == 31)
        return NAN;
    return (value & 0x8000u ? -1.f : 1.f) *
           std::ldexp(exponent ? 1.f + mantissa / 1024.f : mantissa / 1024.f,
                      exponent ? int(exponent) - 15 : -14);
}
int wmain(int argc, wchar_t** argv) {
    try {
        check(argc >= 4 && argc <= 8,
              "temporal_upscaler_probe dlss|xess RUNTIME_DIRECTORY RESULT_JSON "
              "[--padded] [--ultrawide] [--packed-color] [--combined-depth]");
        const auto has_option = [&](const wchar_t* option) {
            return std::any_of(argv + 4, argv + argc, [&](const auto* argument) {
                return std::wstring(argument) == option;
            });
        };
        for (int index = 4; index < argc; ++index)
            check(std::wstring(argv[index]) == L"--padded" ||
                      std::wstring(argv[index]) == L"--ultrawide" ||
                      std::wstring(argv[index]) == L"--packed-color" ||
                      std::wstring(argv[index]) == L"--combined-depth",
                  "Unknown probe option");
        const bool ultrawide = has_option(L"--ultrawide");
        const bool packed_color = has_option(L"--packed-color");
        const bool combined_depth = has_option(L"--combined-depth");
        const unsigned output_width = ultrawide ? 2560 : 1920;
        check(std::wstring(argv[1]) == L"dlss" || std::wstring(argv[1]) == L"xess",
              "Unknown probe upscaler");
        const int method = std::wstring(argv[1]) == L"dlss" ? 1 : 2;
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;
        phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        phoenix_native::hr(factory->EnumAdapterByGpuPreference(
            0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
        DXGI_ADAPTER_DESC1 adapter_description{};
        adapter->GetDesc1(&adapter_description);
        ComPtr<ID3D12Device> device;
        phoenix_native::hr(
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> messages;
        device.As(&messages);
        ComPtr<ID3D12InfoQueue1> live_messages;
        if (SUCCEEDED(device.As(&live_messages))) {
            DWORD cookie = 0;
            live_messages->RegisterMessageCallback(
                [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID,
                   LPCSTR text, void*) {
                    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                        std::cerr << text << std::endl;
                },
                D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
        }
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC queue_description{};
        phoenix_native::hr(device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator;
        phoenix_native::hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> command;
        phoenix_native::hr(device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&command)));
        ComPtr<ID3D12Fence> fence;
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        auto geometry = phoenix::fit_resolution(output_width, 1080, 1280, 720);
        // A second run with a padded canvas exercises the vendor input origin.
        if (has_option(L"--padded"))
            geometry = phoenix::fit_resolution(output_width, 1080, 640, 384);
        if (method == 1) {
#ifdef PHOENIX_PROBE_NGX_SDK
            // Offline harness owns its NGX initialization. The addon never
            // initializes or shuts down the host game's context.
            const wchar_t* search[]{argv[2]};
            NVSDK_NGX_FeatureCommonInfo info{};
            info.PathListInfo = {search, 1};
            auto cache = std::filesystem::temp_directory_path() / L"DLSS5Phoenix/ngx_probe";
            std::filesystem::create_directories(cache);
            check(NVSDK_NGX_SUCCEED(NVSDK_NGX_D3D12_Init_with_ProjectID(
                      "853cf9ad-19cb-4f49-96b6-7b3896477d20", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                      cache.c_str(), device.Get(), &info)),
                  "Offline NGX SDK initialization failed");
#else
            check(false, "Build this offline DLSS probe with --ngx-library nvsdk_ngx_d.lib");
#endif
        }
        std::vector<ComPtr<ID3D12Resource>> uploads;
        auto texture = [&](unsigned width, unsigned height, DXGI_FORMAT format, unsigned components,
                           int kind) {
            auto image = phoenix_native::image(device.Get(), width, height, format,
                                               D3D12_RESOURCE_STATE_COPY_DEST);
            const auto description = image->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            UINT64 bytes = 0;
            device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr,
                                          &bytes);
            auto upload = phoenix_native::buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped = nullptr;
            phoenix_native::hr(upload->Map(0, nullptr, &mapped));
            std::memset(mapped, 0, size_t(bytes));
            for (unsigned row = 0; row < height; ++row)
                for (unsigned column = 0; column < width; ++column) {
                    auto* pixel = static_cast<uint8_t*>(mapped) + footprint.Offset +
                                  row * footprint.Footprint.RowPitch +
                                  column * components * (kind == 2 ? 4 : 2);
                    if (kind == 2) {
                        float depth = .8f;
                        std::memcpy(pixel, &depth, 4);
                    } else if (kind != 1) {
                        auto* color = reinterpret_cast<uint16_t*>(pixel);
                        color[0] = half(.125f + .75f * column / width);
                        color[1] = half(.125f + .75f * row / height);
                        color[2] = half(((column / 16 + row / 16) % 2) ? .25f : .75f);
                        color[3] = half(.25f);
                    }
                }
            upload->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
            source.pResource = upload.Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            source.PlacedFootprint = footprint;
            destination.pResource = image.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            phoenix_native::transition(command.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            uploads.push_back(upload);
            return image;
        };
        auto color = texture(geometry.width, geometry.height, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, 0);
        auto motion = texture(geometry.width, geometry.height, DXGI_FORMAT_R16G16_FLOAT, 2, 1);
        auto depth = texture(geometry.width, geometry.height, DXGI_FORMAT_R32_FLOAT, 1, 2);
        auto game = texture(output_width, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, 0);
        phoenix::direct::TemporalUpscaler upscaler(device.Get(), geometry, method, {}, argv[2]);
        auto bridge = std::make_unique<phoenix::direct::ResolutionBridge>(
            device.Get(), geometry, DXGI_FORMAT_R16G16B16A16_FLOAT, nullptr);
        phoenix::direct::Runner* active_runner = nullptr;
        auto submit = [&](unsigned serial) {
            phoenix_native::hr(command->Close());
            ID3D12CommandList* lists[]{command.Get()};
            queue->ExecuteCommandLists(1, lists);
            const bool committed =
                !active_runner || active_runner->after_submit(queue.Get(), 1, lists);
            phoenix_native::hr(queue->Signal(fence.Get(), serial));
            const auto deadline = GetTickCount64() + 60000;
            while (fence->GetCompletedValue() < serial && GetTickCount64() < deadline)
                Sleep(1);
            check(fence->GetCompletedValue() == serial,
                  "Vendor probe submission failed or timed out");
            check(committed, "DLL bridge submission not fenced");
            phoenix_native::hr(allocator->Reset());
            phoenix_native::hr(command->Reset(allocator.Get(), nullptr));
        };
        submit(1);
        uploads.clear();
        // Match the R11G11B10 color surface used by Resident Evil, rather
        // than assuming every host supplies half-precision RGBA input.
        ComPtr<ID3D12Resource> packed;
        std::unique_ptr<phoenix::direct::Pass> convert;
        if (packed_color) {
            packed = phoenix_native::image(device.Get(), geometry.width, geometry.height,
                                           DXGI_FORMAT_R11G11B10_FLOAT,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            convert = std::make_unique<phoenix::direct::Pass>(device.Get(), R"(
Texture2D<float4> source:register(t0);RWTexture2D<float4> target:register(u0);
cbuffer Geometry:register(b0){uint width,height;};
[numthreads(16,16,1)]void main(uint3 pixel:SV_DispatchThreadID){
 if(pixel.x<width&&pixel.y<height)target[pixel.xy]=source.Load(int3(pixel.xy,0));
})");
            convert->record(command.Get(), {color.Get(), nullptr, nullptr}, packed.Get(),
                            {geometry.width, geometry.height, 0, 0, 0, 0, 0, 0, 0});
            phoenix_native::transition(command.Get(), packed.Get(),
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        for (unsigned frame = 0; frame < 12; ++frame) {
            auto output =
                upscaler.record(command.Get(), packed ? packed.Get() : color.Get(), motion.Get(),
                                depth.Get(), frame == 0 || frame == 6, true, 1, 1, 0, 0);
            // Initialization may record work even when the vendor refuses it.
            if (!output) {
                submit(frame + 2);
                throw std::runtime_error(upscaler.status());
            }
            bridge->present_full(command.Get(), output, game.Get(),
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            submit(frame + 2);
        }
        // Identical root signatures may be shared by the D3D12 device. End
        // the unaccounted standalone compositor before measuring runner-owned
        // lifetimes, so it cannot retain a shared pipeline interface.
        bridge.reset();
        convert.reset();
        std::cerr << "Vendor API frames completed; checking DLL bridge" << std::endl;
        phoenix_native::memory::Ledger ledger;
        auto generation = ledger.begin(output_width, 1080);
        {
            phoenix::Settings settings;
            settings.mode = phoenix::nvidia;
            settings.upscaler = method;
            settings.resolution_mode = 1;
            phoenix::direct::Runner runner(
                device.Get(), queue.Get(), {{"width", geometry.width}, {"height", geometry.height}},
                {}, {}, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, false, 1,
                {{1, 1, 1, 1, 1, 1, 1, 1}}, {}, &settings, generation, true, output_width, 1080,
                argv[2]);
            std::cerr << "DLL bridge constructed" << std::endl;
            auto full_motion = texture(output_width, 1080, DXGI_FORMAT_R16G16_FLOAT, 2, 1);
            auto full_depth = texture(output_width, 1080, DXGI_FORMAT_R32_FLOAT, 1, 2);
            ComPtr<ID3D12DescriptorHeap> depth_view;
            // The upload for the original texture is already recorded. Keep it
            // alive through submit(14), even when testing a replacement format.
            ComPtr<ID3D12Resource> uploaded_depth = full_depth;
            if (combined_depth) {
                auto description = full_depth->GetDesc();
                description.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
                description.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                D3D12_HEAP_PROPERTIES properties{};
                properties.Type = D3D12_HEAP_TYPE_DEFAULT;
                full_depth.Reset();
                phoenix_native::hr(device->CreateCommittedResource(
                    &properties, D3D12_HEAP_FLAG_NONE, &description,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&full_depth)));
                D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1,
                                                D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
                phoenix_native::hr(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&depth_view)));
                D3D12_DEPTH_STENCIL_VIEW_DESC view{};
                view.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                auto handle = depth_view->GetCPUDescriptorHandleForHeapStart();
                device->CreateDepthStencilView(full_depth.Get(), &view, handle);
                command->ClearDepthStencilView(
                    handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, .8f, 0, 0, nullptr);
                phoenix_native::transition(command.Get(), full_depth.Get(),
                                           D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            submit(14);
            uploads.clear();
            active_runner = &runner;
            for (unsigned frame = 0; frame < 6; ++frame) {
                Controls controls;
                controls.reset = frame == 0;
                std::cerr << "Bridge frame " << frame << std::endl;
                runner.encode_shared_proxy(command.Get(), game.Get(),
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, controls,
                                           settings);
                runner.prepare_native_motion(command.Get(), full_motion.Get(),
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                runner.prepare_upscale(command.Get(), controls, full_depth.Get(),
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                // The synthetic low surface stands in for NR output. This
                // checks the DLL route's shared resolve/upscale/fence path.
                phoenix_native::transition(command.Get(), color.Get(),
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                runner.resolve_fullscreen(command.Get(), game.Get(),
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                          runner.proxy_image(), color.Get());
                phoenix_native::transition(command.Get(), color.Get(),
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                submit(15 + frame);
                check(runner.upscale_status().find("active") != std::string::npos,
                      runner.upscale_status().c_str());
            }
            check(runner.idle(), "DLL bridge runner did not retire");
            active_runner = nullptr;
        }
        check(ledger.snapshot()["totals"]["live_allocation_info_bytes"] == 0 &&
                  generation->reclaimable(),
              "Temporal DLL bridge retained owned resources");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 bytes = 0;
        auto description = game->GetDesc();
        device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        auto readback = phoenix_native::buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK,
                                               D3D12_RESOURCE_STATE_COPY_DEST);
        phoenix_native::transition(command.Get(), game.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = game.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = readback.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint;
        command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        submit(21);
        void* mapped = nullptr;
        phoenix_native::hr(readback->Map(0, nullptr, &mapped));
        double minimum = 1e30, maximum = -1e30;
        for (unsigned row = 0; row < 1080; ++row)
            for (unsigned column = 0; column < output_width; ++column) {
                const auto* pixel = reinterpret_cast<const uint16_t*>(
                                        static_cast<const uint8_t*>(mapped) + footprint.Offset +
                                        row * footprint.Footprint.RowPitch) +
                                    column * 4;
                for (unsigned channel = 0; channel < 3; ++channel) {
                    const float value = real(pixel[channel]);
                    check(std::isfinite(value), "Nonfinite vendor output");
                    minimum = std::min(minimum, double(value));
                    maximum = std::max(maximum, double(value));
                }
                check(pixel[3] == half(.25f), "Vendor presentation changed game alpha");
            }
        readback->Unmap(0, nullptr);
        check(maximum - minimum > .2, "Vendor output is constant or unwritten");
        if (messages)
            for (UINT64 index = 0; index < messages->GetNumStoredMessages(); ++index) {
                SIZE_T bytes = 0;
                messages->GetMessage(index, nullptr, &bytes);
                std::vector<uint8_t> storage(bytes);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                messages->GetMessage(index, message, &bytes);
                check(message->Severity != D3D12_MESSAGE_SEVERITY_ERROR &&
                          message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION,
                      message->pDescription);
            }
        char hardware[512]{};
        WideCharToMultiByte(CP_UTF8, 0, adapter_description.Description, -1, hardware, 512, nullptr,
                            nullptr);
        Json result{{"passed", true},
                    {"hardware", hardware},
                    {"upscaler", method == 1 ? "NVIDIA DLSS SR" : "Intel XeSS"},
                    {"status", upscaler.status()},
                    {"frames", upscaler.frames()},
                    {"input", {geometry.width, geometry.height}},
                    {"output", {output_width, 1080}},
                    {"finite_nonconstant_output", true},
                    {"alpha_preserved", true},
                    {"debug_errors", 0},
                    {"bridge_only_runner_frames", 6},
                    {"owned_bridge_bytes_after_shutdown", 0},
                    {"packed_color", packed_color},
                    {"combined_depth", combined_depth},
                    {"synthetic_scene", true},
                    {"game_quality_validated", false}};
        std::ofstream(std::filesystem::path(argv[3])) << result.dump(2) << '\n';
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
