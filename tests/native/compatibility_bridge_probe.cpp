// Run as x86 and x64 against the same embedded x64 helper. Transport mode
// requires exact pixel identity; inference mode requires a retired, changed
// image. A killed helper must never block the game's D3D11 queue.
#define NOMINMAX
#include "../../src/integration/compatibility/d3d11_bridge.h"
#include <iostream>
using namespace phoenix_compat;
int main(int argc, char** argv) {
    try {
        require(argc >= 3,
                "usage: probe weight_source.json transport|phoenix|nvidia [width height]");
        Json config;
        std::ifstream(argv[1]) >> config;
        auto source = phoenix_model::WeightSource::config(
            config.contains("weight_source") ? config.at("weight_source") : config);
        const std::string mode = argv[2];
        bool transport = mode == "transport" || mode == "cpu_transport";
        phoenix_addon::phoenix::Settings settings;
        settings.mode = mode == "nvidia" ? 1 : mode == "split" ? 2 : 3;
        settings.offload_age_frames = 6000;
        settings.offload_age_ms = 120000;
        unsigned width = argc > 3 ? unsigned(std::stoul(argv[3])) : 128,
                 height = argc > 4 ? unsigned(std::stoul(argv[4])) : 72;
        Ptr<ID3D11Device> device;
        Ptr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL level{};
        hr(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                             D3D11_SDK_VERSION, &device, &level, &context));
        D3D11_TEXTURE2D_DESC description{};
        description.Width = width;
        description.Height = height;
        description.MipLevels = description.ArraySize = 1;
        description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        description.SampleDesc.Count = 1;
        description.BindFlags = D3D11_BIND_RENDER_TARGET;
        Ptr<ID3D11Texture2D> back;
        hr(device->CreateTexture2D(&description, nullptr, &back));
        Ptr<ID3D11RenderTargetView> target;
        hr(device->CreateRenderTargetView(back.Get(), nullptr, &target));
        std::vector<uint8_t> original(size_t(width) * height * 4);
        for (unsigned row = 0; row < height; ++row)
            for (unsigned column = 0; column < width; ++column) {
                auto pixel = (size_t(row) * width + column) * 4;
                original[pixel] = uint8_t(column * 255 / width);
                original[pixel + 1] = uint8_t(row * 255 / height);
                original[pixel + 2] = uint8_t(((column / 8) ^ (row / 8)) & 1 ? 180 : 70);
                original[pixel + 3] = 255;
            }
        if (mode.rfind("cpu_", 0) == 0) {
            FeedBuild build{};
            build.width = build.output_width = width;
            build.height = build.output_height = height;
            build.color_fmt = build.output_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            build.cpu_bridge = 1;
            build.transport = transport;
            build.mv_scale_x = build.mv_scale_y = 1;
            HostClient host(GetModuleHandleW(nullptr), source, settings, build);
            std::vector<uint8_t> result;
            const unsigned cycles = argc > 5 ? unsigned(std::stoul(argv[5])) : 3;
            require(cycles >= 3 && cycles <= 1000, "Invalid bridge measurement count");
            std::vector<double> roundtrips;
            unsigned delivered = 0;
            const auto until = GetTickCount64() + 90000;
            while (delivered < cycles && GetTickCount64() < until) {
                if (host.phase() == Phase::Ready)
                    require(host.submit(delivered + 1, true, original), "CPU submit failed");
                if (host.phase() == Phase::Finished) {
                    result = host.take();
                    if (delivered) roundtrips.push_back(host.roundtrip());
                    ++delivered;
                }
                if (host.phase() == Phase::Failed)
                    throw std::runtime_error(host.error());
                Sleep(2);
            }
            require(delivered == cycles && result.size() == original.size(),
                    "CPU helper produced no completed image");
            size_t changed = 0;
            for (size_t pixel = 0; pixel < result.size(); ++pixel)
                changed += result[pixel] != original[pixel];
            require(transport ? changed == 0 : changed > 0, "CPU image validation failed");
            std::sort(roundtrips.begin(), roundtrips.end());
            std::ofstream raw("output.rgba", std::ios::binary);
            raw.write(reinterpret_cast<const char*>(result.data()), result.size()); raw.close();
            std::cout << Json({{"passed", true},
                               {"mode", mode},
                               {"client_bits", sizeof(void*) * 8},
                               {"delivered", delivered},
                               {"warm_roundtrip_median_ms", roundtrips[roundtrips.size() / 2]},
                               {"width", width},
                               {"height", height},
                               {"changed_bytes", changed},
                               {"output_sha256",
                                phoenix_model::weight_digest(result.data(), result.size())}})
                             .dump()
                      << '\n';
            return 0;
        }
        D3D11Bridge bridge(device.Get(), width, height, description.Format);
        bridge.start(GetModuleHandleW(nullptr), source, settings, transport);
        const auto deadline = GetTickCount64() + 90000;
        unsigned frames = 0;
        while (bridge.delivered() < 3 && GetTickCount64() < deadline) {
            context->UpdateSubresource(back.Get(), 0, nullptr, original.data(), width * 4, 0);
            D3D11_VIEWPORT expected{7, 9, 11, 13, 0, 1};
            context->RSSetViewports(1, &expected);
            bridge.frame(back.Get(), target.Get(), ++frames, settings);
            D3D11_VIEWPORT restored{};
            UINT count = 1;
            context->RSGetViewports(&count, &restored);
            require(count == 1 && restored.TopLeftX == 7 && restored.TopLeftY == 9 &&
                        restored.Width == 11 && restored.Height == 13,
                    "Bridge changed game graphics state");
            if (bridge.host()->phase() == Phase::Failed)
                throw std::runtime_error(bridge.host()->error());
            Sleep(2);
        }
        require(bridge.delivered() >= 3, "No retired helper output");
        // The thin addon receives real device/settings/timing metadata and
        // sends actions without loading the 64-bit inference DLL itself.
        require(!bridge.host()->ui_action(static_cast<phoenix_ui::Action>(999)),
                "Invalid UI command was queued");
        require(bridge.host()->ui_action(phoenix_ui::Action::Marker), "UI marker was not queued");
        Json ui;
        const auto ui_deadline = GetTickCount64() + 5000;
        do {
            ui = bridge.host()->ui_snapshot();
            if (ui.value("markers", uint64_t(0)) >= 1 && ui.contains("resolution")) break;
            context->UpdateSubresource(back.Get(), 0, nullptr, original.data(), width * 4, 0);
            bridge.frame(back.Get(), target.Get(), ++frames, settings);
            Sleep(5);
        } while (GetTickCount64() < ui_deadline);
        require(ui.value("schema", std::string()) == "phoenix-panel/1" &&
                    ui.value("markers", uint64_t(0)) >= 1 && !ui.at("execution_devices").empty(),
                "Helper UI metadata/action round trip failed");
        for (const char* key : {"CpuThreads", "ExecutionDevice", "ResolutionMode", "ResolutionAspect",
                                "TransformerBlocks", "RefinementPasses", "ModelSource", "OverrideStrengths"})
            require(ui.at("settings").contains(key), "Helper omitted a shared setting");
        description.BindFlags = 0;
        description.Usage = D3D11_USAGE_STAGING;
        description.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Ptr<ID3D11Texture2D> readback;
        hr(device->CreateTexture2D(&description, nullptr, &readback));
        context->CopyResource(readback.Get(), back.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        std::vector<uint8_t> result(original.size());
        for (unsigned row = 0; row < height; ++row)
            std::memcpy(result.data() + size_t(row) * width * 4,
                        static_cast<const uint8_t*>(mapped.pData) + size_t(row) * mapped.RowPitch,
                        size_t(width) * 4);
        context->Unmap(readback.Get(), 0);
        size_t changed = 0;
        unsigned maximum = 0;
        for (size_t index = 0; index < result.size(); ++index) {
            unsigned difference = unsigned(std::abs(int(result[index]) - int(original[index])));
            changed += difference != 0;
            maximum = (std::max)(maximum, difference);
        }
        std::cerr << "changed=" << changed << " maximum=" << maximum
                  << " first=" << unsigned(result[0]) << "," << unsigned(result[1]) << ","
                  << unsigned(result[2]) << " expected=" << unsigned(original[0]) << ","
                  << unsigned(original[1]) << "," << unsigned(original[2]) << "\n";
        std::ofstream output(std::string(argv[2]) + "_" + std::to_string(sizeof(void*) * 8) +
                                 ".rgba",
                             std::ios::binary);
        output.write(reinterpret_cast<const char*>(result.data()), result.size());
        output.close();
        require(transport ? changed == 0 : changed > 0,
                "Unexpected output: transport must match, inference must change the image");
        if (mode == "split") {
            for (unsigned row = 0; row < height; ++row)
                require(std::memcmp(result.data() + size_t(row) * width * 4,
                                    original.data() + size_t(row) * width * 4,
                                    size_t(width / 2) * 4) == 0,
                        "Split view changed its original half");
        }
        // A stale result must not replace or modify the next live game frame.
        context->UpdateSubresource(back.Get(), 0, nullptr, original.data(), width * 4, 0);
        auto age_settings = settings;
        age_settings.offload_age_frames = 1;
        frames += 6001;
        bridge.frame(back.Get(), target.Get(), frames, age_settings);
        context->CopyResource(readback.Get(), back.Get());
        hr(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        bool original_preserved = true;
        for (unsigned row = 0; row < height; ++row)
            original_preserved &= std::memcmp(original.data() + size_t(row) * width * 4,
                                              static_cast<const uint8_t*>(mapped.pData) +
                                                  size_t(row) * mapped.RowPitch,
                                              size_t(width) * 4) == 0;
        context->Unmap(readback.Get(), 0);
        require(original_preserved, "Expired helper output altered the current game frame");
        // Destroy the host during a submitted frame. The next game submission
        // and readback still retire; no GPU Wait can be stranded on its fence.
        const auto child = bridge.host()->process_id();
        Handle process;
        process.value = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, child);
        require(process.value != nullptr, "Cannot open owned helper for failure test");
        require(TerminateProcess(process.value, 99) != 0, "Cannot stop owned helper");
        const auto killed = GetTickCount64();
        while (bridge.host()->phase() != Phase::Failed && GetTickCount64() - killed < 5000) {
            bridge.frame(back.Get(), target.Get(), ++frames, settings);
            Sleep(5);
        }
        require(bridge.host()->phase() == Phase::Failed, "Helper death not detected");
        // Even with an expired result and a dead helper, coherent presentation
        // must hold the last whole frame rather than expose the newer game image.
        auto moved = original;
        for (size_t index = 0; index < moved.size(); index += 4) {
            moved[index] ^= 0x7f;
            moved[index + 1] ^= 0x3f;
        }
        context->UpdateSubresource(back.Get(), 0, nullptr, moved.data(), width * 4, 0);
        frames += 6001;
        bridge.frame(back.Get(), target.Get(), frames, age_settings, true);
        context->CopyResource(readback.Get(), back.Get());
        hr(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        bool completed_preserved = true;
        for (unsigned row = 0; row < height; ++row)
            completed_preserved &= std::memcmp(result.data() + size_t(row) * width * 4,
                                               static_cast<const uint8_t*>(mapped.pData) +
                                                   size_t(row) * mapped.RowPitch,
                                               size_t(width) * 4) == 0;
        context->Unmap(readback.Get(), 0);
        require(completed_preserved, "Matched-frame hold leaked a newer game image");
        std::cout << Json({{"passed", true},
                           {"client_bits", sizeof(void*) * 8},
                           {"width", width},
                           {"height", height},
                           {"mode", argv[2]},
                           {"delivered", bridge.delivered()},
                           {"changed_bytes", changed},
                           {"maximum_byte_difference", maximum},
                           {"output_sha256",
                            phoenix_model::weight_digest(result.data(), result.size())},
                           {"helper_death_game_gpu_retired", true},
                           {"expired_output_preserves_current_frame", true},
                           {"helper_ui_metadata_and_marker_verified", true},
                           {"matched_frame_held_after_expiry_and_helper_death", true}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
