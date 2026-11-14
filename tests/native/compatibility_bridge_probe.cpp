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
        const bool single_threaded = argc > 6 && std::string(argv[6]) == "singlethread";
        hr(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                             single_threaded ? D3D11_CREATE_DEVICE_SINGLETHREADED : 0, nullptr, 0,
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
        if (mode == "cpu_motion") {
            require(argc >= 8 && argc <= 11, "Motion benchmark requires frames, provider and analysis height");
            const unsigned queue_depth = argc >= 9 ? unsigned(std::stoul(argv[8])) : 0;
            require(argc == 8 || (queue_depth >= 1 && queue_depth <= 4), "Invalid benchmark queue depth");
            const unsigned frames = std::stoul(argv[5]);
            settings.pipeline.motion_source = std::stoi(argv[6]);
            settings.pipeline.flow_height = std::stoi(argv[7]);
            settings.pipeline.motion_enabled = true;
            settings.pipeline.scene_detection = false;
            settings.pipeline.monitoring = int(phoenix_pipeline::Monitoring::Detailed);
            const bool full_depth = argc >= 10 && std::string_view(argv[9]) == "full-depth";
            const bool game_depth = full_depth || (argc >= 10 && std::string_view(argv[9]) == "depth");
            settings.pipeline.legacy_depth = game_depth ? 4 : 3;
            settings.refinement_passes = argc == 11 ? std::stoi(argv[10]) : 2;
            require(settings.pipeline.valid(), "Invalid motion benchmark settings");
            require(frames >= 24 && frames <= 1000, "Invalid motion benchmark frame count");
            FeedBuild build{};
            build.width = build.output_width = width;
            build.height = build.output_height = height;
            build.color_fmt = build.output_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            build.cpu_bridge = 1;
            build.mv_scale_x = build.mv_scale_y = 1;
            HostClient host(GetModuleHandleW(nullptr), source, settings, build);
            // A textured background and a separately moving, opaque foreground
            // provide changing history, motion boundaries and disocclusion.
            auto pattern = [](unsigned x, unsigned y) {
                uint32_t hash = (x / 5) * 0x9e3779b9u + (y / 5) * 0x85ebca6bu;
                hash ^= hash >> 16; hash *= 0x7feb352du; hash ^= hash >> 15;
                return hash;
            };
            std::vector<double> roundtrip;
            std::vector<uint8_t> output;
            unsigned submitted = 0, delivered = 0;
            using Clock = std::chrono::steady_clock;
            Handle poll_timer;
            if (queue_depth) {
                poll_timer.value = CreateWaitableTimerExW(nullptr, nullptr,
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_MODIFY_STATE | SYNCHRONIZE);
                require(poll_timer.value != nullptr, "High-resolution ring benchmark timer unavailable");
            }
            struct Flight { Clock::time_point capture; };
            std::deque<Flight> outstanding;
            auto capture = [&](std::span<uint8_t> pixels, unsigned frame) {
                const unsigned left = width / 4 + frame * 6 % (width / 3);
                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                    const bool foreground = x >= left && x < left + width / 5 && y > height / 3 && y < 2 * height / 3;
                    const auto hash = foreground ? pattern(x - left, y + 512) : pattern(x + frame * 3, y + frame);
                    auto* rgba = pixels.data() + (size_t(y) * width + x) * 4;
                    for (unsigned channel = 0; channel < 3; ++channel)
                        rgba[channel] = uint8_t((foreground ? 105 : 38) + ((hash >> (channel * 8)) & 127));
                    rgba[3] = 255;
                }
            };
            auto milliseconds = [](Clock::duration elapsed) {
                return std::chrono::duration<double, std::milli>(elapsed).count();
            };
            Json frame_trace = Json::array();
            Clock::time_point window_start{};
            unsigned staged_ahead = 0, maximum_outstanding = 0;
            const auto deadline = GetTickCount64() + 180000;
            while (delivered < frames && GetTickCount64() < deadline) {
                if (host.phase() == Phase::Failed) throw std::runtime_error(host.error());
                bool progressed = false;
                if (queue_depth && submitted < frames && outstanding.size() < queue_depth && host.phase() == Phase::Ready) {
                    if (auto slot = host.acquire_cpu_input()) {
                        auto pixels = host.cpu_input_span(*slot);
                        require(pixels.size() == original.size(), "Unexpected input plane size");
                        const auto started = Clock::now();
                        capture(pixels, ++submitted);
                        // Ready/Busy with an older outstanding frame means it
                        // had not been acknowledged when this capture finished.
                        const bool ahead = !outstanding.empty() && host.phase() != Phase::Finished;
                        const unsigned dw = full_depth ? width : width / 2;
                        const unsigned dh = full_depth ? height : height / 2;
                        if (game_depth) {
                            auto depth = host.cpu_depth_span(*slot);
                            for (unsigned y = 0; y < dh; ++y)
                                for (unsigned x = 0; x < dw; ++x)
                                    depth[size_t(y) * dw + x] = .1f + float((x + submitted) % dw) * .8f / dw;
                        }
                        require(host.submit_cpu_input(*slot, submitted, submitted == 1, game_depth, {},
                            game_depth ? feed_pack_depth_dims(dw, dh) : 0), "Ring submit failed");
                        if (ahead && submitted > 12) ++staged_ahead;
                        outstanding.push_back({started});
                        maximum_outstanding = std::max(maximum_outstanding, unsigned(outstanding.size()));
                        progressed = true;
                    }
                } else if (!queue_depth && host.phase() == Phase::Ready && submitted == delivered) {
                    capture(original, ++submitted);
                    require(host.submit(submitted, submitted == 1, original), "Motion benchmark submission failed");
                }
                if (host.phase() == Phase::Finished) {
                    if (queue_depth) {
                        auto lease = host.lease_cpu_output();
                        require(lease.has_value() && !outstanding.empty(), "Missing ring output");
                        require(lease->pixels.size() == original.size(), "Unexpected output plane size");
                        // Retain the final result for output verification. Normal
                        // presentation consumes a leased slot without an owning vector.
                        if (delivered + 1 == frames) output.assign(lease->pixels.begin(), lease->pixels.end());
                        host.release_cpu_output();
                        const auto now = Clock::now();
                        if (delivered == 11) window_start = now;
                        if (delivered >= 12)
                            frame_trace.push_back({{"frame", delivered + 1},
                                {"completion_ms", milliseconds(now - window_start)},
                                {"capture_to_delivery_ms", milliseconds(now - outstanding.front().capture)},
                                {"helper_roundtrip_ms", host.roundtrip()}});
                        outstanding.pop_front();
                        progressed = true;
                    } else output = host.take();
                    if (++delivered > 12) roundtrip.push_back(host.roundtrip());
                }
                if (!progressed && queue_depth) {
                    LARGE_INTEGER interval{};
                    interval.QuadPart = -1000; // 100 microseconds in 100 ns units.
                    require(SetWaitableTimer(poll_timer.value, &interval, 0, nullptr, nullptr, FALSE) &&
                        WaitForSingleObject(poll_timer.value, INFINITE) == WAIT_OBJECT_0,
                        "Ring benchmark timer failed");
                } else if (!progressed) Sleep(1);
            }
            if (queue_depth) capture(original, frames);
            require(delivered == frames && output.size() == original.size() && output != original,
                "Motion benchmark did not complete neural rendering");
            require(host.ui_action(phoenix_ui::Action::Marker), "Unable to request final helper snapshot");
            Json status;
            for (const auto until = GetTickCount64() + 5000; GetTickCount64() < until; Sleep(5)) {
                status = host.ui_snapshot();
                if (status.value("markers", uint64_t(0)) > 0 && status.contains("helper_motion")) break;
            }
            const auto motion = status.at("helper_motion").at("streams").at(0);
            require(motion.at("frames").get<unsigned>() == frames, "Final motion snapshot is stale");
            require(motion.at("motion_source").get<int>() == settings.pipeline.motion_source, motion.dump().c_str());
            require(settings.pipeline.motion_source == int(phoenix_pipeline::MotionSource::Zero) ||
                motion.at("estimated_frames").get<unsigned>() >= frames - 8, motion.dump().c_str());
            std::sort(roundtrip.begin(), roundtrip.end());
            Json result{{"passed", true}, {"source", settings.pipeline.motion_source}, {"frames", delivered},
                {"size", {width,height}}, {"analysis_height", settings.pipeline.flow_height}, {"client_bits", sizeof(void*) * 8},
                {"motion", motion}, {"helper_timing", status.at("helper_timing")}, {"cpu_transport", status.at("cpu_transport")},
                {"output_sha256", phoenix_model::weight_digest(output.data(), output.size())},
                {"evaluate_stages_ms", status.value("evaluate_stages_ms", Json::object())},
                {"cpu_prefetch", status.value("cpu_prefetch", Json::object())},
                {"legacy_depth", status.value("legacy_depth", Json::object())},
                {"gpu_timing", status.value("gpu_timing", Json::object())},
                {"roundtrip_median_ms", roundtrip[roundtrip.size()/2]},
                {"roundtrip_p95_ms", roundtrip[(roundtrip.size()*95-1)/100]}};
            if (queue_depth) {
                result["ring"] = {{"requested_depth", queue_depth}, {"max_outstanding", maximum_outstanding},
                    {"staged_ahead", staged_ahead}, {"frames", std::move(frame_trace)}};
            }
            std::cout << result.dump() << '\n';
            return 0;
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
