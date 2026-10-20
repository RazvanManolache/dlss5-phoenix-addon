#include "../../src/native/support/direct.h"
#include "../../src/native/support/panel_visibility.h"
#include "../../src/native/support/control_bridge.h"
#include "../../src/native/support/worker_timing.h"
#include "../../src/native/support/timing_archive.h"
#include <iostream>
using namespace phoenix_addon;
int main() {
    try {
        unsigned accepted_commands = 0;
        auto test = [&](bool value) {
            check(value, "Direct policy assertion");
            ++accepted_commands;
        };
        {
            PhoenixLiveControls controls;Controls host;host.intensity=.7f;host.style=1;
            test(controls.apply(host).intensity==.7f&&controls.apply(host).style==1);
            controls.appearance=true;controls.intensity=.3f;controls.style=2;
            test(controls.apply(host).intensity==.3f&&controls.apply(host).style==2);
            controls.intensity=3;test(!controls.valid());controls.intensity=.3f;
            phoenix::Settings original,copy;original.controls=controls;
            test(phoenix::copy_live_tuning(copy,original)&&copy.controls.appearance&&copy.controls.style==2);
            unsigned applied=0,ignored=0;
            test(phoenix::merge_overrides("[Phoenix]\nPhoenixIntensity=0.8\nPhoenixStyle=1\nPhoenixAppearance=true\n",copy,applied,ignored)&&
                applied==3&&ignored==0&&copy.controls.intensity==.8f&&copy.controls.style==1);
            phoenix::reset_neural_section(copy);test(!copy.controls.appearance&&copy.controls.intensity==1&&copy.controls.style==0);
        }
        {
            using namespace phoenix;
            auto normal = choose_resolution(1920, 1080, 0, 0, 0, 16, 9, true);
            test(!normal.scaled() && normal.image_width == 1920);
            auto quality = choose_resolution(1920, 1080, 1, 0, 0, 16, 9, true);
            test(quality.width == 1280 && quality.height == 720 && !quality.fallback);
            auto performance = choose_resolution(1920, 1080, 1, 1, 0, 16, 9, true);
            test(performance.width == 960 && performance.height == 540 &&
                 performance.image_height == 540 && performance.top == 0);
            test(std::abs(performance.motion_x() - 0.5f) < 1e-6f);
            test(std::abs(performance.motion_y() - 0.5f) < 1e-6f);
            auto small_resolution = choose_resolution(640, 384, 1, 0, 0, 16, 9, true);
            test(!small_resolution.fallback && small_resolution.scaled());
            auto odd = choose_resolution(1919, 1079, 0, 0, 0, 16, 9, true);
            test(odd.width == 1919 && odd.height == 1079 && !odd.scaled());
            auto tiny = choose_resolution(1, 1, 1, 0, 0, 16, 9, true);
            test(tiny.fallback && !tiny.scaled());
            for (const auto extent : {std::array<unsigned, 2>{2560, 1440},
                                      {3840, 2160}, {1080, 1920}, {3440, 1440}, {801, 603}})
                for (int quality_index = 0; quality_index < 3; ++quality_index) {
                    auto scaled = choose_resolution(extent[0], extent[1], 1, quality_index,
                                                     0, 16, 9, true);
                    test(scaled.scaled() && !scaled.fallback && scaled.width && scaled.height);
                    test(scaled.left + scaled.image_width <= scaled.width &&
                         scaled.top + scaled.image_height <= scaled.height);
                }
            auto portrait = choose_resolution(1920, 1080, 1, 0, 5, 9, 16, false);
            test(portrait.width < portrait.height && portrait.top > 0);
            for (int aspect = 0; aspect < 6; ++aspect)
                for (bool native : {false, true}) {
                    auto geometry = choose_resolution(2560, 1080, 1, 0, aspect, 7, 5, native);
                    test(geometry.left + geometry.image_width <= geometry.width &&
                         geometry.top + geometry.image_height <= geometry.height);
                    test(std::abs(double(geometry.image_width) / geometry.image_height -
                                  2560.0 / 1080) < .01);
                }
            Settings saved;
            saved.upscaler = 3;
            test(!saved.valid());
            saved.upscaler = 2;
            saved.resolution_mode = 1;
            saved.resolution_aspect = 5;
            saved.resolution_ratio_height = 0;
            test(!saved.valid());
            saved.resolution_ratio_height = 10;
            test(saved.valid());
            Json values;
            visit_settings(saved, [&](const char* name, auto& value) { values[name] = value; });
            test(values.at("Upscaler") == 2 && values.at("ResolutionMode") == 1 &&
                 values.at("ResolutionRatioHeight") == 10);
        }
        {
            phoenix::TimingArchive archive;
            phoenix::TimingArchive::Key gpu{3, 8, 1920, 1080, "{\"ExecutionDevice\":\"native\"}"};
            phoenix::TimingArchive::Key cpu{3, 8, 1920, 1080, "{\"ExecutionDevice\":\"cpu\"}"};
            archive.sample(gpu, archive.epoch(), 100, 5, 4, 0, 1, {});
            archive.sample(cpu, archive.epoch(), 200, 2, 0, 0, 0, {});
            archive.sample(gpu, archive.epoch(), 300, 7, 6, 0, 1, {});
            auto rows = archive.rows(300);
            test(rows.size() == 2 && archive.history_count() == 3);
            bool found = false;
            for (const auto& row : rows)
                if (row.at("execution_device") == "native") {
                    test(row.at("route").at("samples") == 2 && row.at("route").at("mean_ms") == 6);
                    found = true;
                }
            test(found);
            const auto old_epoch = archive.epoch();
            archive.clear();
            archive.sample(gpu, old_epoch, 400, 5, 4, 0, 1, {});
            test(archive.rows(400).empty() && archive.history_count() == 0);
            archive.sample(gpu, archive.epoch(), 400, std::nan(""), 4, 0, 1, {});
            test(archive.rows(400).empty());
            for (unsigned index = 0; index < 160; ++index) {
                auto key = gpu;
                std::get<2>(key) = index;
                archive.sample(key, archive.epoch(), 500 + index, 5, 4, 0, 1, {});
            }
            test(archive.rows(700).size() == 128 && archive.history_count() == 120);
        }
        {
            unsigned file_checks = 0;
            bool marker_present = false;
            auto file_present = [&] {
                ++file_checks;
                return marker_present;
            };
            test(!phoenix::direct::recording_allowed(false, false, file_present));
            test(!phoenix::direct::recording_allowed(false, true, file_present));
            test(phoenix::direct::recording_allowed(true, false, file_present));
            test(file_checks == 0); // Default gameplay never checks the readiness file.
            test(!phoenix::direct::recording_allowed(true, true, file_present));
            marker_present = true;
            test(phoenix::direct::recording_allowed(true, true, file_present));
            marker_present = false;
            test(!phoenix::direct::recording_allowed(true, true, file_present));
            test(file_checks == 3); // Explicit test control still observes removal.
        }
        {
            phoenix::WorkerTiming timing;
            Json execution{{"worker_pid", 10},
                           {"state", "inference_running"},
                           {"last_result", Json::object()}};
            timing.sample("cpu", "CPU", execution);
            test(timing.report().at("rows").at(0).at("completed") == 0);
            execution["last_result"] = {
                {"ok", true}, {"frame_id", 1}, {"generation", 2}, {"elapsed_ms", 20.0}};
            timing.sample("cpu", "CPU", execution);
            timing.sample("cpu", "CPU", execution);
            test(timing.report().at("rows").at(0).at("completed") == 1);
            test(timing.report().at("rows").at(0).at("prepare").at("samples") == 0);
            execution["worker_pid"] = 11;
            timing.sample("cpu", "CPU", execution);
            test(timing.report().at("rows").at(0).at("completed") == 2);
            timing.sample("amd/matrix", "AMD", execution);
            test(timing.report().at("rows").size() == 2);
            execution["last_result"] = {{"ok", false}, {"frame_id", 2}, {"error", "test"}};
            timing.sample("cpu", "CPU", execution);
            test(timing.report().at("rows").at(1).at("failed") == 1);
            timing.clear();
            test(timing.report().at("rows").empty());
        }
        {
            const phoenix::Settings defaults;
            Json base{{"schema", "phoenix-control/1"}, {"session", "test"}, {"pid", 77},
                      {"expires_tick_ms", 2000},       {"id", "unit-1"},    {"op", "status"}};
            auto validate = [&](const Json& inner_index) {
                return phoenix::control_bridge::validate(inner_index, "test", 77, 1000, defaults);
            };
            auto reject = [&](Json inner_index) {
                try {
                    validate(inner_index);
                    return false;
                } catch (...) {
                    return true;
                }
            };
            test(validate(base).op == "status");
            for (const auto* operation_name : {"reset_history", "screenshot"}) {
                auto inner_index = base;
                inner_index["op"] = operation_name;
                test(validate(inner_index).op == operation_name);
            }
            for (const auto* key : {"session", "pid", "schema", "expires_tick_ms", "id", "op"}) {
                auto inner_index = base;
                inner_index.erase(key);
                test(reject(inner_index));
            }
            auto inner_index = base;
            inner_index["session"] = "old";
            test(reject(inner_index));
            inner_index = base;
            inner_index["pid"] = 78;
            test(reject(inner_index));
            for (auto timeout_ms : {-1, 1000, 31001}) {
                inner_index = base;
                inner_index["expires_tick_ms"] = timeout_ms;
                test(reject(inner_index));
            }
            inner_index = base;
            inner_index["surprise"] = true;
            test(reject(inner_index));
            inner_index = base;
            inner_index["id"] = "../unsafe";
            test(reject(inner_index));
            inner_index = base;
            inner_index["op"] = "shell";
            test(reject(inner_index));
            inner_index = base;
            inner_index["op"] = "capture";
            inner_index["frames"] = 8;
            test(validate(inner_index).frames == 8);
            for (auto count : {-1, 0, 9}) {
                inner_index["frames"] = count;
                test(reject(inner_index));
            }
            inner_index["frames"] = 1.5;
            test(reject(inner_index));
            inner_index = base;
            inner_index["op"] = "set";
            inner_index["settings"] = {{"Block1Repeats", 0}};
            test(validate(inner_index).settings.block_repeats[0] == 0);
            test(defaults.block_repeats[0] == 1);
            for (int count : {0, 1, 8, 16, 32, 65535}) {
                auto threads = inner_index;
                threads["settings"] = {{"CpuThreads", count}};
                test(validate(threads).settings.cpu_threads == count);
            }
            {
                auto model = inner_index;
                model["settings"] = {{"ModelSource", "model_folder"},
                                     {"ModelFolder", "E:/LocalModels/example"}};
                const auto selected = validate(model).settings;
                test(selected.model_source == "model_folder" &&
                     selected.model_folder == "E:/LocalModels/example");
                for (const auto& source : {Json("unknown"), Json(1), Json(true)}) {
                    model["settings"] = {{"ModelSource", source}};
                    test(reject(model));
                }
                for (const auto& path :
                     {Json(std::string(4096, 'x')), Json(std::string("a\0b", 3)), Json(1)}) {
                    model["settings"] = {{"ModelFolder", path}};
                    test(reject(model));
                }
            }
            for (const auto& count : {Json(-1), Json(65536), Json(1.5), Json("32"), Json(true)}) {
                auto threads = inner_index;
                threads["settings"] = {{"CpuThreads", count}};
                test(reject(threads));
            }
            {
                auto device = inner_index;
                device["settings"] = {{"ExecutionDevice", "cpu"}};
                test(validate(device).settings.execution_device == "cpu");
                device["settings"] = {
                    {"ExecutionDevice", "cuda:GPU-01234567-89ab-cdef-0123-456789abcdef"}};
                test(validate(device).settings.valid());
                device["settings"] = {
                    {"ExecutionDevice", "vulkan:0123456789abcdef0123456789abcdef"}};
                test(validate(device).settings.valid());
                for (const auto& device_id :
                     {Json("cuda:0"), Json("cuda:GPU-bad"), Json("vulkan:0"),
                      Json("vulkan:0000000074000000000000000000000G"),
                      Json("vulkan:0123456789abcdef0123456789abcdef0"), Json("native "), Json(1),
                      Json(true)}) {
                    device["settings"] = {{"ExecutionDevice", device_id}};
                    test(reject(device));
                }
                device["settings"] = {{"OffloadAgeFrames", 6000}, {"OffloadAgeMs", 120000}};
                test(validate(device).settings.valid());
                device["settings"] = {{"OffloadAgeMs", 120001}};
                test(reject(device));
                device["settings"] = {{"OffloadAgeMs", 0}};
                test(reject(device));
                device["settings"] = {{"OffloadAgeFrames", 6001}};
                test(reject(device));
            }

            for (const auto* engine : {"auto", "portable", "matrix", "fp8"}) {
                auto selection = inner_index;
                selection["settings"] = {{"WorkerEngine", engine}};
                test(validate(selection).settings.worker_engine == engine);
            }
            for (const auto& engine : {Json("invalid"), Json("FP8"), Json(1), Json(true)}) {
                auto selection = inner_index;
                selection["settings"] = {{"WorkerEngine", engine}};
                test(reject(selection));
            }
            for (const auto& value : {Json(-1), Json(5), Json(1.5), Json("0"), Json(true)}) {
                inner_index["settings"]["Block1Repeats"] = value;
                test(reject(inner_index));
            }
            for (const auto* key : {"Unknown", "ScreenshotRequest", "CaptureBurst"}) {
                inner_index["settings"] = {{key, 1}};
                test(reject(inner_index));
            }
            inner_index = base;
            inner_index["settings"] = {{"Backend", 3}};
            test(reject(inner_index));
        }
        using namespace phoenix::direct;
        test(extent_supported(1920, 1080));
        test(extent_supported(640, 384));
        test(extent_supported(1920, 1079));
        test(extent_supported(1, 1));
        test(extent_supported(16384, 16384));
        test(!extent_supported(16385, 1080));
        test(!extent_supported(0, 0));
        test(source_format(DXGI_FORMAT_R11G11B10_FLOAT));
        test(!source_format(DXGI_FORMAT_R8G8B8A8_UNORM));
        test(motion_format(DXGI_FORMAT_R16G16_FLOAT));
        test(!motion_format(DXGI_FORMAT_D32_FLOAT));
        phoenix::Settings settings;
        settings.mode = phoenix::split;
        for (int mode = 0; mode < 6; ++mode)
            for (int source_a = 0; source_a < 3; ++source_a)
                for (int source_b = 0; source_b < 3; ++source_b) {
                    settings.comparison = mode;
                    settings.source_a = source_a;
                    settings.source_b = source_b;
                    test(comparison_supported(settings, false) ==
                         (mode != 0 && source_a != 1 && source_b != 1));
                    test(comparison_supported(settings, true) ==
                         (mode != 0 && (mode == 1 || (source_a != 1 && source_b != 1))));
                }
        test(std::string(startup_status(true, false, false, false, false, false, false))
                 .find("Not armed") != std::string::npos);
        test(std::string(startup_status(true, true, false, false, false, false, false))
                 .find("Paused") != std::string::npos);
        test(std::string(startup_status(true, true, true, true, true, false, false))
                 .find("no NR provider") != std::string::npos);
        Resources resources(reinterpret_cast<ID3D12Resource*>(0x1000),
                            reinterpret_cast<ID3D12Resource*>(0x2000));
        ID3D12Resource* resource = nullptr;
        test(NVSDK_NGX_SUCCEED(resources.Get("DLSSNR.Color", &resource)) &&
             uintptr_t(resource) == 0x1000);
        test(NVSDK_NGX_FAILED(resources.Get("DLSSNR.Depth", &resource)) && resource == nullptr);
        void* untyped = nullptr;
        test(NVSDK_NGX_SUCCEED(resources.Get("DLSSNR.MVec", &untyped)) &&
             uintptr_t(untyped) == 0x2000);
        test(Pass::compile(meter_shader)->GetBufferSize() > 0);
        for (auto* shader : {normalize_shader, output_shader})
            test(Pass::compile((std::string(bridge_common) + shader).c_str())->GetBufferSize() > 0);
        // Wipe furniture follows the displayed route, not the saved preset:
        // a stale two-way/three-way preset must not draw over fullscreen
        // Input, NVIDIA DLL, or DLSS5 Phoenix frames after leaving Comparison.
        for (int route : {phoenix::input, phoenix::nvidia, phoenix::ours}) {
            phoenix::Settings stale;
            stale.mode = route;
            stale.comparison = 1;
            test(!phoenix::PanelVisibility{stale}.wipe_hud());
            stale.comparison = 0;
            test(!phoenix::PanelVisibility{stale}.wipe_hud());
        }
        for (int preset = 0; preset < 6; ++preset) {
            phoenix::Settings viewing;
            viewing.mode = phoenix::split;
            viewing.comparison = preset;
            test(phoenix::PanelVisibility{viewing}.wipe_hud());
        }
        // NR wipe staging: NVIDIA source 1 must reach a shader-read slot,
        // never the wipe destination. Slot 2 reads the network result texture;
        // staging anywhere else shows DLSS5 Phoenix mislabeled as NVIDIA.
        auto stage = [&](int mode, int comparison, int source_a, int source_b,
                         bool have_nvidia_output) {
            return nr_wipe_stage(mode, comparison, source_a, source_b, have_nvidia_output);
        };
        for (int mode : {phoenix::input, phoenix::nvidia, phoenix::ours})
            test(!stage(mode, 1, 0, 1, true).stage);
        test(!stage(phoenix::split, 1, 0, 1, false).stage);
        test(!stage(phoenix::split, 0, 0, 1, true).stage);
        test(!stage(phoenix::split, 1, 0, 2, true).stage);
        test(!stage(phoenix::split, 1, 2, 0, true).stage);
        {
            auto wipe_plan = stage(phoenix::split, 1, 0, 1, true);
            test(wipe_plan.stage && !wipe_plan.to_input && wipe_plan.effective_source_a == 0 &&
                 wipe_plan.effective_source_b == 2 && wipe_plan.resolve == 2);
        }
        {
            auto wipe_plan = stage(phoenix::split, 1, 1, 0, true);
            test(wipe_plan.stage && !wipe_plan.to_input && wipe_plan.effective_source_a == 2 &&
                 wipe_plan.effective_source_b == 0 && wipe_plan.resolve == 1);
        }
        {
            auto wipe_plan = stage(phoenix::split, 1, 1, 1, true);
            test(wipe_plan.stage && wipe_plan.to_input && wipe_plan.effective_source_a == 0 &&
                 wipe_plan.effective_source_b == 0 && wipe_plan.resolve == 3);
        }
        {
            auto wipe_plan = stage(phoenix::split, 1, 1, 2, true);
            test(wipe_plan.stage && wipe_plan.to_input && wipe_plan.effective_source_a == 0 &&
                 wipe_plan.effective_source_b == 2 && wipe_plan.resolve == 1);
        }
        {
            auto wipe_plan = stage(phoenix::split, 1, 2, 1, true);
            test(wipe_plan.stage && wipe_plan.to_input && wipe_plan.effective_source_a == 2 &&
                 wipe_plan.effective_source_b == 0 && wipe_plan.resolve == 2);
        }
        for (int preset = 1; preset < 6; ++preset) {
            auto wipe_plan = stage(phoenix::split, preset, 0, 1, true);
            test(wipe_plan.stage && !wipe_plan.to_input && wipe_plan.effective_source_a == 0 &&
                 wipe_plan.effective_source_b == 2 && wipe_plan.resolve == 2);
        }
        // Resolve flags mark exactly the sides reading the staged NR slot and
        // never a DLSS5 Phoenix/input side: the resolve pass transfers the NR answer
        // onto those sides.
        for (int source_a = 0; source_a < 3; ++source_a)
            for (int source_b = 0; source_b < 3; ++source_b) {
                auto wipe_plan = stage(phoenix::split, 1, source_a, source_b, true);
                if (source_a != 1 && source_b != 1) {
                    test(!wipe_plan.stage && wipe_plan.resolve == 0);
                    continue;
                }
                test(wipe_plan.stage);
                const unsigned slot = wipe_plan.to_input ? 0u : 2u;
                test(wipe_plan.resolve == ((wipe_plan.effective_source_a == slot ? 1u : 0u) |
                                           (wipe_plan.effective_source_b == slot ? 2u : 0u)));
                test((wipe_plan.resolve & 1) == 0 || wipe_plan.effective_source_a == 0 ||
                     wipe_plan.effective_source_a == 2);
                if (source_a != 1)
                    test((wipe_plan.resolve & 1) == 0);
                if (source_b != 1)
                    test((wipe_plan.resolve & 2) == 0);
            }
        // Exposure flicker detector: steady scenes sit at zero flips, a
        // jitter-period pump accumulates one flip per oscillation, and lost
        // readbacks never read as scene cuts.
        {
            ExposureStability stability;
            test(!stability.push(1.0f) && stability.flips == 0);
            test(!stability.push(1.0f) && stability.flips == 0);
            test(!stability.push(1.01f) && stability.flips == 0);
            test(stability.push(1.05f) && stability.flips == 1);
            test(stability.push(1.0f) && stability.flips == 2);
            test(stability.max_delta_pct >= 4.9f && stability.max_delta_pct < 5.1f);
        }
        {
            ExposureStability stability;
            test(!stability.push(0.0f) && stability.flips == 0);
            test(!stability.push(-1.0f) && stability.flips == 0);
            test(!stability.push(1.0f) && stability.flips == 0);
            test(stability.push(2.0f) && stability.flips == 1);
        }
        // Capture meta round-trips every field, including negatives, and
        // refuses truncated input: the replay probe trusts these blindly.
        {
            CaptureMeta metadata;
            metadata.width = 1920;
            metadata.height = 1080;
            metadata.color_format = 10;
            metadata.motion_format = 16;
            metadata.color_bpp = 8;
            metadata.motion_bpp = 4;
            metadata.frame = 7;
            metadata.blocks = 8;
            metadata.reset = 1;
            metadata.enabled = 1;
            metadata.automask = 1;
            metadata.style = 0;
            metadata.inverted = 1;
            metadata.uicorrection = 0;
            metadata.srgb = 0;
            metadata.unjitter = 1;
            metadata.mode = 3;
            metadata.jitter_x = -0.375f;
            metadata.jitter_y = 0.2778f;
            metadata.motion_x = 1920;
            metadata.motion_y = 1080;
            metadata.intensity = 1;
            metadata.tone = 1;
            metadata.structure = 1;
            metadata.skin = -1;
            CaptureMeta back{};
            test(capture_meta_parse(capture_meta_write(metadata), back));
            {
                auto mapped=metadata;mapped.appearance_flags=31;mapped.appearance_generation=17;
                CaptureMeta parsed;
                test(!capture_meta_parse(capture_meta_write(mapped),parsed,true));
                test(capture_meta_parse(capture_meta_write(mapped),parsed,true,true)&&
                     parsed.appearance_flags==31&&parsed.appearance_generation==17);
                mapped.appearance_flags=32;
                test(!capture_meta_parse(capture_meta_write(mapped),parsed,true,true));
            }
            {
                auto repeated = metadata;
                repeated.refinement_passes = 3;
                repeated.block_repeats[4] = 2;
                CaptureMeta parsed;
                const auto text = capture_meta_write(repeated);
                test(!capture_meta_parse(text, parsed));
                test(capture_meta_parse(text, parsed, true) && parsed.refinement_passes == 3 &&
                     parsed.block_repeats[4] == 2);
                repeated.block_repeats[4] = 0;
                test(!capture_meta_parse(capture_meta_write(repeated), parsed));
                test(capture_meta_parse(capture_meta_write(repeated), parsed, true) &&
                     parsed.block_repeats[4] == 0);
                repeated.refinement_passes = 11;
                test(!capture_meta_parse(capture_meta_write(repeated), parsed, true));
            }
            test(back.width == 1920 && back.height == 1080 && back.color_format == 10 &&
                 back.motion_format == 16);
            test(back.color_bpp == 8 && back.motion_bpp == 4 && back.frame == 7 &&
                 back.blocks == 8);
            test(back.reset == 1 && back.enabled == 1 && back.automask == 1 && back.style == 0);
            test(back.inverted == 1 && back.uicorrection == 0 && back.srgb == 0 &&
                 back.unjitter == 1 && back.mode == 3);
            test(back.jitter_x == metadata.jitter_x && back.jitter_y == metadata.jitter_y);
            test(back.motion_x == 1920 && back.motion_y == 1080);
            test(back.intensity == 1 && back.tone == 1 && back.structure == 1 && back.skin == -1);
            CaptureMeta bad{};
            test(!capture_meta_parse("width=1920\nheight=1080\n", bad));
            test(!capture_meta_parse("", bad));
        }
        {
            ExposureStability stability;
            test(!stability.push(0.4f) && stability.flips == 0);
            test(!stability.push(1.0e7f) && stability.flips == 0);
            test(!stability.push(100.0f) && stability.flips == 0);
            test(stability.max_delta_pct == 0);
            test(!stability.push(0.4f) && stability.flips == 0);
        }
        {
            ExposureStability stability;
            stability.push(1.0f);
            test(!stability.push(1.0f + ExposureStability::kFlipStep * 0.5f) &&
                 stability.flips == 0);
            test(stability.push(1.05f) && stability.flips == 1);
            stability.reset();
            test(stability.flips == 0 && stability.max_delta_pct == 0);
            test(!stability.push(1.0f) && stability.flips == 0);
        }
        std::cout << "{\"passed\":true,\"cases\":" << accepted_commands
                  << ",\"gpu_executed\":false}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
