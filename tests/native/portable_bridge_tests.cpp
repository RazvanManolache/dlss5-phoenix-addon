#include "../../src/integration/compatibility/host_client.h"
#include <iostream>
using namespace phoenix_compat;
unsigned checks = 0;
void verify(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error("Portable bridge contract " + std::to_string(checks));
}
void wait(const std::shared_ptr<HostState>& state, Phase desired) {
    const auto end = GetTickCount64() + 5000;
    while (state->phase() != desired && state->phase() != Phase::Failed && GetTickCount64() < end) Sleep(1);
    if (state->phase() != desired) throw std::runtime_error("Unexpected phase: " + state->error());
}
int wmain(int argc, wchar_t** argv) try {
    verify(argc == 2);
    Settings settings;
    settings.controls.enabled = true;
    settings.controls.tone = .5f;
    auto metadata = portable_metadata(settings, false);
    verify(metadata["controls"]["local_tone"] == .5f && metadata["reset"] == false);
    settings.controls.enabled = false;
    verify(portable_metadata(settings, true)["controls"]["local_tone"] == 1.f);
    settings.controls.enabled = true;
    settings.controls.advanced = true;
    settings.controls.skin = 1.4f;
    settings.refinement_passes = 2;
    settings.experimental = true; settings.transformer_blocks = 4;
    metadata = portable_metadata(settings, true);
    verify(metadata["controls"]["skin_structure"] == 1.4f && metadata["passes"] == 2);
    verify(metadata["repeats"][3] == 1 && metadata["repeats"][4] == 0);
    std::vector<uint8_t> pixels(4*2*4, 80);
    auto color = portable_color(pixels);
    verify(color.size() == 4*2*3 && color[0] == 80/255.f);
    std::vector<float> enhanced(color.size(), .8f);
    settings.mode = phoenix_addon::phoenix::split;
    settings.source_a = 0; settings.source_b = 2; settings.divider = .5f;
    for (int mode = 1; mode <= 5; ++mode) {
        settings.comparison = mode;
        auto result = portable_compose(pixels, enhanced, 4, settings);
        verify(result.size() == pixels.size() && result[3] == 80);
        if (mode == 1) verify(result[0] == 80 && result[8] == 204);
        if (mode == 2 || mode == 5) verify(result[0] == 124);
        if (mode == 3) verify(result[0] == 66);
    }
    settings.source_b = 1;
    bool rejected = false;
    try { portable_compose(pixels, enhanced, 4, settings); } catch (...) { rejected = true; }
    verify(rejected);
    std::vector<float> nvidia(enhanced.size(),.2f);
    settings.source_a=1;settings.source_b=2;settings.comparison=1;
    auto comparison=portable_compose(pixels,enhanced,4,settings,&nvidia);
    verify(comparison[0]==51&&comparison[8]==204&&comparison[3]==80);
    settings.mode=phoenix_addon::phoenix::nvidia;
    verify(portable_compose(pixels,enhanced,4,settings,&nvidia)[0]==51);
    nvidia.pop_back();rejected=false;
    try {portable_compose(pixels,enhanced,4,settings,&nvidia);} catch(...) {rejected=true;}
    verify(rejected);
    Settings native_settings;
    validate_native_settings(native_settings,true);
    for(unsigned failure=0;failure<7;++failure) {
        auto invalid=native_settings;
        if(failure==0) invalid.resolution_mode=1;
        if(failure==1) invalid.host_unjitter=true;
        if(failure==2) invalid.automatic_budget=true;
        if(failure==3) invalid.block_repeats[0]=2;
        if(failure==4) {invalid.experimental=true;invalid.transformer_blocks=4;}
        if(failure==5) invalid.mode=phoenix_addon::phoenix::nvidia;
        if(failure==6) invalid.execution_device="cpu";
        rejected=false;
        try {validate_native_settings(invalid,true);} catch(...) {rejected=true;}
        verify(rejected);
    }
    settings.mode = phoenix_addon::phoenix::ours;
    settings.refinement_passes = 1;
    settings.experimental = false;
    settings.controls.tone = .5f;
    FeedBuild build{};
    build.width = build.output_width = 4;
    build.height = build.output_height = 2;
    build.cpu_bridge = true;
    PortableOptions options{fs::absolute(argv[1]), "cpu", true};
    phoenix_model::WeightSource source;
    source.dll = options.root / "external-test-only.dll";
    source.dll_sha256 = std::string(64, '0');
    auto state = std::make_shared<HostState>(settings);
    state->start(nullptr, source, settings, build, options);
    wait(state, Phase::Ready);
    verify(state->process_id() == 0); // No helper was started.
    verify(state->submit(1, true, pixels));
    verify(!state->submit(2, true, pixels)); // Bounded to one owned frame.
    const auto active = reinterpret_cast<int(*)()>(GetProcAddress(
        GetModuleHandleW(L"phoenix_runtime_cpp.dll"), "phoenix_test_frame_active"));
    verify(active != nullptr);
    const auto entry_deadline = GetTickCount64() + 5000;
    while (!active() && GetTickCount64() < entry_deadline) Sleep(1);
    verify(active() != 0);
    settings.controls.tone = 1.2f;
    verify(state->update_tuning(settings)); // Edit while the previous frame is busy.
    wait(state, Phase::Finished);
    auto first = state->take();
    verify(first[0] == uint8_t(std::lround((80/255.f*.5f+.1f)*255)) && first[3] == 80);
    verify(!state->tuning_current_output());
    verify(state->submit(2, true, pixels));
    wait(state, Phase::Finished);
    auto second = state->take();
    verify(second[0] == uint8_t(std::lround((80/255.f*1.2f+.1f)*255)) && state->tuning_current_output());
    verify(state->ui_snapshot()["processing_location"] == "in_process_portable");
    verify(state->ui_snapshot()["inference_pid"] == GetCurrentProcessId());
    verify(state->ui_action(phoenix_ui::Action::ClearTiming));
    verify(!state->ui_action(phoenix_ui::Action::CaptureInputs));
    settings.controls.tone = 2;
    verify(state->update_tuning(settings));
    verify(state->submit(3, true, pixels));
    const auto end = GetTickCount64() + 5000;
    while (state->phase() != Phase::Failed && GetTickCount64() < end) Sleep(1);
    verify(state->phase() == Phase::Failed && state->error().find("injected inference failure") != std::string::npos);
    state->stop();
    settings.controls.tone = 1;
    auto owner = std::make_shared<HostState>(settings);
    owner->start(nullptr, source, settings, build, options);
    wait(owner, Phase::Ready);
    auto successor = std::make_shared<HostState>(settings);
    successor->start(nullptr, source, settings, build, options);
    Sleep(10);
    verify(successor->phase() == Phase::Starting); // Never collide with the active provider.
    owner->stop();
    wait(successor, Phase::Ready); // Canvas recreation waits asynchronously, not a spurious failure.
    successor->stop();
    wait(successor, Phase::Stopped);
    // A cancellation does not join the model thread or start a helper process.
    settings.controls.tone = 1;
    auto cancelled = std::make_shared<HostState>(settings);
    cancelled->start(nullptr, source, settings, build, options);
    wait(cancelled, Phase::Ready);
    verify(cancelled->submit(1, true, pixels));
    const auto running = reinterpret_cast<int(*)()>(GetProcAddress(
        GetModuleHandleW(L"phoenix_runtime_cpp.dll"), "phoenix_test_frame_active"));
    const auto running_deadline = GetTickCount64() + 5000;
    while (running && !running() && GetTickCount64() < running_deadline) Sleep(1);
    verify(running && running());
    const auto started = GetTickCount64();
    cancelled->stop();
    verify(GetTickCount64() - started < 20);
    wait(cancelled, Phase::Stopped);
    auto missing = std::make_shared<HostState>(settings);
    options.root /= "missing";
    missing->start(nullptr, source, settings, build, options);
    const auto until = GetTickCount64() + 5000;
    while (missing->phase() != Phase::Failed && GetTickCount64() < until) Sleep(1);
    verify(missing->phase() == Phase::Failed && missing->process_id() == 0);
    missing->stop();
    auto native_missing=std::make_shared<HostState>(settings);
    native_missing->start(nullptr,source,settings,build,std::nullopt,NativeOptions{options.root,false});
    const auto native_until=GetTickCount64()+5000;
    while(native_missing->phase()!=Phase::Failed&&GetTickCount64()<native_until) Sleep(1);
    verify(native_missing->phase()==Phase::Failed&&native_missing->error().find("Native x86 processing")!=std::string::npos);
    native_missing->stop();
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"checks\":" << checks << "}\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what(); return 1; }
