// Render the real shared widgets in an ImGui context, without a game or GPU.
// Link matching external ImGui 1.92.5 sources for this optional UI test only.
#ifndef IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#endif
#include "../../src/integration/reshade/shared_panel.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <iostream>
#include <stdexcept>
namespace {
ImRect clear_timing_button;
}
// Minimal test-engine hooks identify real widget geometry. The test then feeds
// ordinary ImGui events; production code needs no testing callbacks or hooks.
void ImGuiTestEngineHook_ItemAdd(ImGuiContext *, ImGuiID, const ImRect &,
                                 const ImGuiLastItemData *) {}
void ImGuiTestEngineHook_ItemInfo(ImGuiContext *context, ImGuiID, const char *label,
                                  ImGuiItemStatusFlags) {
    if (std::strcmp(label, "Clear GPU timing") == 0)
        clear_timing_button = context->LastItemData.Rect;
}
void ImGuiTestEngineHook_Log(ImGuiContext *, const char *, ...) {}
const char *ImGuiTestEngine_FindItemDebugLabel(ImGuiContext *, ImGuiID) { return ""; }
int run_tests() {
    if (!IMGUI_CHECKVERSION())
        throw std::runtime_error("ImGui implementation/header structure layouts do not match");
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {1600, 6000};
    io.DeltaTime = 1.f / 60;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    io.Fonts->SetTexID(ImTextureID(1));
    phoenix_ui::Settings settings;
    std::array<phoenix_ui::Settings, 4> presets;
    std::array<int, 6> hotkeys{};
    int capture_key = -1;
    uint64_t deadline = 0, markers = 0;
    phoenix_model_tools::Task picker;
    std::vector<phoenix_ui::phoenix::ExecutionDevice> devices{{"native", "Native RTX"},
                                                              {"cpu", "CPU"}};
    std::string error;
    unsigned rendered = 0, actions = 0, validation_polls = 0;
    for (bool bridge : {false, true})
        for (int mode = 0; mode < 4; ++mode)
            for (int variant = 0; variant < 3; ++variant) {
                settings = {};
                settings.mode = mode;
                settings.execution_device = variant == 2 ? "cpu" : "native";
                settings.cpu_threads = variant == 2 ? 1 : 0;
                settings.resolution_mode = 1;
                settings.resolution_aspect = 5;
                settings.experimental = settings.automatic_budget = true;
                settings.controls.enabled = settings.controls.advanced = settings.native_options =
                    true;
                settings.comparison = variant + 1;
                settings.source_a = 0;
                settings.source_b = 2;
                settings.model_source = variant == 1   ? "model_folder"
                                        : variant == 2 ? "nvidia_dll"
                                                       : "configured";
                phoenix_ui::Context context{
                    settings, settings.controls, presets, hotkeys, capture_key, deadline,
                    picker,   devices,           error,   markers};
                context.real_guides = context.temporal_history = context.comparison_views = !bridge;
                context.fps_budget = !bridge;
                context.model_validation.poll = [&] {
                    ++validation_polls;
                    return phoenix_model_tools::Result{};
                };
                context.model_validation.validate = [](const auto &, const auto &) {};
                context.save = context.reload = context.marker = [&] { ++actions; };
                context.capture = [&](unsigned) { ++actions; };
                context.action = [&](phoenix_ui::Action) { ++actions; };
                context.direct_runner =
                    phoenix_ui::Preview{{1920, 1080, 1280, 720, 1280, 720}, "Spatial"};
                context.live_timing.emplace();
                context.live_timing->data["active_settings"] = nlohmann::json::object();
                ImGui::NewFrame();
                ImGui::SetNextWindowPos({0, 0});
                ImGui::SetNextWindowSize({1600, 6000});
                ImGui::Begin("Shared addon UI");
                for (const char *section :
                     {"History policy", "DLSS5 Phoenix experiments",
                      "Transformer block repetitions", "Hotkeys", "Settings / logging",
                      "DLSS5 Phoenix model", "GPU stage breakdown"})
                    ImGui::GetStateStorage()->SetInt(ImGui::GetID(section), 1);
                phoenix_ui::draw_controls(context);
                phoenix_ui::draw_timing(context);
                ImGui::End();
                ImGui::Render();
                if (ImGui::GetDrawData()->TotalVtxCount <= 0 || !settings.valid())
                    throw std::runtime_error("Shared UI failed to render or corrupted settings");
                ++rendered;
            }
    if (actions || validation_polls != 12)
        throw std::runtime_error("Drawing triggered a runtime action or omitted model controls");
    // Policy and adapter actions are tested independently of a game. Passive
    // vertex generation cannot establish that Save merely persists a preset.
    unsigned policy_checks = 0, persisted = 0, applied = 0;
    const auto check = [&](bool value) {
        if (!value)
            throw std::runtime_error("Shared UI policy/action contract failed at check " +
                                     std::to_string(policy_checks + 1));
        ++policy_checks;
    };
    settings = {};
    hotkeys = {{0x78, 0x79, 0x7a, 0x77, 0x76, 0}};
    const auto original_keys = hotkeys;
    using phoenix_ui::BindingResult;
    check(phoenix_ui::capture_binding(hotkeys, 0, 0x1b) == BindingResult::Cancelled);
    check(hotkeys == original_keys);
    for (unsigned key : {1u, 7u, 0x24u, 0x7bu}) {
        check(phoenix_ui::capture_binding(hotkeys, 0, key) == BindingResult::Reserved);
        check(hotkeys == original_keys);
    }
    check(phoenix_ui::capture_binding(hotkeys, 0, 0x79) == BindingResult::Duplicate);
    check(hotkeys == original_keys);
    check(phoenix_ui::capture_binding(hotkeys, -1, 0x75) == BindingResult::Invalid);
    check(phoenix_ui::capture_binding(hotkeys, 6, 0x75) == BindingResult::Invalid);
    check(phoenix_ui::capture_binding(hotkeys, 0, 256) == BindingResult::Invalid);
    check(phoenix_ui::capture_binding(hotkeys, 0, 0x75) == BindingResult::Accepted);
    check(hotkeys[0] == 0x75);
    std::array<int, 6> imported{{0x78, 0x78, 0x24, -1, 256, 0x1b}};
    check(phoenix_ui::sanitize_bindings(imported) == 5);
    check(imported[0] == 0x78 && imported[1] == 0 && imported[5] == 0);
    check(phoenix_ui::sanitize_bindings(imported) == 0);
    phoenix_ui::Context interaction{
        settings, settings.controls, presets, hotkeys, capture_key, deadline,
        picker, devices, error, markers};
    interaction.persist = [&] { ++persisted; };
    interaction.save = [&] { ++applied; };
    interaction.action = [&](phoenix_ui::Action action) {
        check(action == phoenix_ui::Action::ClearTiming);
        ++actions;
    };
    interaction.status = "Helper running";
    interaction.timing_provenance = "Helper GPU (D3D12 companion)";
    interaction.inference_process = true;
    interaction.telemetry_available = false;
    interaction.model_validation.poll = [] { return phoenix_model_tools::Result{}; };
    interaction.live_timing.emplace();
    interaction.live_timing->data["active_settings"] = nlohmann::json::object();
    settings.transformer_blocks = 5;
    settings.experimental = true;
    check(phoenix_ui::store_preset(interaction, 0));
    check(persisted == 1 && applied == 0 && presets[0].transformer_blocks == 5);
    settings.transformer_blocks = 8;
    check(phoenix_ui::load_preset(interaction, 0));
    check(settings.transformer_blocks == 5 && persisted == 1 && applied == 0);
    auto before_invalid = settings;
    presets[1].transformer_blocks = 0;
    check(!phoenix_ui::load_preset(interaction, 1));
    check(settings.transformer_blocks == before_invalid.transformer_blocks);
    check(!phoenix_ui::store_preset(interaction, 4));
    phoenix_ui::clear_binding(interaction, 0);
    check(hotkeys[0] == 0 && persisted == 2 && applied == 0);
    // The direct adapter may intentionally use the same persistence callback.
    interaction.persist = {};
    check(phoenix_ui::store_preset(interaction, 0));
    check(applied == 1);

    // Click the actual Clear GPU timing widget, including press and release
    // frames. This tests the shared view dispatch, rather than fake UI calls.
    io.ConfigInputTrickleEventQueue = false;
    ImGui::GetCurrentContext()->TestEngineHookItems = true;
    const auto draw_interaction = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({1200, 6000});
        ImGui::SetNextWindowFocus();
        ImGui::Begin("Shared UI action checks");
        phoenix_ui::draw_timing(interaction);
        ImGui::End();
        ImGui::Render();
    };
    draw_interaction();
    // A newly appearing ImGui window can use its first frame for layout.
    draw_interaction();
    check(clear_timing_button.GetWidth() > 0);
    const auto button_center = clear_timing_button.GetCenter();
    io.AddMousePosEvent(button_center.x, button_center.y);
    io.AddMouseButtonEvent(0, true);
    draw_interaction();
    check(actions == 0);
    io.AddMouseButtonEvent(0, false);
    draw_interaction();
    check(actions == 1);
    ImGui::NewFrame();
    ImGui::Begin("Shared compact panel checks");
    check(!phoenix_ui::draw_compact(interaction));
    ImGui::End();
    ImGui::Render();
    ImGui::DestroyContext();
    std::cout << nlohmann::json({{"passed", true},
                                 {"bits", sizeof(void *) * 8},
                                 {"rendered_panels", rendered},
                                 {"validation_controls_drawn", validation_polls},
                                 {"runtime_actions_while_drawing", 0},
                                 {"policy_action_checks", policy_checks},
                                 {"clicked_runtime_actions", actions}})
                     .dump()
              << '\n';
    return 0;
}
int main() {
    try {
        return run_tests();
    } catch (const std::exception &failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
